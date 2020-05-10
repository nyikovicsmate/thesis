from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from src.callbacks import OptimizerCallback, TrainIterationEndCallback
from src.config import *
from src.dataset import Dataset
from src.models.reinforced.reinforced_model import ReinforcedModel
from src.networks.network import Network
from src.networks.reinforced.state import State


class ReinforcedNetwork(Network):
    """
    Pixelwise asyncronous agent network. (PixelwiseA3CNetwork)
    """

    def __init__(self):
        model = ReinforcedModel()
        super().__init__(model)

    def predict(self, x: np.ndarray, steps_per_episode: int = 5, *args, **kwargs) -> np.ndarray:
        size = self._parse_predict_optionals(x, args, kwargs)  # keep it for the logs
        s_t0_channels = tf.split(tf.convert_to_tensor(x), num_or_size_splits=x.shape[-1], axis=3)
        result = None
        for s_t0 in s_t0_channels:
            for t in range(steps_per_episode):
                # predict the actions and values
                a_t, _ = self.model(s_t0)
                # sample the actions
                sampled_a_t = self._sample_most_probable(a_t)
                # TODO: Eliminate this back-and-forth conversion nonsense by converting State.update() function from NCHW to NHWC (channel first to channel last).
                # convert to NCHW
                s_t0_nchw = tf.transpose(s_t0, perm=[0, 3, 1, 2])
                sampled_a_t_nchw = tf.transpose(sampled_a_t, perm=[0, 3, 1, 2])
                # update the current state/image with the predicted actions
                s_t1 = tf.convert_to_tensor(State.update(s_t0_nchw.numpy(), sampled_a_t_nchw.numpy()), dtype=tf.float32)
                # convert the back to NHWC
                s_t1 = tf.transpose(s_t1, perm=[0, 2, 3, 1])
                s_t0 = s_t1
            result = s_t0 if result == None else tf.concat([result, s_t0], axis=3)
        LOGGER.info(f"Predicted images with shape: {result.shape}")
        return result.numpy()

    # @tf.function
    # TODO: Make it tf.function decorator ready
    def _train_step(self, x, y, optimizer, steps_per_episode: int = 5, discount_factor: float = 0.95):
        s_t0 = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        episode_r = 0
        r = {}  # reward
        V = {}  # expected total rewards from state
        past_action_log_prob = {}
        past_action_entropy = {}
        with tf.GradientTape() as tape:
            for t in range(steps_per_episode):
                # predict the actions and values
                a_t, V_t = self.model(s_t0)
                # sample the actions
                sampled_a_t = self._sample_random(a_t)
                # clip distribution into range to avoid 0 values, which cause problem with calculating logarithm
                a_t = tf.clip_by_value(a_t, 1e-6, 1)
                a_t_log = tf.math.log(a_t)
                past_action_log_prob[t] = self._mylog_prob(a_t_log, sampled_a_t)
                past_action_entropy[t] = self._myentropy(a_t, a_t_log)
                V[t] = V_t
                # TODO: Eliminate this back-and-forth conversion nonsense by converting State.update() function from NCHW to NHWC (channel first to channel last).
                # convert to NCHW
                s_t0_nchw = tf.transpose(s_t0, perm=[0, 3, 1, 2])
                sampled_a_t_nchw = tf.transpose(sampled_a_t, perm=[0, 3, 1, 2])
                # update the current state/image with the predicted actions
                s_t1 = tf.convert_to_tensor(State.update(s_t0_nchw.numpy(), sampled_a_t_nchw.numpy()), dtype=tf.float32)
                # convert the back to NHWC
                s_t1 = tf.transpose(s_t1, perm=[0, 2, 3, 1])
                r_t = self._mse(y, s_t0, s_t1)
                r[t] = tf.cast(r_t, dtype=tf.float32)
                s_t0 = s_t1
                episode_r += tf.reduce_mean(r_t) * tf.math.pow(discount_factor, t)

            R = 0
            actor_loss = 0
            critic_loss = 0
            beta = 1e-2
            for t in reversed(range(steps_per_episode)):
                R *= discount_factor
                R += r[t]
                A = R - V[t]  # advantage
                # Accumulate gradients of policy
                log_prob = past_action_log_prob[t]
                entropy = past_action_entropy[t]

                # Log probability is increased proportionally to advantage
                actor_loss -= log_prob * A
                # Entropy is maximized
                actor_loss -= beta * entropy
                actor_loss *= 0.5  # multiply loss by 0.5 coefficient
                # Accumulate gradients of value function
                critic_loss += (R - V[t]) ** 2 / 2

            total_loss = tf.reduce_mean(actor_loss + critic_loss)
            actor_grads = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))

        return episode_r, total_loss

    def train(self, dataset_x, dataset_y, loss_func, epochs, learning_rate=0.001, callbacks=None):
        assert isinstance(dataset_y, Dataset), "Reinforced network can only be trained with single dataset."
        assert loss_func is None, "Reinforced network uses it's own loss function. Pass it `None`"
        learning_rate = tf.Variable(learning_rate)  # wrap variable according to callbacks.py:25
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-8)
        # open the datasets
        with dataset_x, dataset_y:
            iter_x = dataset_x.as_numpy_iterator()
            iter_y = dataset_y.as_numpy_iterator()
            # train
            e_idx = 0
            while e_idx < epochs:
                # process a batch
                try:
                    start_sec = time.time()
                    x = iter_x.next()
                    y = iter_y.next()
                    episode_r, train_loss = self._train_step(x, y, optimizer)
                    # update state
                    delta_sec = time.time() - start_sec
                    self.state.epochs += 1
                    self.state.train_loss = train_loss.numpy()
                    self.state.train_time = delta_sec
                    LOGGER.info(f"Epoch: {e_idx} episode_r: {episode_r:.2f} train_loss: {train_loss:.2f}")
                    e_idx += 1
                    # manually update learning rate and call iteration end callbacks
                    for cb in callbacks:
                        if isinstance(cb, OptimizerCallback):
                            learning_rate.assign(cb(self))
                        if isinstance(cb, TrainIterationEndCallback):
                            cb(self)
                except StopIteration:
                    # reset iterators
                    iter_x.reset()
                    iter_y.reset()

    @staticmethod
    @tf.function
    def _mse(a, b, c):
        """
        Calculates the mean squared error for image batches given by the formula:
        mse = (a-b)**2 - (a-c)**2
        :param a:
        :param b:
        :param c:
        :return:
        """
        mse = tf.math.square(a - b) * 255
        mse -= tf.math.square(a - c) * 255
        return mse

    @staticmethod
    @tf.function
    def _myentropy(prob, log_prob):
        return tf.stack([- tf.math.reduce_sum(prob * log_prob, axis=3)], axis=3)

    @staticmethod
    @tf.function
    def _mylog_prob(data, indexes):
        """
        Selects elements from a multidimensional array.
        :param data: The 4D actions vector with logarithmic values.
        :param indexes: The indexes to select.
        :return: The selected indices from data eg.: data=[[11, 2], [3, 4]], indexes=[[0],[1]] --> [[11], [4]]
        """
        data_flat = tf.reshape(data, (-1, data.shape[-1]))
        indexes_flat = tf.reshape(indexes, (-1,))
        one_hot_mask = tf.one_hot(indexes_flat, data_flat.shape[-1], on_value=True, off_value=False, dtype=tf.bool)
        output = tf.boolean_mask(data_flat, one_hot_mask)
        return tf.reshape(output, (*data.shape[0:-1], 1))

    # @staticmethod
    # @tf.function
    # def _sample(distribution):
    #     """
    #     Samples the image action distribution returned by the last softmax activation.
    #     :param distribution: output of a softmax activated layer, an array with probability distributions,
    #     usually shaped (batch_size, channels, widht, height, number_of_actions) NCHW!
    #     :return: array shaped of (batch_size, channels, widht, height)
    #     """
    #     distribution = distribution if type(distribution) == np.ndarray else np.array(distribution)
    #     flat_dist = np.reshape(distribution, (-1, distribution.shape[-1]))
    #     flat_indexes = []
    #     for d in flat_dist:
    #         sample_value = np.random.choice(d, p=d)
    #         sample_idx = np.argmax(d == sample_value)
    #         flat_indexes.append(sample_idx)
    #     sample_idxs = np.reshape(flat_indexes, distribution.shape[0:-1])
    #     return sample_idxs

    @staticmethod
    @tf.function
    def _sample_random(distribution):
        """
        Samples the image action distribution returned by the last softmax activation.
        :param distribution: A 4D array with probability distributions shaped (batch_size, height, width, samples)
        :return: The sampled 4D vector shaped of (batch_size, height, width, 1)
        """
        d = tf.reshape(distribution, (-1, distribution.shape[-1]))
        d = tf.math.log(d)
        d = tf.random.categorical(logits=d, num_samples=1)  # draw samples from the categorical distribution
        d = tf.reshape(d, (*distribution.shape[0:-1], 1))
        return d

    @staticmethod
    @tf.function
    def _sample_most_probable(distribution):
        """
        Samples the image action distribution returned by the last softmax activation by returning the
        most probable action indexes from samples.
        :param distribution: A 4D array with probability distributions shaped (batch_size, height, width, samples)
        :return: The sampled 4D vector shaped of (batch_size, height, width, 1)
        """
        d = tf.reshape(distribution, (-1, distribution.shape[-1]))
        d = tf.argmax(d, axis=1)
        d = tf.reshape(d, (*distribution.shape[0:-1], 1))
        return d
