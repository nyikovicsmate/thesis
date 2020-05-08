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
        s_t0 = x
        for t in range(steps_per_episode):
            # image_batch_nchw = tf.transpose(s_t0, perm=[0, 2, 3, 1])
            # predict the actions and values
            a_t, _ = self.model(s_t0)
            # sample the actions
            sampled_a_t = self._sample_most_probable(a_t)
            # update the current state/image with the predicted actions
            s_t1 = State.update(s_t0, sampled_a_t.numpy())
            s_t0 = s_t1
        LOGGER.info(f"Predicted images with shape: {s_t0.shape}")
        return s_t0

    # @tf.function
    def _train_step(self, x, y, optimizer, steps_per_episode: int = 5, discount_factor: float = 0.95):
        s_t0 = tf.convert_to_tensor(x)
        y = tf.transpose(tf.convert_to_tensor(y), perm=[0, 2, 3, 1])
        episode_r = 0
        r = {}  # reward
        V = {}  # expected total rewards from state
        past_action_log_prob = {}
        past_action_entropy = {}
        with tf.GradientTape() as tape:
            for t in range(steps_per_episode):
                # image_batch_nchw = tf.transpose(s_t0, perm=[0, 2, 3, 1])
                # predict the actions and values
                a_t, V_t = self.model(s_t0)
                # sample the actions
                sampled_a_t = self._sample_random(a_t)
                # clip distribution into range to avoid 0 values, which cause problem with calculating logarithm
                a_t = tf.clip_by_value(a_t, 1e-6, 1)
                # convert to NCHW
                a_t = tf.transpose(a_t, perm=[0, 3, 1, 2])
                V_t = tf.transpose(V_t, perm=[0, 3, 1, 2])

                past_action_log_prob[t] = self._mylog_prob(tf.math.log(a_t), sampled_a_t)
                past_action_entropy[t] = self._myentropy(a_t, tf.math.log(a_t))
                V[t] = V_t
                # update the current state/image with the predicted actions
                s_t1 = tf.convert_to_tensor(State.update(s_t0.numpy(), sampled_a_t.numpy()), dtype=tf.float32)
                r_t = self._mse(y, s_t0, s_t1)
                r[t] = tf.cast(r_t, dtype=tf.float32)
                s_t0 = s_t1
                episode_r += np.mean(r_t) * np.power(discount_factor, t)

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
        return tf.stack([- tf.math.reduce_sum(prob * log_prob, axis=1)], axis=1)

    @staticmethod
    @tf.function
    def _mylog_prob(data, indexes):
        """
        Selects elements from a multidimensional array.
        :param data: image_batch
        :param indexes: indices to select
        :return: the selected indices from data eg.: data=[[11, 2], [3, 4]] indexes=[0,1] --> [11, 4]
        """
        n_batch, n_actions, h, w = data.shape
        p_trans = tf.transpose(data, perm=[0, 2, 3, 1])
        p_trans = tf.reshape(p_trans, [-1, n_actions])
        indexes_flat = tf.reshape(indexes, [-1])
        one_hot_mask = tf.one_hot(indexes_flat, p_trans.shape[1], on_value=True, off_value=False, dtype=tf.bool)
        output = tf.boolean_mask(p_trans, one_hot_mask)
        return tf.reshape(output, (n_batch, 1, h, w))

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
        :param distribution: output of a softmax activated layer, an array with probability distributions,
        usually shaped (batch_size, channels, widht, height, number_of_actions) NCHW!
        :return: array shaped of (batch_size, channels, widht, height)
        # TODO: Fix shapes, this supposed to be nhwc
        """
        d = tf.reshape(distribution, (-1, distribution.shape[-1]))
        d = tf.math.log(d)
        d = tf.random.categorical(d, num_samples=1)
        d = tf.reshape(d, distribution.shape[0:-1])
        return d

    @staticmethod
    @tf.function
    def _sample_most_probable(distribution):
        """
        Returns the most probable action index based on the distribution returned by the last softmax activation.
        :param distribution: output of a softmax activated layer, an array with probability distributions,
        usually shaped (batch_size, channels, widht, height, number_of_actions) NCHW!
        :return: array shaped of (batch_size, channels, widht, height)
        # TODO: Fix shapes, this supposed to be nhwc
        """
        d = tf.reshape(distribution, (-1, distribution.shape[-1]))
        d = tf.argmax(d, axis=1)
        d = tf.reshape(d, distribution.shape[0:-1])
        return d
