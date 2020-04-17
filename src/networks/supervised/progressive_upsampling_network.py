import contextlib
from typing import Tuple, List

import tensorflow as tf
import numpy as np

from src.config import *
from src.dataset import Dataset
from src.networks.network import Network
from src.models import ProgressiveUpsamplingModel


class ProgressiveUpsamplingNetwork(Network):

    def __init__(self):
        model = ProgressiveUpsamplingModel()  # increase the default scaling factor
        super().__init__(model)

    @staticmethod
    # @tf.function
    def _charbonnier_loss(x: tf.Tensor):
        epsilon = tf.constant(1e-3, dtype=tf.float32)
        return tf.sqrt(tf.add(tf.square(x), tf.square(epsilon)))

    @staticmethod
    # @tf.function
    def custom_loss(values: Tuple[List[tf.TensorArray], List[tf.TensorArray]]):
        def map_fn(t):
            y, yl = t
            x = tf.subtract(y, yl)
            return ProgressiveUpsamplingNetwork._charbonnier_loss(x)
        # TODO possibly change to vectorized map
        loss = [tf.reduce_sum(tf.map_fn(map_fn, (values[0][i], values[1][i],), dtype=tf.float32)) for i in range(3)]
        loss = tf.reduce_mean(loss)
        return loss

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        size = self._parse_predict_optionals(x, args, kwargs)
        y_pred = self.model(x)
        # TODO return image batch specified by input arguments, as of now it constantly returns 2x images
        y_pred = y_pred[0].numpy()
        LOGGER.info(f"Predicted images with shape: {y_pred.shape}")
        return y_pred

    # @tf.function
    def _train_step(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = ProgressiveUpsamplingNetwork.custom_loss(([tf.convert_to_tensor(y_i) for y_i in y], y_pred,))
            grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return tf.reduce_sum(loss)

    def train(self, dataset_x, dataset_y, loss_func, epochs, learning_rate):
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                       decay_steps=epochs,
                                                                       decay_rate=0.9,
                                                                       staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # threat a single value as a list regardless
        if isinstance(dataset_y, Dataset):
            dataset_y = [dataset_y]
        # open the datasets
        with contextlib.ExitStack() as stack:
            stack.enter_context(dataset_x)
            for dataset in dataset_y:
                stack.enter_context(dataset)
            iter_x = dataset_x.as_numpy_iterator()
            iter_y = [dataset.as_numpy_iterator() for dataset in dataset_y]
            # train
            e_idx = 0
            train_loss = 0
            start_sec = time.time()
            while e_idx < epochs:
                # process a batch
                try:
                    x = iter_x.next()
                    y = [iter_y[idx].next() for idx in range(len(iter_y))]
                    train_loss += self._train_step(x, y, optimizer)
                except StopIteration:
                    # reset iterators
                    iter_x.reset()
                    for _iter in iter_y:
                        _iter.reset()
                    # update state
                    delta_sec = time.time() - start_sec
                    self.state.epochs += 1
                    self.state.train_loss = train_loss.numpy()
                    self.state.train_time = delta_sec
                    LOGGER.info(f"Epoch: {e_idx} train_loss: {train_loss:.2f}")
                    e_idx += 1
                    train_loss = 0
                    start_sec = time.time()
                    if e_idx > 0 and (e_idx + 1) % 100 == 0:
                        LOGGER.info(f"Saving state after {e_idx + 1} epochs.")
                        self.save_state()

