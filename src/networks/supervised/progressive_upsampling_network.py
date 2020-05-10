import contextlib
from typing import Tuple, List

import tensorflow as tf
import numpy as np

from src.config import *
from src.dataset import Dataset
from src.callbacks import OptimizerCallback, TrainIterationEndCallback
from src.networks.network import Network
from src.models.supervised.progressive_upsampling_model import ProgressiveUpsamplingModel


class ProgressiveUpsamplingNetwork(Network):

    def __init__(self):
        model = ProgressiveUpsamplingModel()  # increase the default scaling factor
        super().__init__(model)

    @staticmethod
    @tf.function
    def _charbonnier_loss(x: tf.Tensor):
        epsilon_square = tf.square(tf.constant(1e-4, dtype=tf.float32))
        return tf.sqrt(tf.square(x) + epsilon_square)

    @staticmethod
    @tf.function
    def custom_loss(values: Tuple[List[np.ndarray], tf.Tensor]):
        y_list, yl_list = values    # (scale_levels, batch, height, width, depth)
        loss = 0
        for y, yl in zip(y_list, yl_list):
            N = tf.constant(len(y), dtype=tf.float32)
            loss += tf.reduce_sum(ProgressiveUpsamplingNetwork._charbonnier_loss(y - yl)) / N
        return loss

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        size = self._parse_predict_optionals(x, args, kwargs)
        y_pred_list = self.model(x)
        for y_pred in y_pred_list:
            if tuple(y_pred.shape[1:3]) == size:
                LOGGER.info(f"Predicted images with shape: {y_pred.shape}")
                return y_pred.numpy()
        LOGGER.warn(f"Couldn't predict.")

    @tf.function
    def _train_step(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = ProgressiveUpsamplingNetwork.custom_loss((y, y_pred,))
            grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, dataset_x, dataset_y, loss_func, epochs, learning_rate=0.001, callbacks=None):
        learning_rate = tf.Variable(learning_rate)      # wrap variable according to callbacks.py:25
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-8)
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
                    # manually update learning rate and call iteration end callbacks
                    for cb in callbacks:
                        if isinstance(cb, OptimizerCallback):
                            learning_rate.assign(cb(self))
                        if isinstance(cb, TrainIterationEndCallback):
                            cb(self)
