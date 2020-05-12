import contextlib
from typing import Tuple, List, Optional, Union, Iterable

import tensorflow as tf
import numpy as np

from src.config import *
from src.dataset import Dataset
from src.callbacks import OptimizerCallback, TrainIterationEndCallback
from src.networks.network import Network
from src.models.supervised.progressive_upsampling_model import ProgressiveUpsamplingModel


class ProgressiveUpsamplingNetwork(Network):

    def __init__(self, input_shape: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, 1)):
        model = ProgressiveUpsamplingModel(input_shape=input_shape)
        super().__init__(model)

    @staticmethod
    @tf.function
    def _charbonnier_loss(x: tf.Tensor):
        epsilon_square = tf.square(tf.constant(1e-4, dtype=tf.float32))
        return tf.sqrt(tf.square(x) + epsilon_square)

    @staticmethod
    @tf.function
    def custom_loss(y_true, y_pred):
        # TODO: convert to tf.function decorator ready state
        loss = 0
        for i in range(len(y_true)):
            N = tf.constant(len(y_true[i]), dtype=tf.float32)
            loss += tf.reduce_sum(ProgressiveUpsamplingNetwork._charbonnier_loss(y_true[i] - y_pred[i])) / N
        return loss

    def _predict(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
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
            loss = ProgressiveUpsamplingNetwork.custom_loss(y, y_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def _train(self, x, y, loss_func, epochs, learning_rate, callbacks):
        assert loss_func is None, "Reinforced network uses it's own loss function. Pass it `None`"
        learning_rate = tf.Variable(learning_rate)      # wrap variable according to callbacks.py:25
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-8)
        # train
        for e_idx in range(epochs):
            train_loss = tf.constant(0, dtype=tf.float32)
            start_sec = time.time()
            # process a batch
            for x_b, *y_b in zip(x, *y):
                train_loss += self._train_step(x_b, y_b, optimizer)
            # update state
            delta_sec = time.time() - start_sec
            self.state.epochs += 1
            self.state.train_loss = train_loss.numpy()
            self.state.train_time = delta_sec
            LOGGER.info(f"Epoch: {e_idx} train_loss: {train_loss:.2f}")
            if callbacks:
                # manually update learning rate and call iteration end callbacks
                for cb in callbacks:
                    if isinstance(cb, OptimizerCallback):
                        learning_rate.assign(cb(self))
                    if isinstance(cb, TrainIterationEndCallback):
                        cb(self)
