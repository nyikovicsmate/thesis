import contextlib
from itertools import zip_longest
from typing import Tuple, Optional, Union

import tensorflow as tf
import numpy as np

from src.config import *
from src.dataset import Dataset
from src.callbacks import OptimizerCallback, TrainIterationEndCallback
from src.networks.network import Network
from src.models.supervised.pre_upsampling_model import PreUpsamplingModel


class PreUpsamplingNetwork(Network):

    def __init__(self, input_shape: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, 1)):
        model = PreUpsamplingModel(input_shape=input_shape)
        super().__init__(model)
        self.learning_rate = tf.Variable(tf.constant(0, dtype=tf.float32))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=1e-8)

    def _predict(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        size = self._parse_predict_optionals(x, args, kwargs)
        x = tf.image.resize(x, size, tf.image.ResizeMethod.BICUBIC)
        y_pred = self.model(x).numpy()
        LOGGER.info(f"Predicted images with shape: {y_pred.shape}")
        return y_pred

    @tf.function
    def _train_step(self, x, y, loss_func):
        y = tf.convert_to_tensor(y)
        with tf.GradientTape() as tape:
            _shape = tf.shape(y)  # expecting 4D tensor in channel_last format
            x = tf.image.resize(x, (_shape[1], _shape[2]), tf.image.ResizeMethod.BICUBIC)
            y_pred = self.model(x)
            loss = loss_func(y, y_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return tf.reduce_sum(loss)

    def _train(self, x, y, loss_func, epochs, learning_rate, callbacks):
        self.learning_rate.assign(tf.constant(learning_rate, dtype=tf.float32))
        # train
        for e_idx in range(epochs):
            train_loss = tf.constant(0, dtype=tf.float32)
            start_sec = time.time()
            # process a batch
            random_y_idx = 0 if len(y) == 1 else np.random.randint(len(y))
            for x_b, y_b in zip_longest(x, y[random_y_idx]):
                train_loss += self._train_step(x_b, y_b, loss_func)
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
                        self.learning_rate.assign(tf.constant(cb(self), dtype=tf.float32))
                    if isinstance(cb, TrainIterationEndCallback):
                        cb(self)
