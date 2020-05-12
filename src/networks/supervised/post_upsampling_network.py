from typing import Tuple, Optional, Union

import tensorflow as tf
import numpy as np
import contextlib

from src.config import *
from src.dataset import Dataset
from src.callbacks import OptimizerCallback, TrainIterationEndCallback
from src.networks.network import Network
from src.models.supervised.post_upsampling_model import PostUpsamplingModel


class PostUpsamplingNetwork(Network):

    def __init__(self, input_shape: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, 1)):
        model = PostUpsamplingModel(input_shape=input_shape)
        super().__init__(model)

    def _predict(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        # as of now only constant 2x upsampling is supported
        # TODO: Implement transfer learning for quickly re-trainig the last deconv layer for diff upsampling rates
        # size = self._parse_predict_optionals(x, args, kwargs)
        y_pred = self.model(x)
        LOGGER.info(f"Predicted images with shape: {y_pred.shape}")
        return y_pred

    @tf.function
    def _train_step(self, x, y, optimizer, loss_func):
        y = tf.convert_to_tensor(y)
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = loss_func(y, y_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return tf.reduce_sum(loss)

    def _train(self, x, y, loss_func, epochs, learning_rate, callbacks):
        # TODO: remove when solved
        assert len(y) == 1, "Post upsampling is currently only trainable with fixed rates because of the fixed deconv layer."
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
            random_y_idx = 0 if len(y) == 1 else np.random.randint(len(y))
            for x_b, y_b in zip(x, y[random_y_idx]):
                train_loss += self._train_step(x_b, y_b, optimizer, loss_func)
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
