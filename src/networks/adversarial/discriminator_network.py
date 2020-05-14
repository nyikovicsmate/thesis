from itertools import zip_longest
from typing import Tuple, Optional, Union

import tensorflow as tf
import numpy as np
import contextlib

from src.config import *
from src.dataset import Dataset
from src.callbacks import OptimizerCallback, TrainIterationEndCallback
from src.networks.network import Network
from src.models.adversarial.discriminator_model import DiscriminatorModel


class DiscriminatorNetwork(Network):

    def __init__(self, input_shape: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, 1)):
        model = DiscriminatorModel(input_shape=input_shape)
        super().__init__(model)
        # workaround for tf.function decorator issue with multiple optimizer initializations, see https://github.com/tensorflow/tensorflow/issues/27120
        self.learning_rate = tf.Variable(tf.constant(0, dtype=tf.float32))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-8)

    def _predict(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        y_pred = self.model(x)
        return y_pred

    # @tf.function
    def _train_step(self, x, y, loss_func):
        with tf.GradientTape() as tape:
            fake_pred = self.model(x)
            loss_fake = loss_func(tf.zeros_like(fake_pred), self.model(x))
            real_pred = self.model(y)
            loss_real = loss_func(tf.ones_like(real_pred), real_pred)
            # y_pred = self.model(x)
            # loss = loss_func(y, y_pred)
            loss = loss_fake + loss_real
            grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return tf.reduce_sum(loss)

    def _train(self, x, y, loss_func, epochs, learning_rate, callbacks):
        """
        :param x: G(LR)
        :param y: HR
        :param loss_func:
        :param epochs:
        :param learning_rate:
        :param callbacks:
        :return:
        """
        self.learning_rate.assign(tf.constant(learning_rate, dtype=tf.float32))
        # train
        for e_idx in range(epochs):
            train_loss = tf.constant(0, dtype=tf.float32)
            start_sec = time.time()
            # process a batch
            random_y_idx = 0 if len(y) == 1 else np.random.randint(len(y))
            for x_b, y_b in zip_longest(x, y[random_y_idx]):
                # y_real = tf.constant(1, dtype=tf.float32, shape=(y_b.shape[0], 1))
                # y_fake = tf.constant(0, dtype=tf.float32, shape=(y_b.shape[0], 1))
                # train_real_loss = self._train_step(y_b, y_real, loss_func)
                # train_fake_loss = self._train_step(x_b, y_fake, loss_func)
                # train_loss += train_fake_loss + train_real_loss
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
                        self.learning_rate.assign(cb(self))
                    if isinstance(cb, TrainIterationEndCallback):
                        cb(self)