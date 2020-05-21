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

    def __init__(self, input_shape: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, 3)):
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

    # TODO: uncomment in production
    # @tf.function
    def _train_step(self, x, y, loss_func):
        y = tf.convert_to_tensor(y)
        with tf.GradientTape() as tape:
            # fake_pred = self.model(x)
            # loss_fake = loss_func(tf.zeros_like(fake_pred), self.model(x))
            # real_pred = self.model(y)
            # loss_real = loss_func(tf.ones_like(real_pred), real_pred)
            # # y_pred = self.model(x)
            # # loss = loss_func(y, y_pred)
            # loss = loss_fake + loss_real
            y_pred = self.model(x)
            loss = loss_func(y, y_pred)
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
        # step = 8
        for e_idx in range(epochs):
            train_loss = tf.constant(0, dtype=tf.float32)
            start_sec = time.time()
            random_y_idx = 0 if len(y) == 1 else np.random.randint(len(y))
            # for _ in range(step//2):
            #     for y_b in y[random_y_idx]:
            #         labels = np.ones((len(y_b), 1))
            #         train_loss += self._train_step(y_b, labels, loss_func)
            # for _ in range(step//2):
            #     for x_b in x:
            #         labels = np.zeros((len(x_b), 1))
            #         train_loss += self._train_step(x_b, labels, loss_func)
            for x_b, y_b in zip_longest(x, y[random_y_idx]):
                # y_real = tf.constant(1, dtype=tf.float32, shape=(y_b.shape[0], 1))
                # y_fake = tf.constant(0, dtype=tf.float32, shape=(y_b.shape[0], 1))
                # train_real_loss = self._train_step(y_b, y_real, loss_func)
                # train_fake_loss = self._train_step(x_b, y_fake, loss_func)
                # train_loss += train_fake_loss + train_real_loss

                # _len = len(x_b)
                # split = np.random.randint(0, 2)
                # labels = [0] * split + [1] * (_len-split)
                # labels = np.array(labels, dtype=np.float32)
                # np.random.shuffle(labels)
                # x_b_iter = iter(x_b)
                # y_b_iter = iter(y_b)
                # items = []
                # for l in labels:
                #     item = next(x_b_iter) if l == 0 else next(y_b_iter)
                #     items.append(item)
                # train_loss += self._train_step(np.array(items, dtype=np.float32), labels[:, np.newaxis], loss_func)
                labels = np.ones((len(y_b), 1))
                # sneak in some impostor (low resolution) values in random positions
                impostors = np.random.choice(np.arange(len(x_b)), size=len(x_b)//4, replace=False)
                for i in impostors:
                    y_b[i] = x_b[i]
                    labels[i][0] = 0
                train_loss += self._train_step(y_b, labels, loss_func)
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