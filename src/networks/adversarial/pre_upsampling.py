import copy
from itertools import zip_longest

import numpy as np
import tensorflow as tf

from typing import Tuple, Optional, Iterable

from src.config import *
from src.dataset import Dataset
from src.networks.adversarial.discriminator_network import DiscriminatorNetwork
from src.networks.network import Network
from src.networks.supervised.pre_upsampling_network import PreUpsamplingNetwork


class PreUpsampling:

    def __init__(self, input_shape: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, 1)):
        self.generator: Network = PreUpsamplingNetwork(input_shape)
        self.generator.load_state()
        self.discriminator: Network = DiscriminatorNetwork(input_shape)

    def predict(self, x: tf.Tensor, *args, **kwargs):
        return self.generator.predict(x, *args, **kwargs)

    def discriminator_loss(self, y: tf.Tensor, y_pred: tf.Tensor):
        """
        :param y: D(HR)
        :param y_pred: D(G(LR))
        :return:
        """
        epsilon = 1e-6
        result = -tf.math.log(tf.clip_by_value(y, epsilon, 1-epsilon)) - tf.math.log(tf.constant(1, dtype=tf.float32) - tf.clip_by_value(y_pred, epsilon, 1-epsilon))
        return result

    def generator_loss(self, y: tf.Tensor, y_pred: tf.Tensor):
        """
        :param y: HR
        :param y_pred: G(LR)
        :return:
        """
        epsilon = 1e-6
        w0 = tf.constant(0.6, dtype=tf.float32)
        w1 = tf.constant(0.4, dtype=tf.float32)
        mse_loss = tf.reduce_sum(w0 * tf.losses.mse(y, y_pred))    # aka. "content loss"
        # disc_loss = tf.reduce_sum(w1 * -tf.math.log(tf.clip_by_value(self.discriminator.predict(y_pred), epsilon, 1-epsilon)))       # aka. "adversarial loss"
        y_disc = self.discriminator.predict(y_pred)
        disc_loss = tf.reduce_sum(w1 * tf.keras.losses.binary_crossentropy(tf.ones_like(y_disc), y_disc))       # aka. "adversarial loss"
        result = mse_loss + disc_loss
        return result

    def train(self,
              x: Iterable,
              y: Iterable,
              generator_epochs: int,
              discriminator_epochs: int,
              alternating_ratio: int,
              generator_lr: float,
              discriminator_lr: float):
        d_e = 0
        g_e = 0
        discriminator_stint = discriminator_epochs // alternating_ratio
        generator_stint = generator_epochs // alternating_ratio

        def predict_wrapper(x_b):
            # disable logging to avoid dozens of prediction messages
            LOGGER.setLevel(logging.WARNING)
            result = self.generator.predict(x_b)
            # re-enable logging
            LOGGER.setLevel(logging.INFO)
            return result

        # keep separate copies of the datasets to make sure they stay in sync during the alternating trainig
        if isinstance(x, Dataset):
            x_gen = x.map(predict_wrapper)
        else:
            x_gen = map(predict_wrapper, x)
        y_gen = copy.copy(y)
        x_disc = copy.copy(x)
        y_disc = copy.copy(y)

        # alternate between training generator and discriminator
        while d_e < discriminator_epochs and g_e < generator_epochs:
            if d_e < discriminator_epochs:
                self.discriminator.train(x_gen, y_gen, tf.keras.losses.binary_crossentropy, discriminator_stint, discriminator_lr)
                d_e += discriminator_stint
            if g_e < generator_epochs:
                self.generator.train(x_disc, y_disc, self.generator_loss , generator_stint, generator_lr)
                g_e += generator_stint
