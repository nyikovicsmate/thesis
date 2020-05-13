import numpy as np
import tensorflow as tf

from typing import Tuple, Optional, Iterable

from src.config import *
from src.networks.adversarial.discriminator_network import DiscriminatorNetwork
from src.networks.network import Network
from src.networks.supervised.pre_upsampling_network import PreUpsamplingNetwork


class PreUpsampling:

    def __init__(self, input_shape: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, 1)):
        self.generator: Network = PreUpsamplingNetwork(input_shape)
        self.discriminator: Network = DiscriminatorNetwork(input_shape)

    def predict(self, x: tf.Tensor, *args, **kwargs):
        return self.generator.predict(x, args, kwargs)

    def discriminator_loss(self, y: tf.Tensor, y_pred: tf.Tensor):
        epsilon = 1e-6
        result = -tf.math.log(tf.clip_by_value(y, epsilon, 1)) - tf.math.log(
            tf.constant(1, dtype=tf.float32) - tf.clip_by_value(y_pred, 0, 1 - epsilon))
        return result

    def generator_loss(self, y: tf.Tensor, y_pred: tf.Tensor):
        w0 = tf.constant(0.5)
        w1 = tf.constant(0.05)
        mse_loss = tf.reshape(w0 * tf.losses.mse(y, y_pred), shape=y.shape)
        disc_loss = tf.expand_dims(tf.expand_dims(w1 * self.discriminator.predict(y_pred), axis=-1), axis=-1)
        result = mse_loss + disc_loss  # disc_loss auto-broadcasted  (n,1,1,1) --> (n, h, w, c)
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
        # alternate between training generator and discriminator
        while d_e < discriminator_epochs and g_e < generator_epochs:
            if g_e < generator_epochs:
                self.generator.train(x, y, self.generator_loss, generator_stint, generator_lr)
                g_e += generator_stint
            if d_e < discriminator_epochs:
                # TODO: work out a better way to iteratively pass the predicted data
                with x:
                    LOGGER.setLevel(logging.WARNING)
                    x_list = [self.generator.predict(i) for i in x]
                    LOGGER.setLevel(logging.INFO)
                self.discriminator.train(x_list, y, self.discriminator_loss, discriminator_stint, discriminator_lr)
                d_e += discriminator_stint
