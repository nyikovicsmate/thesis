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
        result = -tf.math.log(y) - tf.math.log(tf.constant(1, dtype=tf.float32) - y_pred)
        return result

    def generator_loss(self, y: tf.Tensor, y_pred: tf.Tensor):
        """
        :param y: HR
        :param y_pred: G(LR)
        :return:
        """
        w0 = tf.constant(0.5, dtype=tf.float32)
        w1 = tf.constant(5, dtype=tf.float32)
        mse_loss = tf.reduce_sum(w0 * tf.losses.mse(y, y_pred))    # aka. "content loss"
        disc_loss = tf.reduce_sum(w1 * -tf.math.log(self.discriminator.predict(y_pred)))       # aka. "adversarial loss"
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
        # alternate between training generator and discriminator
        while d_e < discriminator_epochs and g_e < generator_epochs:
            if d_e < discriminator_epochs:
                # TODO: work out a better way to iteratively pass the predicted data
                with x:
                    LOGGER.setLevel(logging.WARNING)
                    x_list = [self.generator.predict(i) for i in x]
                    LOGGER.setLevel(logging.INFO)
                self.discriminator.train(x_list, y, tf.losses.binary_crossentropy, discriminator_stint, discriminator_lr)
                d_e += discriminator_stint
            if g_e < generator_epochs:
                self.generator.train(x, y, self.generator_loss, generator_stint, generator_lr)
                g_e += generator_stint
