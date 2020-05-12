import numpy as np
import tensorflow as tf

from typing import Tuple, Optional, Iterable

from src.dataset import Dataset
from src.networks.adversarial.discriminator_network import DiscriminatorNetwork
from src.networks.network import Network
from src.networks.supervised.pre_upsampling_network import PreUpsamplingNetwork


class PreUpsampling:

    def __init__(self, input_shape: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, 1)):
        self.generator: Network = PreUpsamplingNetwork(input_shape)
        self.discriminator: Network = DiscriminatorNetwork(input_shape)

    def _predict(self, x: tf.Tensor, *args, **kwargs):
        return self.generator.predict(x, args, kwargs)

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
        def discriminator_loss(y: tf.Tensor, y_pred: tf.Tensor):
            return -tf.math.log(self.discriminator.predict(y)) - \
                   tf.math.log(tf.constant(1, dtype=tf.float32) - self.discriminator.predict(y_pred))
        def generator_loss(y: tf.Tensor, y_pred: tf.Tensor):
            w0 = tf.constant(0.5)
            w1 = tf.constant(0.05)
            mse_loss = tf.reshape(w0 * tf.losses.mse(y, y_pred), shape=y.shape)
            disc_loss = tf.expand_dims(tf.expand_dims(w1 * self.discriminator.predict(y_pred), axis=-1), axis=-1)
            result = mse_loss + disc_loss   # disc_loss auto-broadcasted  (n,1,1,1) --> (n, h, w, c)
            return result
        def predict_gen(x: Iterable):
            with x:
                for i in x:
                    yield self.generator.predict(i)


        # alternate between training generator and discriminator
        while d_e < discriminator_epochs and g_e < generator_epochs:
            if g_e < generator_epochs:
                self.generator.train(x, y, generator_loss, generator_stint, generator_lr)
                g_e += generator_stint
            if d_e < discriminator_epochs:
                self.discriminator.train(predict_gen(x), y, discriminator_loss, discriminator_stint, discriminator_lr)
                d_e += discriminator_stint

    @staticmethod
    @tf.function
    def _charbonnier_loss(x: tf.Tensor):
        epsilon_square = tf.square(tf.constant(1e-4, dtype=tf.float32))
        return tf.sqrt(tf.square(x) + epsilon_square)