from typing import Tuple, Optional

import tensorflow as tf

from src.networks.adversarial.adversarial_network import AdversarialNetwork
from src.networks.adversarial.discriminator_network import DiscriminatorNetwork
from src.networks.supervised.progressive_upsampling_network import ProgressiveUpsamplingNetwork


class AdversarialProgressiveUpsamplingNetwork(AdversarialNetwork):

    def __init__(self,
                 input_shape: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, 3)):
        gen = ProgressiveUpsamplingNetwork(input_shape)
        # TODO: fix loading colored states
        gen.load_state("_color")
        disc = DiscriminatorNetwork(input_shape)
        disc.load_state("_color")
        super().__init__(gen, disc)

    def generator_loss(self, y: tf.Tensor, y_pred: tf.Tensor):
        """
        :param y: HR
        :param y_pred: list of G(LR) values
        :return:
        """
        # epsilon = 1e-6
        y_pred = y_pred[0]  # TODO: adopt approach for multiple dataset training
        w0 = tf.constant(0.3, dtype=tf.float32)
        w1 = tf.constant(0.7, dtype=tf.float32)
        epsilon_square = tf.square(tf.constant(1e-4, dtype=tf.float32))
        charbonnier_loss = tf.reduce_sum(w0 * tf.sqrt(tf.square(y-y_pred) + epsilon_square))  # aka. "content loss"
        # disc_loss = tf.reduce_sum(w1 * -tf.math.log(tf.clip_by_value(self.discriminator.predict(y_pred), epsilon, 1-epsilon)))       # aka. "adversarial loss"
        y_disc = self.discriminator_network.predict(y_pred)
        disc_loss = tf.reduce_sum(
            w1 * tf.keras.losses.binary_crossentropy(tf.ones_like(y_disc), y_disc))  # aka. "adversarial loss"
        result = charbonnier_loss + disc_loss
        return result

    def discriminator_loss(self, y: tf.Tensor, y_pred: tf.Tensor):
        """
        :param y: D(HR)
        :param y_pred: D(G(LR))
        :return:
        """
        # epsilon = 1e-6
        # result = -tf.math.log(tf.clip_by_value(y, epsilon, 1-epsilon)) - tf.math.log(tf.constant(1, dtype=tf.float32) - tf.clip_by_value(y_pred, epsilon, 1-epsilon))
        # return result
        return tf.keras.losses.binary_crossentropy(y, y_pred)
