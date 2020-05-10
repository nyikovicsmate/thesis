from typing import Tuple, Optional
import tensorflow as tf


class PreUpsamplingModel(tf.keras.models.Model):
    """
    SRCNN model - https://arxiv.org/pdf/1501.00092.pdf
    Base model works with YCbCr images, specifically with the luminance (Y) channel,
    however this model meant to be used with RGB images, wich according to TABLE 5
    sould achieve comparable or even better results.
    """

    def __init__(self,
                 input_shape: Tuple[Optional[int], Optional[int], Optional[int]]):
        super().__init__()
        self.conv_0 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                             filters=64,  # optimum filter number (see TABLE 1)
                                             kernel_size=9,  # optimum filter size (see Fig. 1)
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.Zeros())
        self.act_0 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Zeros())
        self.conv_1 = tf.keras.layers.Conv2D(filters=32,  # optimum filter number (see TABLE 1)
                                             kernel_size=5,  # optimum filter size (see Fig. 1)
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.Zeros())
        self.act_1 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Zeros())
        self.conv_2 = tf.keras.layers.Conv2D(filters=input_shape[-1],  # output depth is the same as the input's depth
                                             kernel_size=5,  # optimum filter size (see Fig. 1)
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.Zeros())
        self.act_2 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Zeros())

    # noinspection DuplicatedCode
    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv_0(inputs)
        x = self.act_0(x)
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        return x

