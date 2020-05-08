from typing import Tuple
import tensorflow as tf


class PostUpsamplingModel(tf.keras.models.Model):
    """
    FSRCNN model - https://arxiv.org/pdf/1608.00367.pdf
    Base model works with YCbCr images, specifically with the luminance (Y) channel,
    however this model meant to be used with RGB images, which according to a different
    paper from the same authors https://arxiv.org/pdf/1501.00092.pdf should achieve
    comparable or even better results bacause of the high correlance between rgb channel
    values.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (None, None, 1)):
        super().__init__()
        # feature extraction
        self.conv_0 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                             filters=56,
                                             kernel_size=5,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.Zeros())
        self.act_0 = tf.keras.layers.PReLU()
        # shrinking
        self.conv_1 = tf.keras.layers.Conv2D(filters=12,
                                             kernel_size=1,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.Zeros())
        self.act_1 = tf.keras.layers.PReLU()
        # mapping (m = 4)
        self.conv_2_0 = tf.keras.layers.Conv2D(filters=12,
                                               kernel_size=3,
                                               strides=1,
                                               padding="same",
                                               data_format="channels_last",
                                               use_bias=True,
                                               dilation_rate=1,
                                               activation="linear",
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               bias_initializer=tf.keras.initializers.Zeros())
        self.act_2_0 = tf.keras.layers.PReLU()
        self.conv_2_1 = tf.keras.layers.Conv2D(filters=12,
                                               kernel_size=3,
                                               strides=1,
                                               padding="same",
                                               data_format="channels_last",
                                               use_bias=True,
                                               dilation_rate=1,
                                               activation="linear",
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               bias_initializer=tf.keras.initializers.Zeros())
        self.act_2_1 = tf.keras.layers.PReLU()
        self.conv_2_2 = tf.keras.layers.Conv2D(filters=12,
                                               kernel_size=3,
                                               strides=1,
                                               padding="same",
                                               data_format="channels_last",
                                               use_bias=True,
                                               dilation_rate=1,
                                               activation="linear",
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               bias_initializer=tf.keras.initializers.Zeros())
        self.act_2_2 = tf.keras.layers.PReLU()
        self.conv_2_3 = tf.keras.layers.Conv2D(filters=12,
                                               kernel_size=3,
                                               strides=1,
                                               padding="same",
                                               data_format="channels_last",
                                               use_bias=True,
                                               dilation_rate=1,
                                               activation="linear",
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               bias_initializer=tf.keras.initializers.Zeros())
        self.act_2_3 = tf.keras.layers.PReLU()
        # expanding
        self.conv_3 = tf.keras.layers.Conv2D(filters=56,
                                             kernel_size=1,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.Zeros())
        self.act_3 = tf.keras.layers.PReLU()
        # deconvolution
        self.conv_4 = tf.keras.layers.Conv2DTranspose(filters=input_shape[-1],
                                                      kernel_size=9,
                                                      strides=2,  # upsampling factor
                                                      padding="same",
                                                      data_format="channels_last",
                                                      use_bias=True,
                                                      dilation_rate=1,
                                                      activation="linear",
                                                      kernel_initializer=tf.keras.initializers.he_uniform(),
                                                      bias_initializer=tf.keras.initializers.Zeros())
        self.act_4 = tf.keras.layers.PReLU()

    # noinspection DuplicatedCode
    @tf.function
    def call(self, inputs, training=None, mask=None):
        # feature extraction
        x = self.conv_0(inputs)
        x = self.act_0(x)
        # shrinking
        x = self.conv_1(x)
        x = self.act_1(x)
        # mapping (m = 4)
        x = self.conv_2_0(x)
        x = self.act_2_0(x)
        x = self.conv_2_1(x)
        x = self.act_2_1(x)
        x = self.conv_2_2(x)
        x = self.act_2_2(x)
        x = self.conv_2_3(x)
        x = self.act_2_3(x)
        # expanding
        x = self.conv_3(x)
        x = self.act_3(x)
        # deconvolution
        x = self.conv_4(x)
        x = self.act_4(x)
        return x
