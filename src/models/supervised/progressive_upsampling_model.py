from typing import Tuple

import tensorflow as tf


class ProgressiveUpsamplingModel(tf.keras.models.Model):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (None, None, 1)):
        super().__init__()
        self.conv_in = tf.keras.layers.Conv2D(input_shape=input_shape,
                                              filters=1,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=1,
                                              activation="relu",
                                              kernel_initializer=None,
                                              bias_initializer=None)
        self.conv_fe_1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=3,
                                                strides=1,
                                                padding="same",
                                                data_format="channels_last",
                                                use_bias=True,
                                                dilation_rate=1,
                                                activation="relu",
                                                kernel_initializer=None,
                                                bias_initializer=None)
        self.conv_fe_2 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=3,
                                                strides=1,
                                                padding="same",
                                                data_format="channels_last",
                                                use_bias=True,
                                                dilation_rate=1,
                                                activation="relu",
                                                kernel_initializer=None,
                                                bias_initializer=None)
        self.conv_fe_3 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=3,
                                                strides=1,
                                                padding="same",
                                                data_format="channels_last",
                                                use_bias=True,
                                                dilation_rate=1,
                                                activation="relu",
                                                kernel_initializer=None,
                                                bias_initializer=None)
        self.conv_fu = tf.keras.layers.Conv2DTranspose(filters=input_shape[-1],
                                                       kernel_size=4,
                                                       strides=2,  # upsample by 2x
                                                       padding="same",
                                                       data_format="channels_last",
                                                       dilation_rate=1,
                                                       activation="relu",
                                                       use_bias=True,
                                                       kernel_initializer=None,
                                                       bias_initializer=None)
        self.conv_res = tf.keras.layers.Conv2D(filters=input_shape[-1],
                                               kernel_size=3,
                                               strides=1,
                                               padding="same",
                                               data_format="channels_last",
                                               use_bias=True,
                                               dilation_rate=1,
                                               activation="relu",
                                               kernel_initializer=None,
                                               bias_initializer=None)
        self.conv_up = tf.keras.layers.Conv2DTranspose(filters=input_shape[-1],
                                                       kernel_size=4,
                                                       strides=2,  # upsample by 2x
                                                       padding="same",
                                                       data_format="channels_last",
                                                       dilation_rate=1,
                                                       activation="relu",
                                                       use_bias=True,
                                                       kernel_initializer=None,
                                                       bias_initializer=None)

    @tf.function
    def feature_extraction(self, inputs):
        x = self.conv_fe_1(inputs)
        x = self.conv_fe_2(x)
        x = self.conv_fe_3(x)
        x_up = self.conv_fu(x)
        x = self.conv_res(x_up)
        return x, x_up

    @tf.function
    def image_reconstruction(self, inputs):
        x = self.conv_up(inputs)
        return x

    @tf.function
    def call(self, inputs, training=None, mask=None):
        outputs = [None, None, None]
        x = self.conv_in(inputs)
        # 1st level
        x_rl, x_up = self.feature_extraction(x)
        xl = self.image_reconstruction(inputs)
        yl = tf.add(x_rl, xl)
        outputs[0] = yl
        # 2nd level
        x_rl, x_up = self.feature_extraction(x_up)
        xl = self.image_reconstruction(yl)
        yl = tf.add(x_rl, xl)
        outputs[1] = yl
        # 3rd level
        x_rl, x_up = self.feature_extraction(x_up)
        xl = self.image_reconstruction(yl)
        yl = tf.add(x_rl, xl)
        outputs[2] = yl
        return outputs
