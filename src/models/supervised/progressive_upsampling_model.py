from typing import Tuple

import tensorflow as tf


class ProgressiveUpsamplingModel(tf.keras.models.Model):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (None, None, 1)):
        super().__init__()
        self.conv_in = tf.keras.layers.Conv2D(input_shape=input_shape,
                                              filters=128,
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
        x_up = self.conv_fe_res(x)
        x = self.conv_res(x_up)
        return x, x_up

    @tf.function
    def image_reconstruction(self, inputs):
        x = self.conv_up(inputs)
        return x

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv_in(inputs)
        # 1st layer
        x, x_up = self.feature_extraction(x)
        x_2x = self.image_reconstruction(inputs)
        outputs_2x = tf.add(x, x_2x)
        # 2nd layer
        x, x_up = self.feature_extraction(x_up)
        x_4x = self.image_reconstruction(inputs)
        outputs_4x = tf.add(x, x_4x)
        # 3rd layer
        x, x_up = self.feature_extraction(x_up)
        x_8x = self.image_reconstruction(inputs)
        outputs_8x = tf.add(x, x_8x)
        return outputs_2x, outputs_4x, outputs_8x
