from typing import Tuple, Optional

import tensorflow as tf


class ProgressiveUpsamplingModel(tf.keras.models.Model):

    def __init__(self,
                 input_shape: Tuple[Optional[int], Optional[int], Optional[int]]):
        super().__init__()
        self.conv_in = tf.keras.layers.Conv2D(input_shape=input_shape,
                                              filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=1,
                                              activation="relu",
                                              kernel_initializer=tf.keras.initializers.he_uniform(),
                                              bias_initializer=tf.keras.initializers.Zeros())
        self.conv_fe_1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=3,
                                                strides=1,
                                                padding="same",
                                                data_format="channels_last",
                                                use_bias=True,
                                                dilation_rate=1,
                                                activation="relu",
                                                kernel_initializer=tf.keras.initializers.he_uniform(),
                                                bias_initializer=tf.keras.initializers.Zeros())
        self.conv_fe_2 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=3,
                                                strides=1,
                                                padding="same",
                                                data_format="channels_last",
                                                use_bias=True,
                                                dilation_rate=1,
                                                activation="relu",
                                                kernel_initializer=tf.keras.initializers.he_uniform(),
                                                bias_initializer=tf.keras.initializers.Zeros())
        self.conv_fu = tf.keras.layers.Conv2DTranspose(filters=64,
                                                       kernel_size=4,
                                                       strides=2,  # upsample by 2x
                                                       padding="same",
                                                       data_format="channels_last",
                                                       dilation_rate=1,
                                                       activation="relu",
                                                       use_bias=True,
                                                       kernel_initializer=tf.keras.initializers.he_uniform(),
                                                       bias_initializer=tf.keras.initializers.Zeros())
        self.conv_res = tf.keras.layers.Conv2D(filters=input_shape[-1],
                                               kernel_size=3,
                                               strides=1,
                                               padding="same",
                                               data_format="channels_last",
                                               use_bias=True,
                                               dilation_rate=1,
                                               activation="relu",
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               bias_initializer=tf.keras.initializers.Zeros())
        self.conv_up = tf.keras.layers.Conv2DTranspose(filters=input_shape[-1],
                                                       kernel_size=4,
                                                       strides=2,  # upsample by 2x
                                                       padding="same",
                                                       data_format="channels_last",
                                                       dilation_rate=1,
                                                       activation="relu",
                                                       use_bias=True,
                                                       kernel_initializer=tf.keras.initializers.he_uniform(),
                                                       bias_initializer=tf.keras.initializers.Zeros())

    @tf.function
    def feature_extraction(self, inputs):
        x_fe = self.conv_fe_1(inputs)
        x_fe = self.conv_fe_2(x_fe)
        x_fe_up = self.conv_fu(x_fe)
        x_fe_res = self.conv_res(x_fe_up)
        return x_fe_res, x_fe_up

    @tf.function
    def image_reconstruction(self, inputs):
        x_up = self.conv_up(inputs)
        return x_up

    @tf.function
    def call(self, inputs, training=None, mask=None):
        outputs = [None, None, None]
        x = self.conv_in(inputs)
        # 1st level
        x_fe_res, x_fe_up = self.feature_extraction(x)
        x_up = self.image_reconstruction(inputs)
        y = x_fe_res + x_up
        outputs[0] = y
        # 2nd level
        x_fe_res, x_fe_up = self.feature_extraction(x_fe_up)
        x_up = self.image_reconstruction(y)
        y = x_fe_res + x_up
        outputs[1] = y
        # 3rd level
        x_fe_res, x_fe_up = self.feature_extraction(x_fe_up)
        x_up = self.image_reconstruction(y)
        y = x_fe_res + x_up
        outputs[2] = y
        return outputs
