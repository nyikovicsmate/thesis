from typing import Tuple
import tensorflow as tf


class PreUpsamplingModel(tf.keras.models.Model):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (None, None, 1)):    # (height, width, depth), arbitrary grayscale images as default
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                            filters=64,
                                            kernel_size=11,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,
                                            activation="relu",
                                            kernel_initializer=None,
                                            bias_initializer=None)
        self.conv2 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=7,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,
                                            activation="relu",
                                            kernel_initializer=None,
                                            bias_initializer=None)
        self.conv3 = tf.keras.layers.Conv2D(filters=input_shape[-1],    # output depth is the same as the input's depth
                                            kernel_size=7,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,
                                            activation="linear",    # no activation
                                            kernel_initializer=None,
                                            bias_initializer=None)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        outputs = self.conv3(x)
        return outputs

