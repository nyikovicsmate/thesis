from typing import Tuple
import tensorflow as tf


class PostUpsamplingModel(tf.keras.models.Model):

    def __init__(self,
                 scaling_factor: float,
                 input_shape: Tuple[int, int, int] = (None, None, 1)):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.conv1 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                            filters=64,
                                            kernel_size=6,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,  # no dilation
                                            activation="relu",
                                            kernel_initializer=None,
                                            bias_initializer=None)
        self.conv2 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=1,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,  # no dilation
                                            activation="relu",
                                            kernel_initializer=None,
                                            bias_initializer=None)
        self.conv3 = tf.keras.layers.Conv2D(filters=input_shape[-1],  # output depth is the same as the input's depth
                                            kernel_size=3,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,  # no dilation
                                            activation="linear",  # no activation
                                            kernel_initializer=None,
                                            bias_initializer=None)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        _shape = tf.shape(inputs)  # expecting 4D tensor in channel_last format
        _new_size = (_shape[1] * self.scaling_factor, _shape[2] * self.scaling_factor)
        x = self.conv1(inputs)
        x = self.conv2(x)
        outputs = self.conv3(x)
        # scale the outputs
        outputs = tf.image.resize(outputs, _new_size, tf.image.ResizeMethod.BICUBIC)
        return outputs
