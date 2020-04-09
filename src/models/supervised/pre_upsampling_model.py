from typing import Tuple
import tensorflow as tf


class PreUpsamplingModel(tf.keras.models.Model):
    SCALE: int = 2

    def __init__(self,
                 input_shape: Tuple[int, int, int]):    # (height, width, depth)
        super().__init__()
        self._input_shape = input_shape
        self._output_shape = (*map(lambda x: x * self.SCALE, self._input_shape[:2]), self._input_shape[2])
        self.conv1 = tf.keras.layers.Conv2D(input_shape=self._output_shape,
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
        # self.conv3 = tf.keras.layers.Conv2D(filters=128,
        #                                     kernel_size=3,
        #                                     strides=1,
        #                                     padding="same",
        #                                     data_format="channels_last",
        #                                     use_bias=True,
        #                                     dilation_rate=1,  # no dilation
        #                                     activation="relu",
        #                                     kernel_initializer=None,
        #                                     bias_initializer=None)
        # self.conv4 = tf.keras.layers.Conv2D(filters=128,
        #                                     kernel_size=3,
        #                                     strides=1,
        #                                     padding="same",
        #                                     data_format="channels_last",
        #                                     use_bias=True,
        #                                     dilation_rate=1,  # no dilation
        #                                     activation="relu",
        #                                     kernel_initializer=None,
        #                                     bias_initializer=None)
        # self.conv5 = tf.keras.layers.Conv2D(filters=128,
        #                                     kernel_size=3,
        #                                     strides=1,
        #                                     padding="same",
        #                                     data_format="channels_last",
        #                                     use_bias=True,
        #                                     dilation_rate=1,  # no dilation
        #                                     activation="relu",
        #                                     kernel_initializer=None,
        #                                     bias_initializer=None)
        self.conv6 = tf.keras.layers.Conv2D(filters=self._output_shape[2],
                                            kernel_size=3,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,  # no dilation
                                            activation="linear",
                                            kernel_initializer=None,
                                            bias_initializer=None)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        inputs = tf.image.resize(inputs, tuple(self._output_shape[:2]), tf.image.ResizeMethod.BICUBIC)
        x = self.conv1(inputs)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        outputs = self.conv6(x)
        return outputs

