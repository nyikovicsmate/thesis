from typing import Tuple, Optional
import tensorflow as tf


class DiscriminatorModel(tf.keras.models.Model):

    def __init__(self,
                 input_shape: Tuple[Optional[int], Optional[int], Optional[int]]):
        super().__init__()
        self.conv_0 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                             filters=64,
                                             kernel_size=3,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.RandomUniform(),
                                             bias_initializer=tf.keras.initializers.Zeros())
        self.act_0 = tf.keras.layers.ReLU()
        self.bn_0 = tf.keras.layers.BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                             filters=64,
                                             kernel_size=3,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.RandomUniform(),
                                             bias_initializer=tf.keras.initializers.Zeros())
        self.act_1 = tf.keras.layers.ReLU()
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                             filters=64,
                                             kernel_size=3,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.RandomUniform(),
                                             bias_initializer=tf.keras.initializers.Zeros())
        self.act_2 = tf.keras.layers.ReLU()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=1)
        self.softmax = tf.keras.layers.Softmax()

    # noinspection DuplicatedCode
    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv_0(inputs)
        x = self.act_0(x)
        x = self.bn_0(x)
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.bn_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x

