from typing import Tuple, Optional
import tensorflow as tf


class DiscriminatorModel(tf.keras.models.Model):

    def __init__(self,
                 input_shape: Tuple[Optional[int], Optional[int], Optional[int]]):
        super().__init__()
        self.conv_0 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                             filters=128,
                                             kernel_size=3,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.RandomNormal())
        self.bn_0 = tf.keras.layers.BatchNormalization()
        self.act_0 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv_1 = tf.keras.layers.Conv2D(filters=32,
                                             kernel_size=3,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.RandomNormal())
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.act_1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv_2 = tf.keras.layers.Conv2D(filters=128,
                                             kernel_size=3,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.RandomNormal())
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.act_2 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv_3 = tf.keras.layers.Conv2D(filters=1,
                                             kernel_size=3,
                                             strides=1,
                                             padding="same",
                                             data_format="channels_last",
                                             use_bias=True,
                                             dilation_rate=1,
                                             activation="linear",
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             bias_initializer=tf.keras.initializers.RandomNormal())
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.act_3 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.dense_0 = tf.keras.layers.Dense(units=512, activation="relu")
        self.dense_1 = tf.keras.layers.Dense(units=1, activation="sigmoid")

    # noinspection DuplicatedCode
    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv_0(inputs)
        x = self.bn_0(x)
        x = self.act_0(x)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act_3(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense_0(x)
        x = self.dense_1(x)
        return x
