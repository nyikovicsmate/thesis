from typing import Tuple, Optional

import tensorflow as tf


# noinspection DuplicatedCode
class ReinforcedModel(tf.keras.models.Model):
    """PixelwiseA3CModel"""

    def __init__(self):
        super().__init__()
        input_shape = (None, None, 1)
        self.conv1 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                            filters=64,
                                            kernel_size=3,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,  # no dilation
                                            activation="relu",
                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                            bias_initializer=tf.keras.initializers.Zeros())
        self.diconv2 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=2,
                                              activation="relu",
                                              kernel_initializer=tf.keras.initializers.he_uniform(),
                                              bias_initializer=tf.keras.initializers.Zeros())
        self.diconv3 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=3,
                                              activation="relu",
                                              kernel_initializer=tf.keras.initializers.he_uniform(),
                                              bias_initializer=tf.keras.initializers.Zeros())
        self.diconv4 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=4,
                                              activation="relu",
                                              kernel_initializer=tf.keras.initializers.he_uniform(),
                                              bias_initializer=tf.keras.initializers.Zeros())
        self.actor_diconv5 = tf.keras.layers.Conv2D(filters=64,
                                                    kernel_size=3,
                                                    strides=1,
                                                    padding="same",
                                                    data_format="channels_last",
                                                    use_bias=True,
                                                    dilation_rate=3,
                                                    activation="relu",
                                                    kernel_initializer=tf.keras.initializers.he_uniform(),
                                                    bias_initializer=tf.keras.initializers.Zeros())
        self.actor_diconv6 = tf.keras.layers.Conv2D(filters=64,
                                                    kernel_size=3,
                                                    strides=1,
                                                    padding="same",
                                                    data_format="channels_last",
                                                    use_bias=True,
                                                    dilation_rate=2,
                                                    activation="relu",
                                                    kernel_initializer=tf.keras.initializers.he_uniform(),
                                                    bias_initializer=tf.keras.initializers.Zeros())
        self.actor_conv7 = tf.keras.layers.Conv2D(filters=9,  # number of actions
                                                  kernel_size=3,
                                                  strides=1,
                                                  padding="same",
                                                  data_format="channels_last",
                                                  use_bias=True,
                                                  dilation_rate=1,
                                                  activation="softmax",
                                                  kernel_initializer=tf.keras.initializers.he_uniform(),
                                                  bias_initializer=tf.keras.initializers.Zeros())
        self.critic_diconv5 = tf.keras.layers.Conv2D(filters=64,
                                                     kernel_size=3,
                                                     strides=1,
                                                     padding="same",
                                                     data_format="channels_last",
                                                     use_bias=True,
                                                     dilation_rate=3,
                                                     activation="relu",
                                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                                     bias_initializer=tf.keras.initializers.Zeros())
        self.critic_diconv6 = tf.keras.layers.Conv2D(filters=64,
                                                     kernel_size=3,
                                                     strides=1,
                                                     padding="same",
                                                     data_format="channels_last",
                                                     use_bias=True,
                                                     dilation_rate=2,
                                                     activation="relu",
                                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                                     bias_initializer=tf.keras.initializers.Zeros())
        self.critic_conv7 = tf.keras.layers.Conv2D(filters=1,
                                                   kernel_size=3,
                                                   strides=1,
                                                   padding="same",
                                                   data_format="channels_last",
                                                   use_bias=True,
                                                   dilation_rate=1,
                                                   activation="linear",
                                                   kernel_initializer=tf.keras.initializers.he_uniform(),
                                                   bias_initializer=tf.keras.initializers.Zeros())

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        actor = self.actor_diconv5(x)
        actor = self.actor_diconv6(actor)
        actor = self.actor_conv7(actor)  # output shape (batch_size, width, height, 9)
        critic = self.critic_diconv5(x)
        critic = self.critic_diconv6(critic)
        critic = self.critic_conv7(critic)  # output shape (batch_size, width, height, 1)
        return actor, critic
