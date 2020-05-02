from typing import Tuple
import tensorflow as tf


class IterativeSamplingModel(tf.keras.models.Model):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (None, None, 1)):
        super().__init__()
        filters = 64
        kernel_size = 6
        strides = 2

        self.feat = self.FeatureExtractionBlock(input_shape=input_shape)
        self.up_1 = self.UpBlock(filters=filters, kernel_size=kernel_size, strides=strides)
        self.down_1 = self.DownBlock(filters=filters, kernel_size=kernel_size, strides=strides)
        self.up_2 = self.UpBlock(filters=filters, kernel_size=kernel_size, strides=strides)
        self.down_2 = self.DownBlock(filters=filters, kernel_size=kernel_size, strides=strides, stage_num=2)
        self.up_3 = self.UpBlock(filters=filters, kernel_size=kernel_size, strides=strides, stage_num=2)
        self.down_3 = self.DownBlock(filters=filters, kernel_size=kernel_size, strides=strides, stage_num=3)
        self.up_4 = self.UpBlock(filters=filters, kernel_size=kernel_size, strides=strides, stage_num=3)
        self.down_4 = self.DownBlock(filters=filters, kernel_size=kernel_size, strides=strides, stage_num=4)
        self.up_5 = self.UpBlock(filters=filters, kernel_size=kernel_size, strides=strides, stage_num=4)
        self.reconst = self.ReconstructionBlock(input_shape=input_shape)

    class FeatureExtractionBlock(tf.keras.models.Model):

        def __init__(self, input_shape):
            super().__init__()
            self.conv_0 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                                 filters=256,
                                                 kernel_size=3,
                                                 strides=1,
                                                 padding="same",
                                                 data_format="channels_last",
                                                 use_bias=True,
                                                 dilation_rate=1,
                                                 activation="relu",
                                                 kernel_initializer=tf.keras.initializers.he_uniform(),
                                                 bias_initializer=tf.keras.initializers.Zeros())
            self.conv_1 = tf.keras.layers.Conv2D(filters=64,
                                                 kernel_size=1,
                                                 strides=1,
                                                 padding="same",
                                                 data_format="channels_last",
                                                 use_bias=True,
                                                 dilation_rate=1,
                                                 activation="relu",
                                                 kernel_initializer=tf.keras.initializers.he_uniform(),
                                                 bias_initializer=tf.keras.initializers.Zeros())

        @tf.function
        def call(self, inputs):
            x = self.conv_0(inputs)
            x = self.conv_1(x)
            return x

    class UpBlock(tf.keras.models.Model):

        def __init__(self, filters, kernel_size, strides, stage_num=0):
            super().__init__()
            self._stage_num = stage_num
            if self._stage_num != 0:
                self.conv_stage = tf.keras.layers.Conv2D(filters=filters,
                                                         kernel_size=1,
                                                         strides=1,
                                                         padding="same",
                                                         data_format="channels_last",
                                                         use_bias=True,
                                                         dilation_rate=1,
                                                         activation="relu",
                                                         kernel_initializer=tf.keras.initializers.he_uniform(),
                                                         bias_initializer=tf.keras.initializers.Zeros())
            self.deconv_0 = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                            kernel_size=kernel_size,
                                                            strides=strides,
                                                            padding="same",
                                                            data_format="channels_last",
                                                            use_bias=True,
                                                            dilation_rate=1,
                                                            activation="relu",
                                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                                            bias_initializer=tf.keras.initializers.Zeros())
            self.conv_0 = tf.keras.layers.Conv2D(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding="same",
                                                 data_format="channels_last",
                                                 use_bias=True,
                                                 dilation_rate=1,
                                                 activation="relu",
                                                 kernel_initializer=tf.keras.initializers.he_uniform(),
                                                 bias_initializer=tf.keras.initializers.Zeros())
            self.deconv_1 = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                            kernel_size=kernel_size,
                                                            strides=strides,
                                                            padding="same",
                                                            data_format="channels_last",
                                                            use_bias=True,
                                                            dilation_rate=1,
                                                            activation="relu",
                                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                                            bias_initializer=tf.keras.initializers.Zeros())

        @tf.function
        def call(self, inputs):
            if self._stage_num != 0:
                inputs = self.conv_stage(inputs)
            h0 = self.deconv_0(inputs)
            l0 = self.conv_0(h0)
            res = l0 - inputs
            h1 = self.deconv_1(res)
            return h0 + h1

    class DownBlock(tf.keras.models.Model):

        def __init__(self, filters, kernel_size, strides, stage_num=0):
            super().__init__()
            self._stage_num = stage_num
            if self._stage_num != 0:
                self.conv_stage = tf.keras.layers.Conv2D(filters=filters,
                                                         kernel_size=1,
                                                         strides=1,
                                                         padding="same",
                                                         data_format="channels_last",
                                                         use_bias=True,
                                                         dilation_rate=1,
                                                         activation="relu",
                                                         kernel_initializer=tf.keras.initializers.he_uniform(),
                                                         bias_initializer=tf.keras.initializers.Zeros())
            self.conv_0 = tf.keras.layers.Conv2D(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding="same",
                                                 data_format="channels_last",
                                                 use_bias=True,
                                                 dilation_rate=1,
                                                 activation="relu",
                                                 kernel_initializer=tf.keras.initializers.he_uniform(),
                                                 bias_initializer=tf.keras.initializers.Zeros())
            self.deconv_0 = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                            kernel_size=kernel_size,
                                                            strides=strides,
                                                            padding="same",
                                                            data_format="channels_last",
                                                            use_bias=True,
                                                            dilation_rate=1,
                                                            activation="relu",
                                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                                            bias_initializer=tf.keras.initializers.Zeros())
            self.conv_1 = tf.keras.layers.Conv2D(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding="same",
                                                 data_format="channels_last",
                                                 use_bias=True,
                                                 dilation_rate=1,
                                                 activation="relu",
                                                 kernel_initializer=tf.keras.initializers.he_uniform(),
                                                 bias_initializer=tf.keras.initializers.Zeros())

        @tf.function
        def call(self, inputs):
            if self._stage_num != 0:
                inputs = self.conv_stage(inputs)
            l0 = self.conv_0(inputs)
            h0 = self.deconv_0(l0)
            res = h0 - inputs
            l1 = self.conv_1(res)
            return l0 + l1

    class ReconstructionBlock(tf.keras.models.Model):

        def __init__(self, input_shape):
            super().__init__()
            self.conv_0 = tf.keras.layers.Conv2D(filters=input_shape[-1],
                                                 kernel_size=3,
                                                 strides=1,
                                                 padding="same",
                                                 data_format="channels_last",
                                                 use_bias=True,
                                                 dilation_rate=1,
                                                 activation="linear",  # no activation
                                                 kernel_initializer=tf.keras.initializers.he_uniform(),
                                                 bias_initializer=tf.keras.initializers.Zeros())

        @tf.function
        def call(self, inputs):
            x = self.conv_0(inputs)
            return x

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.feat(inputs)

        h1 = self.up_1(x)
        l1 = self.down_1(h1)
        h2 = self.up_2(l1)

        concat_h = tf.concat((h2, h1), axis=3)
        l = self.down_2(concat_h)

        concat_l = tf.concat((l, l1), axis=3)
        h = self.up_3(concat_l)

        concat_h = tf.concat((h, concat_h), axis=3)
        l = self.down_3(concat_h)

        concat_l = tf.concat((l, concat_l), axis=3)
        h = self.up_4(concat_l)

        concat_h = tf.concat((h, concat_h), axis=3)
        l = self.down_4(concat_h)

        concat_l = tf.concat((l, concat_l), axis=3)
        h = self.up_5(concat_l)

        concat_h = tf.concat((h, concat_h), axis=3)
        x = self.reconst(concat_h)

        return x
