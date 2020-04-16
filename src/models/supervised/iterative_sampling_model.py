from typing import Tuple
import tensorflow as tf


class IterativeSamplingModel(tf.keras.models.Model):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (None, None, 1)):
        super().__init__()
        # TODO

    @tf.function
    def call(self, inputs, training=None, mask=None):
        # TODO
        pass
