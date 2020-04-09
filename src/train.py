from src.networks import *
from src.dataset import *
import numpy as np
import tensorflow as tf


def train_pre_upsampled():
    batch_size = 40
    input_shape = (35, 35, 1)
    # seed = 1000

    normalize = lambda x: np.asarray(x / 255.0, dtype=np.float16)
    ds_x = HDFDataset("C:/Users/nyiko/Documents/Projects/bme.thesis/bsd500_35_35.h5")\
        .batch(batch_size)\
        .map(normalize)
    ds_y = HDFDataset("C:/Users/nyiko/Documents/Projects/bme.thesis/bsd500_70_70.h5")\
        .batch(batch_size)\
        .map(normalize)

    network = PreUpsamplingNetwork(input_shape)
    loss_func = tf.keras.losses.mse
    network.train(ds_x, ds_y, loss_func, 100, 0.001)


train_pre_upsampled()
