from src.networks import *
from src.dataset import *
import numpy as np
import tensorflow as tf


def train_pre_upsampled():
    batch_size = 80
    # seed = 1000

    normalize = lambda x: np.asarray(x / 255.0, dtype=np.float16)
    ds_x = HDFDataset("bsd500_35_35.h5")\
        .batch(batch_size)\
        .map(normalize)
    ds_y1 = HDFDataset("bsd500_70_70.h5")\
        .batch(batch_size)\
        .map(normalize)
    ds_y2 = HDFDataset("bsd500_140_140.h5")\
        .batch(batch_size)\
        .map(normalize)

    network = PostUpsamplingNetwork()
    loss_func = tf.keras.losses.mse
    network.train(ds_x, [ds_y1, ds_y2], loss_func, 10, 0.001)

    # with ds_x.batch(10) as x, ds_y1.batch(10) as y:
    #     # predict
    #     y_pred = network.predict(next(x))
    #     # evaluate
    #     results = network.evaluate(next(y), y_pred)


train_pre_upsampled()
