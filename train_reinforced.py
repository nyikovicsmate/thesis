from src.callbacks import *
from src.dataset import *
import numpy as np
import tensorflow as tf
import cv2

from src.networks.reinforced.reinforced_network import ReinforcedNetwork


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

batch_size = 20
seed = 1111
epochs = 2000
learning_rate = 1e-4

normalize = lambda x: np.asarray(x / 255.0, dtype=np.float32)
downsample = lambda x: np.array([cv2.resize(x_i, (x.shape[2]//2, x.shape[1]//2), interpolation=cv2.INTER_CUBIC) for x_i in x])
grayscale = lambda x: np.array([cv2.cvtColor(x_i, cv2.COLOR_RGB2GRAY) for x_i in x])[:, :, :, np.newaxis]

ds_lr = DirectoryDataset("./datasets/color/train_g_32_32").map(normalize).batch(batch_size).shuffle(seed).transform()
ds_hr = DirectoryDataset("./datasets/color/train_g_64_64").map(normalize).batch(batch_size).shuffle(seed).transform()
ds_ev_lr = DirectoryDataset("./datasets/color/test_32_32").map(normalize)
ds_ev_hr = DirectoryDataset("./datasets/color/test_64_64").map(normalize)

cb = [TrainingCheckpointCallback(save_freq=10),
       ExponentialDecayCallback(learning_rate, epochs/10, decay_rate=0.9, staircase=True),
       TrainingEvaluationCallback(ds_ev_lr, ds_ev_hr, save_freq=20)]
# cb = [TrainingEvaluationCallback(ds_ev_lr, ds_ev_hr, save_freq=10)]

network = ReinforcedNetwork()
network.train(ds_lr, ds_hr, None, epochs, learning_rate, cb)