from src.callbacks import *
from src.dataset import *
import numpy as np
import tensorflow as tf

from src.networks.supervised.iterative_sampling_network import IterativeSamplingNetwork

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
epochs = 20
learning_rate = 1e-4

normalize = lambda x: np.asarray(x, dtype=np.float32) / 255.0

# ds = HDFDataset("gray_70_70.h5").batch(batch_size).shuffle(seed)
ds_lr = DirectoryDataset("./datasets/color/train_32_32").map(normalize).batch(batch_size).shuffle(seed).transform()
ds_hr = DirectoryDataset("./datasets/color/train_64_64").map(normalize).batch(batch_size).shuffle(seed).transform()
ds_ev_lr = DirectoryDataset("./datasets/color/test_32_32").map(normalize)
ds_ev_hr = DirectoryDataset("./datasets/color/test_64_64").map(normalize)

cb = [TrainingCheckpointCallback(save_freq=2),
      ExponentialDecayCallback(learning_rate, epochs // 5, decay_rate=0.9, staircase=False),
      TrainingEvaluationCallback(ds_ev_lr, ds_ev_hr, save_freq=2)]

network = IterativeSamplingNetwork((32, 32, 3))
network.train(ds_lr, ds_hr, tf.keras.losses.mse, epochs, learning_rate, cb)