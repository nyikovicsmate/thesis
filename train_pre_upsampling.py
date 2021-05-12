from src.callbacks import *
from src.dataset import *
import numpy as np
import tensorflow as tf

from src.networks.supervised.pre_upsampling_network import PreUpsamplingNetwork

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

batch_size = 16
seed = 1111
epochs = 100
learning_rate = 1e-5

normalize = lambda x: np.asarray(x, dtype=np.float32) / 255.0

# ds = HDFDataset("gray_70_70.h5").batch(batch_size).shuffle(seed)
ds_lr = DirectoryDataset("./datasets/color/train/64x64_64_64").map(normalize).batch(batch_size).shuffle(seed)
ds_hr = DirectoryDataset("./datasets/color/train/128x128_128_128").map(normalize).batch(batch_size).shuffle(seed)
ds_ev_lr = DirectoryDataset("./datasets/color/test_64_64").map(normalize).split((1, 67))[0]
ds_ev_hr = DirectoryDataset("./datasets/color/test_128_128").map(normalize).split((1, 67))[0]

# ds_ev_lr = DirectoryDataset("./datasets/color/bsd_120_120").map(normalize).split((1, 20))[0]
# ds_ev_hr = DirectoryDataset("./datasets/color/bsd_240_240").map(normalize).split((1, 20))[0]

cb = [TrainingCheckpointCallback(save_freq=5),
      ExponentialDecayCallback(learning_rate, epochs // 2, decay_rate=0.9, staircase=False),
      TrainingEvaluationCallback(ds_ev_lr, ds_ev_hr, save_freq=1)]
# cb = [ExponentialDecayCallback(learning_rate, epochs//2, decay_rate=0.5, staircase=True)]

network = PreUpsamplingNetwork((128, 128, 3))
network.train(ds_lr, ds_hr, tf.keras.losses.mean_squared_error, epochs, learning_rate, cb)

# jq '."100".metrics[].psnr' eval.json -r | awk '{s+=$1} END {print s / 136}'

# [15.281704 20.324247 21.670929 20.553467 24.317884 28.159758 19.38763
#  17.585741 21.928165 19.487244 22.46517  27.888937 20.709024 21.121864
#  23.673117 23.812391 21.225344 24.193289 19.657637 19.347797 24.782606
#  18.833656 20.969439 17.822294 17.431675 27.02173  18.198013 21.587193
#  16.348948 21.283175 17.13434  21.309475 20.065792 24.312654 21.510984
#  17.361498 17.445236 26.51351  21.069483 16.952627 24.645569 18.061605
#  19.078794 17.479761 18.10015  18.815487 16.242874 16.441975 18.351192
#  19.537003 19.427782 23.003399 19.752796 20.427011 20.075445 17.958033
#  17.610455 18.364946 24.730604 19.896269 18.237427 19.955736 18.639244
#  22.085707 18.774666 25.155338 20.508541 19.246662 19.952835 20.413124
#  19.827202 20.67849  20.858234 15.473083 21.754986 23.987875 19.20236
#  18.590578 19.846542 21.254763 21.25037  19.843466 22.041552 23.093096
#  17.704762 26.4481   19.274023 18.015074 21.393549 19.248528 20.653555
#  20.822155 27.3202   19.553867 22.93219  20.701765 19.649239 17.10923
#  19.92199  17.805136 15.007041 18.750679 18.459047 17.621365 18.889648
#  25.932728 28.023438 18.013103 19.456291 18.83126  20.993063 19.858156
#  33.134113 25.366804 22.324902 27.339518 29.394953 21.593315 22.892223
#  21.209291 20.201729 25.425827 17.858255 19.96138  16.929705 19.033443
#  19.711107 21.900629 19.186613 18.917103 17.436298 19.09948  25.96169
#  19.85461  20.650505 20.555927]
# Mean: 20.68931770324707
