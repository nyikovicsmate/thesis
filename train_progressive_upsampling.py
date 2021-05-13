from src.callbacks import *
from src.dataset import *
import numpy as np
import tensorflow as tf

from src.networks.supervised.progressive_upsampling_network import ProgressiveUpsamplingNetwork

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
ds_hr_0 = DirectoryDataset("./datasets/color/train_64_64").map(normalize).batch(batch_size).shuffle(seed).transform()
ds_hr_1 = DirectoryDataset("./datasets/color/train_128_128").map(normalize).batch(batch_size).shuffle(seed).transform()
ds_ev_lr = DirectoryDataset("./datasets/color/test_32_32").map(normalize)
ds_ev_hr = DirectoryDataset("./datasets/color/test_64_64").map(normalize)

cb = [TrainingCheckpointCallback(save_freq=2)
      # ExponentialDecayCallback(learning_rate, epochs // 5, decay_rate=0.9, staircase=False),
      # TrainingEvaluationCallback(ds_ev_lr, ds_ev_hr, save_freq=2)
      ]

network = ProgressiveUpsamplingNetwork((32, 32, 3))
network.load_state()
# network.train(ds_lr, [ds_hr_0, ds_hr_1], None, epochs, learning_rate, cb)
with ds_ev_lr as x, ds_ev_hr as y:
    lr = next(iter(x))
    y_true = next(iter(y))
    step = 10
    file = ROOT_PATH.joinpath("./evaluations/eval.json")
    for i in range(0, len(lr), step):
        lower = i
        upper = i + step
        print(f"Processing {lower}-{upper}")
        y_pred = network.model(lr[lower:upper])
        metrics = network.evaluate(y_pred[0], y_true[lower:upper])
        contents = {}
        metrics_json = []
        if file.exists():
            with open(str(file), mode="r") as logfile:
                contents = json.load(logfile)
                metrics_json = contents[str(network.state.epochs)]["metrics"]
        for item in metrics:
            t = {}
            for k, v in item.items():
                t[k] = str(v)
            metrics_json.append(t)
        contents[str(network.state.epochs)] = {
            "train_loss": str(network.state.train_loss),
            "train_time": str(network.state.train_time),
            "metrics": metrics_json
        }
        with open(str(file), mode="w") as logfile:
            # logfile.write(f"epoch: {network.state.epochs} loss: {inst.state.train_loss} train_time: {inst.state.train_time}")
            # logfile.write(str(metrics))
            json.dump(contents, logfile, indent=4)

        for j, img in enumerate(y_pred[0].numpy()):
            cv2.imwrite(str(ROOT_PATH.joinpath(f"./evaluations/{network.state.epochs}_{i+j}.png")), img * 255.0)

