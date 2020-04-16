import tensorflow as tf
import numpy as np

from src.config import *
from src.networks.network import Network
from src.models import ProgressiveUpsamplingModel


class ProgressiveUpsamplingNetwork(Network):

    def __init__(self):
        model = ProgressiveUpsamplingModel()    # increase the default scaling factor
        super().__init__(model)

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # TODO
        pass

    @tf.function
    def _train_step(self, x, y, optimizer, loss_func):
        # TODO
        pass

    def train(self, dataset_x, dataset_y, loss_func, epochs, learning_rate):
        # TODO
        pass
