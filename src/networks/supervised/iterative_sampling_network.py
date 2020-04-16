import tensorflow as tf
import numpy as np

from src.config import *
from src.networks.network import Network
from src.models import IterativeSamplingModel


class IterativeSamplingNetwork(Network):

    def __init__(self):
        model = IterativeSamplingModel()
        super().__init__(model)

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # TODO
        pass

    @tf.function
    def _train_step(self, x, y, optimizer, loss_func):
        # TODO
        pass

    def train(self, dataet_x, dataset_y, loss_func, epochs, learning_rate):
        # TODO
        pass
