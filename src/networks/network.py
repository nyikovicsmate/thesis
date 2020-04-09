import tensorflow as tf
import numpy as np
from typing import Callable
from abc import ABC, abstractmethod
from src.dataset import Dataset


class Network(ABC):
    epoch: int
    train_loss: float
    valid_loss: float

    def __init__(self,
                 model: tf.keras.models.Model):
        self.epoch_cnt: int = 0
        self.model = model

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

    @abstractmethod
    def train(self, dataset_x: Dataset, dataset_y: Dataset, loss_func: Callable[[np.ndarray, np.ndarray], float], epochs: int, learning_rate: float):
        pass

    @abstractmethod
    def predict(self):
        """
        kap egy képet, vagy egy batch of képet és predikál
        :return:
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        kap 1x képet és az elvárt 2x képet és a prediktálás mellett az eredményeket
        különböző metrikák szerint kiértékeli a predikált eredmény és a kapott 2x képek összehasonlításával
        :return:
        """
        pass
