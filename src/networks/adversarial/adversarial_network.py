import copy
from abc import ABCMeta, abstractmethod
from typing import Iterable, Optional, List

import tensorflow as tf

from src.callbacks import Callback
from src.config import *
from src.dataset import Dataset
from src.networks.network import Network


class AdversarialNetwork(metaclass=ABCMeta):
    """Abstract base class for adversarial trainig wrapper classes."""

    def __init__(self,
                 generator_network: Network,
                 discriminator_network: Network):
        self.generator_network = generator_network
        self.discriminator_network = discriminator_network

    @abstractmethod
    def generator_loss(self, y: tf.Tensor, y_pred: tf.Tensor):
        pass

    @abstractmethod
    def discriminator_loss(self, y: tf.Tensor, y_pred: tf.Tensor):
        pass

    def predict(self, x: Iterable, *args, **kwargs):
        return self.generator_network.predict(x, *args, **kwargs)

    def train(self,
              x: Iterable,
              y: Iterable,
              generator_epochs: int,
              discriminator_epochs: int,
              alternating_ratio: int,
              generator_lr: float,
              discriminator_lr: float,
              generator_callbacks: Optional[List[Callback]] = None,
              discriminator_callbacks: Optional[List[Callback]] = None):
        d_e = 0
        g_e = 0
        discriminator_stint = discriminator_epochs // alternating_ratio
        generator_stint = generator_epochs // alternating_ratio

        def predict_wrapper(x_b):
            # disable logging to avoid dozens of prediction messages
            LOGGER.setLevel(logging.WARNING)
            result = self.generator_network.predict(x_b)
            # re-enable logging
            LOGGER.setLevel(logging.INFO)
            return result

        # keep separate copies of the datasets to make sure they stay in sync during the alternating trainig
        if isinstance(x, Dataset):
            x_gen = x.map(predict_wrapper)
        else:
            x_gen = map(predict_wrapper, x)
        y_gen = copy.copy(y)
        x_disc = copy.copy(x)
        y_disc = copy.copy(y)

        # alternate between training generator and discriminator
        while d_e < discriminator_epochs and g_e < generator_epochs:
            if d_e < discriminator_epochs:
                LOGGER.info("Training discriminator network.")
                self.discriminator_network.train(x_gen, y_gen, self.discriminator_loss, discriminator_stint, discriminator_lr, discriminator_callbacks)
                d_e += discriminator_stint
            if g_e < generator_epochs:
                LOGGER.info("Training generator network.")
                self.generator_network.train(x_disc, y_disc, self.generator_loss, generator_stint, generator_lr, generator_callbacks)
                g_e += generator_stint

    def evaluate(self, y, y_pred):
        return self.generator_network.evaluate(y, y_pred)