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

    def save_state(self,
                   generator_appendix: str = "",
                   discriminator_appendix: str = ""):
        discriminator_appendix = f"_{str.lower(self.__class__.__name__)}" if discriminator_appendix == "" else discriminator_appendix
        self.generator_network.save_state(generator_appendix)
        self.discriminator_network.save_state(discriminator_appendix)

    def load_state(self,
                   generator_appendix: str = "",
                   discriminator_appendix: str = ""):
        discriminator_appendix = f"_{str.lower(self.__class__.__name__)}" if discriminator_appendix == "" else discriminator_appendix
        self.generator_network.load_state(generator_appendix)
        self.discriminator_network.load_state(discriminator_appendix)

    def predict(self, x: Iterable, *args, **kwargs):
        return self.generator_network.predict(x, *args, **kwargs)

    def train(self,
              generator_x: Iterable,
              generator_y: Iterable,
              discriminator_x: Iterable,
              discriminator_y: Iterable,
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
        assert generator_x is not discriminator_x, "Low res datasets `x` must to be unique for both the generator and the discriminator."
        assert generator_y is not discriminator_y, "High res datasets `y` must to be unique for both the generator and the discriminator."
        discriminator_x = discriminator_x.map(predict_wrapper)   # TODO: fix, this mapping approach only works if x is of type `Dataset`

        # alternate between training generator and discriminator
        while d_e < discriminator_epochs and g_e < generator_epochs:
            if d_e < discriminator_epochs:
                LOGGER.info("Training discriminator network.")
                self.discriminator_network.train(discriminator_x, discriminator_y, self.discriminator_loss, discriminator_stint, discriminator_lr, discriminator_callbacks)
                d_e += discriminator_stint
            if g_e < generator_epochs:
                LOGGER.info("Training generator network.")
                self.generator_network.train(generator_x, generator_y, self.generator_loss, generator_stint, generator_lr, generator_callbacks)
                g_e += generator_stint

    def evaluate(self, y, y_pred):
        return self.generator_network.evaluate(y, y_pred)
