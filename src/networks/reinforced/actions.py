from abc import ABC, abstractmethod
import numpy as np
import cv2

from src.kernel import *


class Action(ABC):
    @abstractmethod
    def apply(self,
              img: np.ndarray,
              i: int,
              j: int) -> np.float32:
        """
        // TODO
        :param img: [0,1] scaled image pixel mx
        :param i: y coordinate [0,m]
        :param j: x coordinate [0,n]
        :return:
        """
        raise NotImplementedError()


class DoNothing(Action):
    def apply(self, img: np.ndarray, i: int, j: int) -> np.float32:
        return img[i, j]


class IncrementByOne(Action):
    def apply(self, img: np.ndarray, i: int, j: int) -> np.float32:
        return img[i, j] + (1.0 / 255.0)


class DecrementByOne(Action):
    def apply(self, img: np.ndarray, i: int, j: int) -> np.float32:
        return img[i, j] - (1.0 / 255.0)


class Gaussian1(Action):
    _kernel = GaussianKernel(size=(5,5), sigma=0.5)

    def apply(self, img: np.ndarray, i: int, j: int) -> np.float32:
        return self._kernel.apply_to_region(img, i, j)


class Gaussian2(Action):
    _kernel = GaussianKernel(size=(5,5), sigma=1.5)

    def apply(self, img: np.ndarray, i: int, j: int) -> np.float32:
        return self._kernel.apply_to_region(img, i, j)

