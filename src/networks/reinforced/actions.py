from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np


class Action(ABC):
    def __init__(self, img: np.ndarray):
        self._img = np.array(img, dtype=np.float32)
        self._processed_img = None

    @property
    def processed_img(self) -> np.ndarray:
        if self._processed_img is None:
            self._processed_img = np.reshape(self._process(), self._img.shape)
        return self._processed_img

    @abstractmethod
    def _process(self) -> np.ndarray:
        pass


class AlterByValue(Action):
    def __init__(self, img, value):
        super().__init__(img)
        self._value = value

    def _process(self) -> np.ndarray:
        return np.array(self._img + self._value, dtype=np.float32)


class GaussianBlur(Action):
    def __init__(self, img: np.ndarray, ksize: Tuple[int, int], sigmaX: float):
        super().__init__(img)
        self._ksize = ksize
        self._sigmaX = sigmaX

    def _process(self) -> np.ndarray:
        return cv2.GaussianBlur(self._img, ksize=self._ksize, sigmaX=self._sigmaX)


class BilaterFilter(Action):
    def __init__(self, img: np.ndarray, d: int, sigma_color: float, sigma_space: float):
        super().__init__(img)
        self._d = d
        self._sigma_color = sigma_color
        self._sigma_space = sigma_space

    def _process(self) -> np.ndarray:
        return cv2.bilateralFilter(self._img, d=self._d, sigmaColor=self._sigma_color,
                                                  sigmaSpace=self._sigma_space)


class BoxFilter(Action):
    def __init__(self, img: np.ndarray, ddepth: int, ksize: Tuple[int, int]):
        super().__init__(img)
        self._ddepth = ddepth
        self._ksize = ksize

    def _process(self) -> np.ndarray:
        return cv2.boxFilter(self._img, ddepth=self._ddepth, ksize=self._ksize)


class MedianBlur(Action):
    def __init__(self, img: np.ndarray, ksize: int):
        super().__init__(img)
        self._ksize = ksize

    def _process(self) -> np.ndarray:
        return cv2.medianBlur(self._img, ksize=self._ksize)
