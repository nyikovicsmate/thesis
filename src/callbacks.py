from abc import ABC, abstractmethod
from typing import Callable


class Callback(ABC, Callable):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, env):
        """
        :param env: the environment, the callback is working with (usually a class instance).
        """
        pass
