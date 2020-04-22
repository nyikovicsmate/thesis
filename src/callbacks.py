from abc import ABC, abstractmethod
from typing import Callable

from src.config import LOGGER


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


class TrainingCheckpointCallback(Callback):

    def __init__(self,
                 appendix: str = "",
                 save_best_only: bool = True,
                 monitor: str = "train_loss",
                 mode: str = "min",
                 save_freq: int = 10):
        """Callback for saving the model after `save_freq` number of epochs.
            `appendix` can contain any valid string, that will be appended to the default directory name
            when saving the model. For example: if `appendix` is `model_1`, and the network type is
            `PreUpsamplingNetwork` then the model checkpoints will be saved under
            "./checkpoints/preupsamplingnetwork_model_1" directory.
            Arguments:
                appendix: string, to append to the directory name (default: "").
                save_best_only: if `save_best_only=True`, the latest best model according
                monitor: quantity to monitor, a NetworkState attribute (default: train_loss).
                mode: one of {min, max}. If `save_best_only=True`, the decision to
                    overwrite the current save file is made based on either the maximization
                    or the minimization of the monitored quantity. For `val_acc`, this
                    should be `max`, for `val_loss` this should be `min`, etc.
                save_freq: number of epochs between each checkpoint (default: 10).
            """
        super().__init__()
        self._appendix = appendix
        self._save_best_only = save_best_only
        self._monitor = monitor
        self._mode = mode
        self._save_freq = save_freq

    # noinspection PyProtectedMember
    def __call__(self, env):
        """
        :param env: the environment, the callback is working with (usually a class instance).
        """
        # don't save the model if it's not time yet
        if env._current_state.epochs <= 0 or env._current_state.epochs % self._save_freq != 0:
            return
        # validate parameters
        assert self._monitor in env._current_state.__dict__.keys(), f"Monitored quantity `{self._monitor}` is not a NetworkState attribute."
        assert self._mode in ["min", "max"]
        # don't save the model either if it doesn't meet the requirements
        if self._save_best_only is True:
            if env._saved_state is not None:
                if self._mode == "min" and env._current_state.__dict__[self._monitor] >= env._saved_state.__dict__[self._monitor]:
                    LOGGER.info(f"Skipping saving. `{self._monitor}` current >= best : "
                                f"[{env._current_state.__dict__[self._monitor]:.4f}] >= [{env._saved_state.__dict__[self._monitor]:.4f}]")
                    return
                elif self._mode == "max" and env._current_state.__dict__[self._monitor] <= env._saved_state.__dict__[self._monitor]:
                    LOGGER.info(f"Skipping saving. `{self._monitor}` current <= best : "
                                f"[{env._current_state.__dict__[self._monitor]:.4f}] <= [{env._saved_state.__dict__[self._monitor]:.4f}]")
                    return
        # checks passed, save the model
        LOGGER.info(f"Saving state after {env._current_state.epochs} epochs.")
        env.save_state(self._appendix)