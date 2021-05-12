import pathlib
from abc import ABCMeta
from typing import Iterable, Union

import cv2
import json

from src.config import LOGGER, ROOT_PATH


class Callback(metaclass=ABCMeta):

    def call(self, inst):
        """
        Callbacks are callable.
        :param inst: a `Network` class instance the callback is working with.
        """
        raise NotImplementedError(f"`{self.__class__.__name__}.call(self, inst)` not implemented.")

    def __call__(self, inst):
        return self.call(inst)


class OptimizerCallback(Callback, metaclass=ABCMeta):
    """
    Special type of callbacks intended to be used with optimizers.
    DOES NOT WORK IN TENSORFLOW 2.1.0 or 2.2.0 with @tf.function decorators
    https://github.com/tensorflow/tensorflow/issues/31323

    Workaround:
        Wrap the `learning_rate` parameter inside a `tf.Variable()` object,
        and update the variable manually by calling `learning_rate.assign(new_value)`.

        OR

        Get rid of all @tf.function decorators.
    """
    # def __call__(self, inst):
    #     self._inst = inst
    #
    #     def wrapper():
    #         """
    #         According to https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/optimizer_v2/adam.py#L129
    #         tensorflow optimizers in eager mode take a no-parameter callable. This function therefore
    #         is the one, that actually gets called. It's purpose is to wrap/preserve the given instance reference.
    #         """
    #         return self.call(self._inst)
    #
    #     return wrapper


class TrainIterationEndCallback(Callback, metaclass=ABCMeta):
    """
    Special type of callbacks intended to be called at the end of a training iteration/step.
    """
    pass


class ExponentialDecayCallback(OptimizerCallback):

    def __init__(self, initial_learning_rate: float, decay_steps: int, decay_rate: float, staircase=False):
        """
        Produces a decayed learning rate when passed the current optimizer step.
        This can be useful for changing the learning rate value across different
        invocations of optimizer functions. It is computed as:

        ```python
        def decayed_learning_rate(step):
            return initial_learning_rate * decay_rate ^ (step / decay_steps)
        ```

        If the argument `staircase` is `True`, then `step / decay_steps` is
        an integer division and the decayed learning rate follows a
        staircase function.

        Arguments:
            initial_learning_rate: A scalar. The initial learning rate.
            decay_steps: A scalar. Must be positive. See the decay computation above.
            decay_rate: A scalar. The decay rate.
            staircase: Boolean.  If `True` decay the learning rate at discrete intervals
        """
        self._initial_learning_rate = initial_learning_rate
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        self._staircase = staircase

    def call(self, inst):
        return self.decayed_learning_rate(inst.state.epochs)

    def decayed_learning_rate(self, step):
        if self._staircase:
            return self._initial_learning_rate * self._decay_rate ** (step // self._decay_steps)
        else:
            return self._initial_learning_rate * self._decay_rate ** (step / self._decay_steps)


class DecayLrOnPlateauCallback(OptimizerCallback):

    def __init__(self,
                 initial_learning_rate: float,
                 monitor: str = "train_loss",
                 factor: float = 0.1,
                 patience: int = 10,
                 mode: str = "min",
                 min_delta: float = 1e-4,
                 min_learning_rate: float=0):

        """
        Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This callback monitors a
        quantity and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.

        Arguments:
            initial_learning_rate: A scalar. The initial learning rate.
            monitor: quantity to be monitored.
            factor: factor by which the learning rate will be reduced. new_lr = lr * factor
            patience: number of epochs with no improvement after which learning rate
            will be reduced.
            mode: one of {min, max}. In `min` mode, lr will be reduced when the
            quantity monitored has stopped decreasing, in `max` mode it will be
            reduced when the quantity monitored has stopped increasing.
            min_delta: threshold for measuring the new optimum, to only focus on
            significant changes.
            min_learning_rate: lower bound on the learning rate.
        """
        self._learning_rate = initial_learning_rate
        self._monitor = monitor
        self._factor = factor
        self._patience = patience
        self._mode = mode
        self._min_delta = min_delta
        self._min_learning_rate = min_learning_rate
        self._current_patience = 0
        self._reference_value = None

    def call(self, inst):
        # validate parameters
        assert self._monitor in inst.state.__dict__.keys(), f"Monitored quantity `{self._monitor}` is not a NetworkState attribute."
        assert self._mode in ["min", "max"]

        if self._reference_value is None:
            self._reference_value = inst.state.__dict__[self._monitor]

        if (self._mode == "min" and inst.state.__dict__[self._monitor] > self._reference_value + self._min_delta) or \
            (self._mode == "max" and inst.state.__dict__[self._monitor] < self._reference_value - self._min_delta):
            self._current_patience += 1
        else:
            self._reference_value = inst.state.__dict__[self._monitor]
            self._current_patience = 0

        if self._current_patience >= self._patience:
            # check whether there is still room for reduction
            LOGGER.info(f"`{self._monitor}` did not {'increase' if self._mode == 'max' else 'decrease'} for {self._patience} epochs.")
            if self._learning_rate > self._min_learning_rate + 1e-10:
                self._learning_rate *= self._factor
                self._current_patience = 0
                self._reference_value = inst.state.__dict__[self._monitor]
                LOGGER.info(f"Reducing learning rate to {self._learning_rate}")
            else:
                LOGGER.info(f"Learning rate is already at minimum value.")
        return self._learning_rate


class TrainingCheckpointCallback(TrainIterationEndCallback):

    def __init__(self,
                 appendix: str = "",
                 save_best_only: bool = True,
                 monitor: str = "train_loss",
                 mode: str = "min",
                 save_freq: int = 10):
        """
        Callback for saving the model after `save_freq` number of epochs.
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
        self._appendix = appendix
        self._save_best_only = save_best_only
        self._monitor = monitor
        self._mode = mode
        self._save_freq = save_freq

    def call(self, inst):
        # don't save the model if it's not time yet
        if inst.state.epochs <= 0 or inst.state.epochs % self._save_freq != 0:
            return
        # validate parameters
        assert self._monitor in inst.state.__dict__.keys(), f"Monitored quantity `{self._monitor}` is not a NetworkState attribute."
        assert self._mode in ["min", "max"]
        # don't save the model either if it doesn't meet the requirements
        if self._save_best_only is True:
            if inst._saved_state is not None:
                if self._mode == "min" and inst.state.__dict__[self._monitor] >= inst._saved_state.__dict__[self._monitor]:
                    LOGGER.info(f"Skipping saving. `{self._monitor}` current >= best : "
                                f"[{inst.state.__dict__[self._monitor]:.4f}] >= [{inst._saved_state.__dict__[self._monitor]:.4f}]")
                    return
                elif self._mode == "max" and inst.state.__dict__[self._monitor] <= inst._saved_state.__dict__[self._monitor]:
                    LOGGER.info(f"Skipping saving. `{self._monitor}` current <= best : "
                                f"[{inst.state.__dict__[self._monitor]:.4f}] <= [{inst._saved_state.__dict__[self._monitor]:.4f}]")
                    return
        # checks passed, save the model
        LOGGER.info(f"Saving state after {inst.state.epochs} epochs.")
        inst.save_state(self._appendix)


class TrainingEvaluationCallback(TrainIterationEndCallback):

    def __init__(self,
                 x: Iterable,
                 y: Iterable,
                 dest_dir: Union[str, pathlib.Path] = "./evaluations",
                 save_freq: int = 10,
                 *args,
                 **kwargs):
        """
        Callback for evaluating the model after `save_freq` number of epochs. At every trigger
        it feeds the values from the `x` parameter into the present state of the model. The results
        of the prediction are then compared with `y` using various metrics (see `Network.evaluate`),
        and get stored under `dest_dir` directory path.

        :param x: The LR dataset to use for evaluation. Must be iterable.
        :param y: The HR dataset to use for evaluation. Must be iterable. Must have the same number of items as `x`.
        :param dest_dir: The output directory for the predicted images. Relative to project directory (default: ./evaluations).
        :param save_freq: number of epochs between each checkpoint (default: 10).
        :param args, kwargs: Optional arguments for predictions. See `Network.predict` on usage.

        """

        self._x = x
        self._y = y
        self._dest_dir = ROOT_PATH.joinpath(dest_dir)
        self._save_freq = save_freq
        self._args = args
        self._kwargs = kwargs

    def call(self, inst):
        # don't evaluate if it's not time yet
        if inst.state.epochs <= 0 or inst.state.epochs % self._save_freq != 0:
            return
        # validate that the dest_dir directory exists, create if necessary
        if not self._dest_dir.exists() or not self._dest_dir.is_dir():
            LOGGER.warning(f"Model directory {self._dest_dir} does not exist. Creating it.")
            self._dest_dir.mkdir(parents=True, exist_ok=False)

        y_pred = inst.predict(self._x)
        metrics = inst.evaluate(y_pred, self._y)
        # persist results
        file = self._dest_dir.joinpath("eval.json")
        contents = {}
        if file.exists():
            with open(str(file), mode="r") as logfile:
                contents = json.load(logfile)
        metrics_json = []
        for item in metrics:
            t = {}
            for k, v in item.items():
                t[k] = str(v)
            metrics_json.append(t)
        contents[inst.state.epochs] = {
            "train_loss": str(inst.state.train_loss),
            "train_time": str(inst.state.train_time),
            "metrics": metrics_json
        }
        with open(str(self._dest_dir.joinpath("eval.json")), mode="w") as logfile:
            # logfile.write(f"epoch: {inst.state.epochs} loss: {inst.state.train_loss} train_time: {inst.state.train_time}")
            # logfile.write(str(metrics))
            json.dump(contents, logfile, indent=4)

        for i, img in enumerate(y_pred.numpy()):
            cv2.imwrite(str(self._dest_dir.joinpath(f"{inst.state.epochs}_{i}.png")), img*255.0)
