import copy
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union, Callable, Tuple

import cv2
import h5py
import numpy as np

from src.config import *


class Dataset(metaclass=ABCMeta):
    """Abstract base class."""

    @abstractmethod
    def __enter__(self) -> "Dataset":
        """Must set `self._handle`"""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Must clear `self._handle` and free the resources."""
        pass

    @abstractmethod
    def _data(self, indexes: List[int]) -> np.ndarray:
        """Returns the data at given indexes as a sequence.

        https://docs.python.org/3/glossary.html#term-sequence
        """
        pass

    def __init__(self, path: Union[str, pathlib.Path]):
        """
        :param path:  absolute or relative path (from the project's source directory) to the dataset file
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        self._path = ROOT_PATH.joinpath(path).absolute()
        if not self._path.exists():
            raise FileExistsError(f"{self._path} does not exist. {chr(10)}"  # chr(10) = newline
                                  f"For relative paths, root is: {chr(10)} {ROOT_PATH} {chr(10)}"
                                  f"Parameter given: {chr(10)} {path}")
        self._handle = None
        self._iter = Dataset.DatasetIterator(self)

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #
    #     def resource_property(func, *args):
    #         @property
    #         def wrapper(self):
    #             if self._handle is None:
    #                 raise SyntaxError("Datasets are supposed to be called from within a resource block. "
    #                                   "Are you missing a 'with' statement?")
    #             return func(self, args)
    #         return wrapper
    #     cls._data = resource_property(cls._data, None)

    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
        self._iter.reset()
        return self._iter.__iter__()

    def __copy__(self):
        _copy = self.__class__(self._path)
        _copy._iter._args = copy.deepcopy(self._iter._args)
        _copy._iter._map_funcs = copy.deepcopy(self._iter._map_funcs)
        return _copy

    def batch(self, batch_size: int, drop_remainder: bool = False) -> "Dataset":
        """Combines consecutive elements of this dataset into batches.

           The components of the resulting element will have an additional outer
           dimension, which will be `batch_size` (or `N % batch_size` for the last
           element if `batch_size` does not divide the number of input elements `N`
           evenly and `drop_remainder` is `False`). If your program depends on the
           batches having the same outer dimension, you should set the `drop_remainder`
           argument to `True` to prevent the smaller batch from being produced.
           Args:
             batch_size: An integer, representing the number of consecutive elements
                of this dataset to combine in a single batch.
             drop_remainder: (Optional.) A boolean, representing whether the
                 last batch should be dropped in the case it has fewer than
                `batch_size` elements; the default behavior is not to drop the smaller
                batch.
           Returns:
             Dataset: A `Dataset`.
           """
        _copy = copy.copy(self)
        _copy._iter.args = {"step": batch_size, "drop_remainder": drop_remainder}
        return _copy

    def shuffle(self, seed: int = np.random.randint(np.iinfo(np.int32).max), reshuffle_each_iteration: bool = True) -> "Dataset":
        """Randomly shuffles the elements of this dataset.

            Args:
              seed: (Optional.) An integer, representing the random
                seed that will be used to create the distribution.
              reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
                that the dataset should be pseudorandomly reshuffled each time it is
                iterated over. (Defaults to `True`.)
            Returns:
              Dataset: A `Dataset`.
            """
        _copy = copy.copy(self)
        _copy._iter.args = {"shuffle": True, "seed": seed, "reshuffle_each_iteration": reshuffle_each_iteration}
        return _copy

    def repeat(self, count: int = 1) -> "Dataset":
        """Repeats this dataset so each original value is seen `count` times.

            Args:
              count: (Optional.) An integer, representing the number of times
                the dataset should be repeated. The default behavior (if
                `count` is `1`) is for the dataset be repeated once.
            Returns:
              Dataset: A `Dataset`.
            """
        if count <= 0:
            raise ValueError("Repeat count should be a positive integer.")
        _copy = copy.copy(self)
        _copy._iter.args = {"repeat": count}
        return _copy

    def split(self, ratio: Tuple, split_exactly: bool = False) -> List["Dataset"]:
        """Splits the dataset into smaller chunks according to the `ratio` parameter.

            This transformation takes the whole length of the dataset, and splits
            it's contents into smaller parts. Each of this parts' length is proportionate
            to the ratio given by the `ratio` parameter. E.g. a common scenario is to
            split the dataset into 3 parts (train, test, validation) in a manner of 80%-10%-10%
            one would call this function with `ratio=(8,1,1)` or `ratio=(80,10,10)` etc.
            Args:
              ratio: The split ratio.
              split_exactly: If `True`, truncates the number of items in the dataset to be a
              multiple of `sum(ratio)`. This way guaranteeing the exact ratios.
            Returns:
              Dataset: A list of `Dataset`-s. The actual number of datasets are determined by
              the `ratio` parameter.
            """
        _copies = []
        for i in range(len(ratio)):
            _copy = copy.copy(self)
            # append the index of the dataset to the ratio parameter
            r = (*ratio, i)
            _copy._iter.args = {"ratio": r, "split_exactly": split_exactly}
            _copies.append(_copy)
        return _copies

    def map(self, map_func: Callable) -> "Dataset":
        """Maps `map_func` across the elements of this dataset.

            This transformation applies `map_func` to each element of this dataset, and
            returns a new dataset containing the transformed elements, in the same
            order as they appeared in the input. `map_func` can be used to change both
            the values and the structure of a dataset's elements. For example, adding 1
            to each element, or projecting a subset of element components.
            Args:
              map_func: A function mapping a dataset element to another dataset element.
            Returns:
              Dataset: A `Dataset`.
            """
        if not callable(map_func):
            raise TypeError("'func' must be callable.")
        _copy = copy.copy(self)
        _copy._iter.map_funcs += [map_func]
        return _copy

    def transform(self):
        """Sets a transformation flag on the dataset. If the flag is set, it makes the
        dataset return transformed items. Transformations include: rotating 0, 90, 180, 270,
        and vertical flipping."""
        _copy = copy.copy(self)
        _copy._iter.args = {"transform": True, "transform_id": 0}
        return _copy

    class DatasetIterator(metaclass=ABCMeta):

        @property
        def args(self):
            return self._args

        @args.setter
        def args(self, value: Dict):
            self.reset()  # reset iteration when setting new arguments
            for k, v in value.items():
                if k not in self._args.keys():
                    raise KeyError(f"Trying to set a non-existent argument key `{k}`.")
                self._args[k] = v if k in self._args.keys() else self._args[k]

        @property
        def map_funcs(self):
            return self._map_funcs

        @map_funcs.setter
        def map_funcs(self, value):
            self.reset()
            self._map_funcs = value

        def __init__(self, dataset: "Dataset"):
            self.dataset = dataset
            self.i = 0
            self.indexes = None
            self._args = {  # default argument values
                "step": -1,
                "drop_remainder": False,
                "repeat": 1,  # repeat/replicate once
                "shuffle": False,
                "seed": None,
                "reshuffle_each_iteration": True,
                "ratio": [],  # indicates whether the dataset's been already split before (datasets cannot re-split)
                "split_exactly": False,
                "transform": False,  # indicates whether to transform values before returning or not
                "transform_id": 0   # current transformation's id
            }
            self._map_funcs = []

        def __iter__(self):
            if self.indexes is None:
                # try to get the length of the dataset
                _len = len(self.dataset)
                if self._args["ratio"]  == []:
                    self.indexes = np.arange(_len)
                else:
                    _total = np.sum(self._args["ratio"][:-1])
                    # truncate the length if necessary
                    if self._args["split_exactly"] is True and _len % _total != 0:
                        _len -= _len % _total
                    _start = 0 if self._args["ratio"][-1] == 0 else (np.sum(
                            self._args["ratio"][:self._args["ratio"][-1]]) / _total) * _len
                    _end = _start + (self._args["ratio"][self._args["ratio"][-1]] / _total) * _len
                    self.indexes = np.arange(int(_start), int(_end))
                if self._args["step"] == -1:
                    self._args["step"] = _len
                if self._args["shuffle"]:
                    np.random.seed(self._args["seed"])
                    np.random.shuffle(self.indexes)
                if self._args["repeat"] > 1:
                    self.indexes = np.tile(self.indexes, self._args["repeat"])
            return self

        def __next__(self):
            if self.i < len(self.indexes):
                result = self._get_index_data(self.indexes[self.i:self.i + self._args["step"]])
                self.i += self._args["step"]
                # recheck boundaries
                if self.i >= len(self.indexes):
                    if self._args["shuffle"] and self._args["reshuffle_each_iteration"]:
                        self._args["seed"] += 1
                        np.random.seed(self._args["seed"])
                        np.random.shuffle(self.indexes)
                    if self._args["drop_remainder"] is True:
                        raise StopIteration()
                # we got through all the checks
                # apply the built in transformations
                if self._args["transform"] is True:
                    result = np.array(list(map(self._transform, result)), dtype=result.dtype)
                # apply the mapping transformations
                for func in self._map_funcs:
                    result = func(result)
                return result
            else:
                self.reset()
                raise StopIteration()

        def _transform(self, img):
            if self._args["transform_id"] == 0:
                self._args["transform_id"] += 1
                return img
            if self._args["transform_id"] == 1:
                self._args["transform_id"] += 1
                rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                return rot_90
            elif self._args["transform_id"] == 2:
                self._args["transform_id"] += 1
                rot_180 = cv2.rotate(img, cv2.ROTATE_180)
                return rot_180
            elif self._args["transform_id"] == 3:
                self._args["transform_id"] += 1
                rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return rot_270
            elif self._args["transform_id"] == 4:
                self._args["transform_id"] = 0
                flip_vertical = cv2.flip(img, flipCode=0)
                return flip_vertical
            else:
                raise IndexError(f"Wrong transformation index [{self._args['transform_id']}].")

        def reset(self):
            self.i = 0
            self.indexes = None

        def _get_index_data(self, idxs: List[int]) -> np.ndarray:
            """Returns values at positions determined by indexes found in `idxs`.

            :param idxs: list of value indexes to get
            :return: 4D shaped np.ndarray (batch, height, width, channels) with 8-bit integer values
            """
            result = None
            # if the indexes are shuffled, some extra work is needed
            if self._args["shuffle"]:
                idxs_sorted = list(sorted(idxs))
                # h5py supports index ranges, read times are significantly faster
                # than reading each image separately
                # but with random indexes the indexes must be in ascending order
                images_sorted = np.array(self.dataset._data(idxs_sorted), dtype=np.float32)
                # drawback of this implementation, is that the images are read in the wrong order,
                # so we have to unsort them
                positions = dict(zip(idxs_sorted, np.arange(len(idxs_sorted))))
                images = np.zeros_like(images_sorted)
                for i, idx in enumerate(idxs):
                    images[i] = images_sorted[positions[idx]]
                result = images
            else:
                result = np.array(self.dataset._data(list(idxs)), dtype=np.float32)
            # reshape the result if necessary to ensure it's 4D (count, height, width, depth)
            if len(result.shape) < 4:
                result = np.reshape(result, (*result.shape, 1))
            return result


class DirectoryDataset(Dataset):

    def __init__(self, path: Union[str, pathlib.Path]):
        super().__init__(path)
        if not self._path.is_dir():
            raise ValueError(f"{self._path} is not a directory.")
        self._supported_extensions: List[str] = ["jpg", "png"]

    def __enter__(self) -> Dataset:
        image_paths = []
        for ext in self._supported_extensions:
            image_paths.extend(list(self._path.rglob(f"*.{ext}")))
        self._handle = image_paths
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._handle = None
        return False

    def _data(self, indexes: List[int]) -> np.ndarray:
        return np.array([cv2.imread(str(self._handle[idx]), cv2.IMREAD_COLOR) for idx in indexes], dtype=np.float32)

    def __len__(self):
        return len(self._handle)


class HDFDataset(Dataset):

    def __init__(self, path: Union[str, pathlib.Path]):
        super().__init__(path)
        if not h5py.is_hdf5(str(self._path)):
            raise ValueError(f"{self._path} is not a valid hdf5 file.")

    def __enter__(self) -> Dataset:
        # defer opening the file until now, this way one can check for the None reference
        # in other functions before actually accessing the file
        self._handle = h5py.File(str(self._path), mode="r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close the file handle
        self._handle.close()
        self._handle = None  # clear reference, otherwise a closed hdf file is tricky to detect
        # re-raise any other exception that might have happened
        return False

    def _data(self, indexes: List[int]) -> np.ndarray:
        return self._handle["images"][indexes]

    def __len__(self):
        return len(self._handle["images"])
