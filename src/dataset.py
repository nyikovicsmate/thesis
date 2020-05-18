import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Callable, Tuple

import cv2
import h5py
import numpy as np

from src.config import *


# TODO: unit tests


class Dataset(ABC):
    """Abstract base class."""

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

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def as_numpy_iterator(self) -> "Dataset.DatasetIterator":
        """Returns an iterator which converts all elements of the dataset to numpy."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def repeat(self, count: int = 1) -> "Dataset":
        """Repeats this dataset so each original value is seen `count` times.

            Args:
              count: (Optional.) An integer, representing the number of times
                the dataset should be repeated. The default behavior (if
                `count` is `1`) is for the dataset be repeated once.
            Returns:
              Dataset: A `Dataset`.
            """
        pass

    @abstractmethod
    def shuffle(self, seed: int = None, reshuffle_each_iteration: bool = True) -> "Dataset":
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    class DatasetIterator(ABC):

        @abstractmethod
        def __iter__(self):
            pass

        @abstractmethod
        def __next__(self):
            pass

        @abstractmethod
        def advance(self):
            """Skips an iteration. Mainly used to keep iterators in sync."""
            pass

        @abstractmethod
        def reset(self):
            """Resets the iterator state."""
            pass


class DirectoryDataset(Dataset):

    def __init__(self, path: Union[str, pathlib.Path]):
        super().__init__(path)
        if not self._path.is_dir():
            raise ValueError(f"{self._path} is not a directory.")
        self._supported_extensions: List[str] = ["jpg", "png"]
        self._image_paths = self._get_image_paths()
        self._iter = self.DirectoryDatasetIterator(self)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, item):
        return DirectoryDataset._get_image(self._image_paths[item])

    def __iter__(self):
        return iter(self._iter)

    def __next__(self):
        return next(iter(self))

    def __enter__(self):
        # TODO: technically the image loading should happen here
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: if `__enter__` loads the images, this should clear the reference
        return False

    def as_numpy_iterator(self) -> "Dataset.DatasetIterator":
        raise NotImplementedError()

    def batch(self, batch_size: int, drop_remainder: bool = False) -> "Dataset":
        raise NotImplementedError()

    def repeat(self, count: int = 1) -> "Dataset":
        raise NotImplementedError()

    def shuffle(self, seed: int = None, reshuffle_each_iteration: bool = True) -> "Dataset":
        raise NotImplementedError()

    def split(self, ratio: Tuple, split_exactly: bool = False) -> List["Dataset"]:
        raise NotImplementedError()

    def map(self, map_func: Callable) -> "Dataset":
        raise NotImplementedError()

    def _get_image_paths(self) -> List[pathlib.Path]:
        """
        Returns a list of absolute image paths found recursively starting from the scripts directory.
        """
        image_paths = []
        for ext in self._supported_extensions:
            image_paths.extend(list(self._path.rglob(f"*.{ext}")))
        return image_paths

    @staticmethod
    def _get_image(path: Union[str, pathlib.Path]):
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    class DirectoryDatasetIterator(Dataset.DatasetIterator):

        def __init__(self, dataset: "DirectoryDataset"):
            self._dataset = dataset

        def __iter__(self):
            self.i = 0
            return self

        def __next__(self):
            if self.i < len(self._dataset):
                result = DirectoryDataset._get_image(self._dataset._image_paths[self.i])
                self.i += 1
                return result
            else:
                raise StopIteration()

        def advance(self):
            raise NotImplementedError()

        def reset(self):
            raise NotImplementedError()


class HDFDataset(Dataset):

    def __init__(self, path: Union[str, pathlib.Path]):
        super().__init__(path)
        if not h5py.is_hdf5(str(self._path)):
            raise ValueError(f"{self._path} is not a valid hdf5 file.")
        self._file = None
        self._iter_inst = None

    def __len__(self):
        return len(self._images_dataset)

    def __getitem__(self, item):
        return self._images_dataset[item]

    def __iter__(self):
        self._iter = copy.copy(self._iter)
        return iter(self._iter)

    def __next__(self):
        return next(iter(self))

    def __enter__(self):
        # defer opening the file until now, this way one can check for the None reference
        # in other functions before actually accessing the file
        self._file = h5py.File(str(self._path), mode="r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close the file handle
        self._file.close()
        self._file = None  # clear reference, otherwise a closed hdf file is tricky to detect
        # re-raise any other exception that might have happened
        return False

    def __copy__(self):
        _copy = HDFDataset(self._path)
        _copy._file = self._file    # don't lose the file object
        _copy._iter._args = copy.deepcopy(self._iter._args)
        _copy._iter._map_funcs = copy.deepcopy(self._iter._map_funcs)
        return _copy

    @property
    def _images_dataset(self):
        if self._file is None:
            raise SyntaxError("Datasets are supposed to be called from within a resource block. "
                              "Are you missing a 'with' statement?")
        return self._file["images"]

    @property
    def _iter(self) -> "HDFDataset.HDFDatasetIterator":
        if self._iter_inst is None:
            self._iter_inst = self.HDFDatasetIterator(self)
        return self._iter_inst

    @_iter.setter
    def _iter(self, value):
        self._iter_inst = value

    def as_numpy_iterator(self):
        return iter(self)

    def batch(self, batch_size: int, drop_remainder: bool = False) -> "HDFDataset":
        _copy = copy.copy(self)
        _copy._iter.args = {"step": batch_size, "drop_remainder": drop_remainder}
        return _copy

    def shuffle(self, seed: int = np.random.randint(np.iinfo(np.int32).max), reshuffle_each_iteration: bool = True) -> "HDFDataset":
        _copy = copy.copy(self)
        _copy._iter.args = {"shuffle": True, "seed": seed, "reshuffle_each_iteration": reshuffle_each_iteration}
        return _copy

    def repeat(self, count: int = 1) -> "HDFDataset":
        if count <= 0:
            raise ValueError("Repeat count should be a positive integer.")
        _copy = copy.copy(self)
        _copy._iter.args = {"repeat": count}
        return _copy

    def split(self, ratio: Tuple, split_exactly: bool = False) -> List["Dataset"]:
        _copies = []
        for i in range(len(ratio)):
            _copy = copy.copy(self)
            # append the index of the dataset to the ratio parameter
            r = (*ratio, i)
            _copy._iter.args = {"ratio": r, "split_exactly": split_exactly}
            _copies.append(_copy)
        return _copies

    def map(self, map_func) -> "HDFDataset":
        if not callable(map_func):
            raise TypeError("'func' must be callable.")
        _copy = copy.copy(self)
        _copy._iter.map_funcs += [map_func]
        return _copy

    class HDFDatasetIterator(Dataset.DatasetIterator):

        def __init__(self, dataset: "HDFDataset"):
            self._dataset = dataset
            self.i = 0
            self.indexes = None
            self._args = {  # default argument values
                "step": -1,
                "drop_remainder": False,
                "repeat": 1,    # repeat/replicate once
                "shuffle": False,
                "seed": None,
                "reshuffle_each_iteration": True,
                "ratio": None,   # indicates whether the dataset's been already split before (datasets cannot re-split)
                "split_exactly": False
            }
            self._map_funcs = []

        def __iter__(self):
            if self.indexes is None:
                # try to get the length of the dataset
                # if the dataset isn't used within a resource block the getter will throw a TypeError
                _len = len(self._dataset)
                if self._args["ratio"] is None:
                    self.indexes = np.arange(_len)
                else:
                    _total = np.sum(self._args["ratio"][:-1])
                    # truncate the length if necessary
                    if self._args["split_exactly"] is True and _len % _total != 0:
                        _len -= _len % _total
                    _start = 0 if self._args["ratio"][-1] == 0 else (np.sum(self._args["ratio"][:self._args["ratio"][-1]]) / _total) * _len
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
                # apply the mapping transformations
                for func in self._map_funcs:
                    result = func(result)
                return result
            else:
                raise StopIteration()

        def __copy__(self):
            _copy = HDFDataset.HDFDatasetIterator(self._dataset)
            _copy.indexes = self.indexes
            _copy._args = self._args
            _copy._map_funcs = self._map_funcs
            return _copy

        @property
        def args(self):
            return self._args

        @args.setter
        def args(self, value: Dict):
            self.reset()  # reset iteration when setting new arguments
            for k, v in value.items():
                self._args[k] = v if k in self._args.keys() else self._args[k]

        @property
        def map_funcs(self):
            return self._map_funcs

        @map_funcs.setter
        def map_funcs(self, value):
            self.reset()
            self._map_funcs = value

        def advance(self):
            self.i += self.args["step"]

        def reset(self):
            self.i = 0
            self.indexes = None

        def _get_index_data(self, idxs: List[int]):
            result = None
            # if the indexes are shuffled, some extra work is needed
            if self._args["shuffle"]:
                idxs_sorted = list(sorted(idxs))
                # h5py supports index ranges, read times are significantly faster
                # than reading each image separately
                # but with random indexes the indexes must be in ascending order
                images_sorted = np.array(self._dataset._images_dataset[idxs_sorted], dtype=np.uint8)
                # drawback of this implementation, is that the images are read in the wrong order,
                # so we have to unsort them
                positions = dict(zip(idxs_sorted, np.arange(len(idxs_sorted))))
                images = np.zeros_like(images_sorted)
                for i, idx in enumerate(idxs):
                    images[i] = images_sorted[positions[idx]]
                result = images
            else:
                result = np.array(self._dataset._images_dataset[list(idxs)], dtype=np.uint8)
            # reshape the result if necessar to ensure it's 4D (count, height, width, depth)
            if len(result.shape) < 4:
                result = np.reshape(result, (*result.shape, 1))
            return result
