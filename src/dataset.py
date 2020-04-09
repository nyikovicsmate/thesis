import copy

import h5py
import pathlib
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Callable


# TODO: unit tests


class Dataset(ABC):
    """Abstract base class."""

    def __init__(self, path: Union[str, pathlib.Path]):
        """
        :param path:  absolute or relative path (from the project's source directory) to the dataset file
        """
        # TODO: import globals, root path sould not depend on this exact file's position
        self._root = pathlib.Path(__file__).parent.parent
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        self._path = self._root.joinpath(path).absolute()
        if not self._path.exists():
            raise FileExistsError(f"{self._path} does not exist. {chr(10)}"  # chr(10) = newline
                                  f"For relative paths, root is: {chr(10)} {self._root.absolute()} {chr(10)}"
                                  f"Parameter given: {chr(10)} {path}")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __next__(self):
        raise NotImplementedError()

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()

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
        raise NotImplementedError()

    @abstractmethod
    def repeat(self, count: int = -1) -> "Dataset":
        """Repeats this dataset so each original value is seen `count` times.

            Args:
              count: (Optional.) An integer, representing the number of times
                the dataset should be repeated. The default behavior (if
                `count` is `-1`) is for the dataset be repeated indefinitely.
            Returns:
              Dataset: A `Dataset`.
            """
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()


class HDFDataset(Dataset):

    def __init__(self, path: Union[str, pathlib.Path]):
        super().__init__(path)
        if not h5py.is_hdf5(str(self._path)):
            raise ValueError(f"{self._path} is not a valid hdf5 file.")
        self._file = None
        self._iter = self.HDFDatasetIterator(self)

    def __iter__(self):
        return self._iter

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
        _copy._iter._args = copy.deepcopy(self._iter._args)
        _copy._iter._map_funcs = copy.deepcopy(self._iter._map_funcs)
        return _copy

    @property
    def _images_dataset(self):
        if self._file is None:
            raise SyntaxError("Datasets are supposed to be called from within a resource block. "
                              "Are you missing a 'with' statement?")
        return self._file["images"]

    def batch(self, batch_size: int, drop_remainder: bool = False) -> "HDFDataset":
        _copy = copy.copy(self)
        _copy._iter.args = {"step": batch_size, "drop_remainder": drop_remainder}
        return _copy

    def shuffle(self, seed: int = None, reshuffle_each_iteration: bool = True) -> "HDFDataset":
        _copy = copy.copy(self)
        _copy._iter.args = {"shuffle": True, "seed": seed, "reshuffle_each_iteration": reshuffle_each_iteration}
        return _copy

    def repeat(self, count: int = -1) -> "HDFDataset":
        _copy = copy.copy(self)
        _copy._iter.args = {"repeat": count}
        return _copy

    def map(self, map_func) -> "HDFDataset":
        if not callable(map_func):
            raise TypeError("'func' must be callable.")
        # vectorize the function to make it able to be used on np arrays later
        map_func = np.vectorize(map_func)
        _copy = copy.copy(self)
        _copy._iter.map_funcs += [map_func]
        return _copy

    class HDFDatasetIterator:

        def __init__(self, dataset: "HDFDataset"):
            self._dataset = dataset
            self.i = 0
            self.indexes = None
            self._args = {  # default argument values
                "step": -1,
                "drop_remainder": False,
                "repeat": -1,
                "shuffle": False,
                "seed": np.random.randint(np.iinfo(np.int32).max),
                "reshuffle_each_iteration": True
            }
            self._map_funcs = []

        def __iter__(self):
            return self

        def __next__(self):
            if self.indexes is None:
                # try to get the length of the dataset
                # if the dataset isn't used within a resource block the getter will throw a TypeError
                _len = len(self._dataset._images_dataset)
                self.indexes = np.arange(_len)
                if self._args["step"] == -1:
                    self._args["step"] = _len
                if self._args["shuffle"]:
                    np.random.seed(self._args["seed"])
                    np.random.shuffle(self.indexes)
            if self.i < len(self.indexes):
                result = self._get_index_data(self.indexes[self.i:self.i + self._args["step"]])
                self.i += self._args["step"]
                # recheck boundaries
                if self.i >= len(self.indexes):
                    if self._args["drop_remainder"] is True:
                        if self._args["repeat"] == -1 or self._args["repeat"] > 0:
                            # if we drop the trunctated result, but have more repeats
                            # get new results
                            self.i = 0
                            if self._args["shuffle"] and self._args["reshuffle_each_iteration"]:
                                np.random.shuffle(self.indexes)
                            result = self._get_index_data(self.indexes[self.i:self.i + self._args["step"]])
                            # decrease repeat counter
                            self._args["repeat"] = -1 if self._args["repeat"] == -1 else self._args["repeat"] - 1
                        else:
                            # if we drop the trunctated result, and have no more repeat counts
                            # then invalidate this (last) iteration
                            raise StopIteration()
                    else:
                        if self._args["repeat"] == -1 or self._args["repeat"] > 0:
                            # if we don't drop this result and have more repeats
                            # reset the index
                            self.i = 0
                            if self._args["shuffle"] and self._args["reshuffle_each_iteration"]:
                                np.random.shuffle(self.indexes)
                            # decrease repeat counter
                            self._args["repeat"] = -1 if self._args["repeat"] == -1 else self._args["repeat"] - 1
                # we got through all the checks
                # apply the mapping transformations
                for func in self._map_funcs:
                    result = func(result)
                return result
            else:
                raise StopIteration()

        @property
        def args(self):
            return self._args

        @args.setter
        def args(self, value: Dict):
            self._reset()  # reset iteration when setting new arguments
            for k, v in value.items():
                self._args[k] = v if k in self._args.keys() else self._args[k]

        @property
        def map_funcs(self):
            return self._map_funcs

        @map_funcs.setter
        def map_funcs(self, value):
            self._reset()
            self._map_funcs = value

        def _reset(self):
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