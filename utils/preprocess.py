"""
Preprocessing script.

usage: preprocess.py [-h] [-a AUGMENT_VALUE] [-f {png,hdf,lmdb}] [-g]
                     [-m {clip,clip_rnd,scale,scale_rnd}] [-s SIZE SIZE]
                     [dst] [src [src ...]]

positional arguments:
  dst                   Name of the output file/directory. Can also be path
                        e.g.: './results_png' or './a/b/hdf.h5'. (default:
                        './dataset')
  src                   Name(s) of the source directory(s) from where the
                        search for images starts. Can also be path(s).
                        (default: '.')

optional arguments:
  -h, --help            show this help message and exit
  -a AUGMENT_VALUE, --augment AUGMENT_VALUE
                        Besides preprocessed images, store augmented ones as
                        well. Augmented image is a processed image with every
                        2nd pixel (in a checkerboard pattern) set to
                        augment_value [0-255].
  -f {png,hdf,lmdb}, --format {png,hdf,lmdb}
                        Output format to use. Supported: png, hdf, lmdb.
                        (default: png)
  -g, --grayscale       Grayscale images.
  -m {clip,clip_rnd,scale,scale_rnd}, --method {clip,clip_rnd,scale,scale_rnd}
                        Processing method to use. (default: scale)
  -s SIZE SIZE, --size SIZE SIZE
                        Output size of images [height, width]. (default: [70,
                        70])

"""
import argparse
import pathlib
import h5py
import lmdb
import cv2
import pickle
from typing import List, Tuple, Callable
import numpy as np
from tqdm import tqdm
from abc import ABC
import copy
import rawpy


class Image:
    def __init__(self, path: pathlib.Path):
        self.path = path
        if self.path.suffix == ".dng":
            with rawpy.imread(str(self.path)) as raw:
                rgb = raw.postprocess()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self.data = np.array(bgr, dtype=np.uint8)
        else:
            self.data = np.array(cv2.imread(str(self.path), cv2.IMREAD_COLOR), dtype=np.uint8)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self) -> Tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]

    def grayscale(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)

    def resize(self, size: Tuple[int, int]):
        self.data = cv2.resize(self.data, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

    def augment(self, value: int) -> "Image":
        augmented_image = copy.deepcopy(self)
        for i in range(augmented_image.size[0]):
            if i % 2 == 0:
                augmented_image.data[i][0::2] = value
            else:
                augmented_image.data[i][1::2] = value
        return augmented_image


class Store(ABC):
    def __init__(self,
                 path: pathlib.Path,
                 augment: bool):
        """
        :param path: absolute path of the dataset file/directory
        """
        self.initialized = False
        self.augment = augment
        self.path = path if path.is_absolute() else path.absolute()

    def initialize(self):
        raise NotImplementedError()

    def store(self, index: int, image: Image):
        if not self.initialized:
            self.initialize()
        self._store(index, image)

    def store_augmented(self, index: int, image: Image):
        if not self.initialized:
            self.initialize()
        self._store_augmented(index, image)

    def _store(self, index: int, image: Image):
        raise NotImplementedError()

    def _store_augmented(self, index: int, image: Image):
        raise NotImplementedError()


class PNGStore(Store):
    def __init__(self, path: pathlib.Path, augment: bool):
        super().__init__(path, augment)
        self.path = path
        self.images_path = self.path.joinpath("images")
        self.augmented_images_path = self.path.joinpath("augmented_images")

    def initialize(self):
        """Creates a cirectory for the png files."""
        if self.path.exists():
            raise FileExistsError(f"Directory {self.path} already exists.")
        self.images_path.mkdir(parents=True, exist_ok=False)
        if self.augment:
            self.augmented_images_path.mkdir(parents=True, exist_ok=False)
        self.initialized = True

    def _store(self, index: int, image: Image):
        cv2.imwrite(str(self.images_path.joinpath(f"{index}.png")), image.data)

    def _store_augmented(self, index: int, image: Image):
        cv2.imwrite(str(self.augmented_images_path.joinpath(f"{index}.png")), image.data)


class HDFStore(Store):
    def __init__(self, path: pathlib.Path, augment: bool, dataset_shape: Tuple):
        super().__init__(path, augment)
        self.dataset_shape = dataset_shape
        if self.path.suffix != ".h5":
            self.path = pathlib.Path(f"{str(self.path)}.h5")

    def initialize(self):
        """Creates .h5 file for the dataset."""
        try:
            with h5py.File(self.path, "w-") as file:
                # Create a dataset in the file
                file.create_dataset("images", shape=self.dataset_shape, dtype=np.uint8)
                if self.augment:
                    file.create_dataset("augmented_images", shape=self.dataset_shape, dtype=np.uint8)
            self.initialized = True
        except OSError:
            raise FileExistsError(f"File {self.path} already exists.")

    def _store(self, index: int, image: Image):
        with h5py.File(self.path, "r+") as file:
            file["images"][index] = image.data

    def _store_augmented(self, index: int, image: Image):
        with h5py.File(self.path, "r+") as file:
            file["augmented_images"][index] = image.data


class LMDBStore(Store):
    def __init__(self, path: pathlib.Path, augment: bool, dataset_shape: Tuple):
        super().__init__(path, augment)
        self.dataset_shape = dataset_shape
        self.map_size = self.calculate_map_size()

    def calculate_map_size(self) -> int:
        # storing each image as unsigned 8 bit integer, so
        # size of an image = width * height * channels * 8
        # size of the whole dataset = number of images * image size
        map_size = np.prod(self.dataset_shape) * 8
        if self.augment:
            map_size *= 2
        return map_size

    def initialize(self):
        if self.path.exists():
            raise FileExistsError(f"Directory {self.path} already exists.")
        self.initialized = True

    def _store(self, index: int, image: Image):
        self._store_db(index, image, "images")

    def _store_augmented(self, index: int, image: Image):
        self._store_db(index, image, "augmented_images")

    def _store_db(self, index: int, image: Image, db_name: str):
        env = lmdb.open(str(self.path), map_size=self.map_size, max_dbs=2, readahead=False)
        db = env.open_db(key=f"{db_name}".encode("utf8"), create=True)
        with env.begin(db=db, write=True) as txn:
            txn.put(key=f"{index}".encode("utf8"), value=pickle.dumps(image.data))
        env.close()


class Processor:
    def __init__(self,
                 augment_value: int,
                 format: str,
                 grayscale: bool,
                 method: Callable[[Image, Tuple[int, int]], Image],
                 size: Tuple[int, int],
                 dst: pathlib.Path,
                 src: List[pathlib.Path]):
        self.supported_extensions: List[str] = ["jpg", "png", "dng"]
        self.augment_value = augment_value
        self.format = format
        self.grayscale = grayscale
        self.method = method
        self.size = size
        self.dst = dst
        self.src = src

    def process(self):
        print(f"Looking for images under: {chr(10)}"  # chr(10) = newline 
              f"{chr(10).join([str(path) for path in self.src])}")
        paths = self._get_image_paths()
        image_cnt = len(paths)
        print(f"Found {image_cnt} images.")

        store = None
        augment = True if 0 <= self.augment_value <= 255 else False
        if format == "png":
            store = PNGStore(self.dst, augment)
        elif format == "hdf":
            dataset_shape = (image_cnt, *self.size) if self.grayscale else (image_cnt, *self.size, 3)
            store = HDFStore(self.dst, augment, dataset_shape)
        elif format == "lmdb":
            dataset_shape = (image_cnt, *self.size) if self.grayscale else (image_cnt, *self.size, 3)
            store = LMDBStore(self.dst, augment, dataset_shape)

        print(f"Processing & saving images under:{chr(10)}{store.path}")
        with tqdm(total=image_cnt) as pbar:
            for i, path in enumerate(paths):
                image = Image(path)
                # process image
                image = self.method(image, self.size)
                if self.grayscale:
                    image.grayscale()
                if augment:
                    augmented_image = image.augment(self.augment_value)
                    store.store_augmented(i, augmented_image)
                store.store(i, image)
                pbar.update(1)
        print("Done.")

    def _get_image_paths(self) -> List[pathlib.Path]:
        """
        Returns a list of absolute image paths found recursively starting from the scripts directory.
        """
        image_paths = []
        for ext in self.supported_extensions:
            for s in self.src:
                image_paths.extend(list(s.rglob(f"*.{ext}")))
        return image_paths

    @staticmethod
    def clip(image: Image, size: Tuple[int, int]) -> Image:
        return Processor._clip(image, size, False)

    @staticmethod
    def clip_rnd(image: Image, size: Tuple[int, int]) -> Image:
        return Processor._clip(image, size, True)

    @staticmethod
    def _clip(image: Image, size: Tuple[int, int], random: bool) -> Image:
        # resize image if necessary
        if image.size[0] < size[0] or image.size[1] < size[1]:
            resize_factor = np.maximum(size[0] / image.size[0], size[1] / image.size[1])
            image.resize(tuple(np.ceil(image.size * resize_factor)))
        h_idx = 0
        w_idx = 0
        if random:
            h_min = 0
            h_max = image.size[0] - size[0]
            h_idx = 0 if h_min == h_max else np.random.randint(h_min, h_max)
            w_min = 0
            w_max = image.size[1] - size[1]
            w_idx = 0 if w_min == w_max else np.random.randint(w_min, w_max)
        # clip a portion out of the image
        image.data = image.data[h_idx:h_idx+size[0], w_idx:w_idx+size[1], :]
        return image

    @staticmethod
    def scale(image: Image, size: Tuple[int, int]) -> Image:
        return Processor._scale(image, size, False)

    @staticmethod
    def scale_rnd(image: Image, size: Tuple[int, int]) -> Image:
        return Processor._scale(image, size, True)

    @staticmethod
    def _scale(image: Image, size: Tuple[int, int], random: bool) -> Image:
        if image.size[0] < image.size[1]:
            h = image.size[0]
            w = np.ceil(image.size[0] / (size[0] / size[1])).astype(np.int16)
        else:
            h = np.ceil(image.size[1] * (size[0] / size[1])).astype(np.int16)
            w = image.size[1]
        if image.size[0] < h or image.size[1] < w:
            resize_factor = np.maximum(h / image.size[0], w / image.size[1])
            h = np.floor(h / resize_factor).astype(np.int16)
            w = np.floor(w / resize_factor).astype(np.int16)
        h_idx = 0
        w_idx = 0
        if random:
            h_min = 0
            h_max = image.size[0] - h
            h_idx = 0 if h_min == h_max else np.random.randint(h_min, h_max)
            w_min = 0
            w_max = image.size[1] - w
            w_idx = 0 if w_min == w_max else np.random.randint(w_min, w_max)
        image.data = image.data[h_idx:h_idx+h, w_idx:w_idx+w, :]
        image.resize(size)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--augment", dest="augment_value", action="store", default="-1", type=int,
                        help="Besides preprocessed images, store augmented ones as well. Augmented image is a processed"
                             " image with every 2nd pixel (in a checkerboard pattern) set to %(dest)s [0-255].")
    parser.add_argument("-f", "--format", dest="format", action="store", default="png", type=str,
                        choices=["png", "hdf", "lmdb"],
                        help="Output format to use. Supported: %(choices)s.  (default: %(default)s)")
    parser.add_argument("-g", "--grayscale", dest="grayscale", action="store_true",
                        help="Grayscale images.")
    parser.add_argument("-m", "--method", dest="method", action="store", default="scale", type=str,
                        choices=["clip", "clip_rnd", "scale", "scale_rnd"],
                        help="Processing method to use. (default: %(default)s)")
    parser.add_argument("-s", "--size", dest="size", action="store", default=[70, 70], nargs=2, type=int,
                        help="Output size of images [height, width]. (default: %(default)s)")
    parser.add_argument("dst", action="store", default="./dataset", type=str, nargs="?",
                        help="Name of the output file/directory. Can also be path e.g.: './results_png' or "
                             "'./a/b/hdf.h5'. (default: '%(default)s')")
    parser.add_argument("src", action="store", default=".", type=str, nargs="*",
                        help="Name(s) of the source directory(s) from where the search for images starts. Can also "
                             "be path(s). (default: '%(default)s')")

    args = parser.parse_args()

    format = args.format
    augment_value = args.augment_value
    grayscale = True if args.grayscale else False
    method = Processor.clip if args.method == "clip" else \
             Processor.clip_rnd if args.method == "clip_rnd" else \
             Processor.scale if args.method == "scale" else \
             Processor.scale_rnd if args.method == "scale_rnd" else \
             None
    size = tuple(args.size)
    dst = pathlib.Path(args.dst).absolute()
    src = [pathlib.Path(s).absolute() for s in args.src]

    p = Processor(augment_value, format, grayscale, method, size, dst, src)
    p.process()
