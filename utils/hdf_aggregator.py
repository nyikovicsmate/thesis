"""
Aggregates hdf dataset files. The shape of the images in each dataset must match.

usage: hdf_aggregator.py [-h] [dst] [src [src ...]]

positional arguments:
  dst         Name of the output file/directory. Can also be path e.g.:
              './a/b/hdf.h5'. (default: './aggregate.h5')
  src         List of hdf files to be aggregated.

optional arguments:
  -h, --help  show this help message and exit

"""
import argparse
import pathlib
from typing import List
import h5py
import numpy as np
from tqdm import tqdm


class Aggregator:
    def __init__(self, dst: pathlib.Path, src: List[pathlib.Path]):
        self.dst = dst
        if self.dst.suffix != ".h5":
            self.dst = pathlib.Path(f"{str(self.dst)}.h5")
        self.src = src

    def aggregate(self):
        print("Aggregating:")
        print(f"{chr(10).join([str(s) for s in self.src])}")
        print("into:")
        print(str(self.dst))

        augment = False
        dataset_shape = None
        for s in self.src:
            with h5py.File(str(s), "r") as file:
                shape = file["images"].shape
                if dataset_shape is not None and dataset_shape[1:] != shape[1:]:
                    raise OSError(f"Dataset mismatch, only datasets with same sized images can be aggregated.")
                dataset_shape = shape if dataset_shape is None else (dataset_shape[0] + shape[0], *shape[1:])
                try:
                    _ = len(file["augmented_images"])
                    augment = True
                except KeyError:
                    if augment:
                        raise OSError(f"Dataset mismatch, {s.name} doesn't contain augmented images, while the other "
                                      f"datasets do.")
        try:
            with tqdm(total=dataset_shape[0]) as pbar:
                with h5py.File(str(self.dst), "w-") as dst_file:
                    # Create a dataset in the file
                    dst_file.create_dataset("images", shape=dataset_shape, dtype=np.uint8)
                    if augment:
                        dst_file.create_dataset("augmented_images", shape=dataset_shape, dtype=np.uint8)

                    for s in self.src:
                        with h5py.File(str(s), "r") as src_file:
                            for idx in range(len(src_file["images"])):
                                dst_file["images"][pbar.n] = src_file["images"][idx]
                                if augment:
                                    dst_file["augmented_images"][pbar.n] = src_file["augmented_images"][idx]
                                pbar.update(1)
        except OSError:
            raise FileExistsError(f"File {self.dst} already exists.")
        print(f"Done aggregating {dataset_shape[0]} images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dst", action="store", default="./aggregate.h5", type=str, nargs="?",
                        help="Name of the output file/directory. Can also be path e.g.: "
                             "'./a/b/hdf.h5'. (default: '%(default)s')")
    parser.add_argument("src", action="store", default=".", type=str, nargs="*",
                        help="List of hdf files to be aggregated.")

    args = parser.parse_args()
    dst = pathlib.Path(args.dst).absolute()
    src = [pathlib.Path(s).absolute() for s in args.src]

    a = Aggregator(dst=dst, src=src)
    a.aggregate()
