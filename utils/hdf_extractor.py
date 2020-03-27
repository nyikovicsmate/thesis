"""
Extract png images out of a hdf dataset for convenience.

usage: hdf_extractor.py [-h] src start_index [end_index]

positional arguments:
  src          Name of the source dataset file.
  start_index  Lower image range index to extract from. (default: 0)
  end_index    Upper image range index to extract to. (default: 0)

optional arguments:
  -h, --help   show this help message and exit

"""
import cv2
import numpy as np
import argparse
import pathlib
import h5py


def extract(src: pathlib.Path, start_index: int, end_index: int):
    with h5py.File(str(src), "r") as file:
        images = np.array(file["images"][start_index:end_index], dtype=np.uint8)
    cwd = pathlib.Path(__file__).parent
    for idx, image in enumerate(images):
        dst = cwd.joinpath(f"{src.stem}_{start_index + idx}.png")
        cv2.imwrite(str(dst), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src", action="store", type=str, nargs=1,
                        help="Name of the source dataset file.")
    parser.add_argument("start_index", action="store", default=0, type=int, nargs=1,
                        help="Lower image range index to extract from. (default: %(default)s)")
    parser.add_argument("end_index", action="store", default=0, type=int, nargs="?",
                        help="Upper image range index to extract to. (default: %(default)s)")
    args = parser.parse_args()

    src = pathlib.Path(args.src[0]).absolute()
    start_index = args.start_index[0]
    end_index = start_index + 1 if start_index >= args.end_index else args.end_index

    extract(src, start_index, end_index)
