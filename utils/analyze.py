import argparse
import os
import pathlib

import cv2
import numpy as np


def analyze(path: pathlib.Path) -> None:
    supported_image_formats = sorted(["png", "dng", "bmp", "jpg", "jpeg"])
    image_map = {}
    for image_format in supported_image_formats:
        print(f"Looking for '{image_format}' images under '{str(path)}'")
        image_map[image_format] = list(path.rglob(f"*.{image_format}"))
        print(f"Found {len(image_map[image_format])}")

    # min, max, average image dimensions
    dimensions = {}
    sizes = []
    for img_path_list in image_map.values():
        for img_path in img_path_list:
            img = cv2.imread(str(img_path.absolute()))
            h, w = img.shape[:2]
            dim = f"{w}x{h}"
            if dim in dimensions.keys():
                dimensions[dim] += 1
            else:
                dimensions[dim] = 1
            size_in_bytes = os.stat(str(img_path)).st_size
            sizes.append(size_in_bytes / 1024)
    print("DIMENSIONS")
    for k, v in dimensions.items():
        print(f"{k}: {v}")
    print()

    # min, max, average image size
    print("SIZES")
    print(f"Min: {np.min(sizes)} KB")
    print(f"Max: {np.max(sizes)} KB")
    print(f"Avg: {np.mean(sizes)} KB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", action="store", default=".", type=str, nargs="?",
                        help="Path of the source directory from where the search for images starts. "
                             "(default: '%(default)s')")

    args = parser.parse_args()
    path = pathlib.Path(args.path)
    analyze(path)