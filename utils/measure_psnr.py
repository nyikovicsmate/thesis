import argparse

import cv2
import numpy as np
# import tensorflow as tf


def psnr(a, b, max_val=255.0):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return np.inf
    return 10 * np.log10(max_val ** 2 / mse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_val", dest="max_val", action="store", default=255.0, nargs=1, type=float,
                        help="The dynamic range of the images (i.e., the difference between the maximum "
                             "the and minimum allowed values). (default: %(default)s)")
    parser.add_argument("images", action="store", type=str, nargs=2,
                        help="The first and second image to compare.")

    args = parser.parse_args()
    max_val = args.max_val
    images = args.images
    assert len(images) == 2, "Missing image."

    a = cv2.imread(images[0])
    a = np.array(a, dtype=np.float32)
    # a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    b = cv2.imread(images[1])
    b = np.array(b, dtype=np.float32)
    # b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

    # psnr_tf = tf.image.psnr(a, b, max_val=max_val).numpy()
    # print(f"{psnr_tf:.4f}")
    # psnr_cv2 = cv2.PSNR(a, b, max_val)
    # print(f"{psnr_cv2:.4f}")
    psnr = psnr(a, b, max_val)
    print(f"{psnr:.4f}")
