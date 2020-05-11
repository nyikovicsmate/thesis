import math
from abc import abstractmethod
from typing import Union, Tuple
import numpy as np
import cv2


class Kernel:
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 border_type: int):
        """
        # TODO
        :param size:
        :param border_type: border type [reflect101 (default opencv bordr), constant]
        """
        size = (size, size) if type(size) is int else size
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"Kernel size must be positive, got {size}.")
        if size[0] % 2 != 1 or size[1] % 2 != 1:
            raise ValueError(f"Kernel size must be odd, got {size}.")
        self.size = size
        self.border_type = border_type

    @abstractmethod
    def apply_to_region(self,
              img,
              i,
              j):
        """
        # TODO
        :param img:
        :param i:
        :param j:
        :return:
        """
        raise NotImplementedError()

    def get_img_region(self,
                   img: np.ndarray,
                   i: int,
                   j: int) -> np.ndarray:
        """

        :param img:
        :param i:
        :param j:
        :return:
        """
        region = np.zeros(shape=self.size, dtype=np.float32)
        i_offset = self.size[1] // 2
        j_offset = self.size[0] // 2
        for i_region, i_img in enumerate(range(i - i_offset, i + i_offset + 1)):
            for j_region, j_img in enumerate(range(j - j_offset, j + j_offset + 1)):
                # calculate the interpolated indexes given certain border type
                i_img = cv2.borderInterpolate(i_img, img.shape[0], self.border_type)
                j_img = cv2.borderInterpolate(j_img, img.shape[1], self.border_type)
                region[i_region, j_region] = img[i_img, j_img]
        return region


class GaussianKernel(Kernel):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 sigma: float,
                 border_type: int = cv2.BORDER_REFLECT_101):
        """
        # TODO
        :param size:
        :param sigma:
        :param border_type:
        """
        super().__init__(size, border_type)
        self.sigma = sigma
        self.gaussian_kernel = cv2.getGaussianKernel(ksize=self.size[0], sigma=self.sigma) * \
                               cv2.getGaussianKernel(ksize=self.size[1], sigma=self.sigma).transpose()

    def apply_to_region(self, img, i, j):
        return (self.get_img_region(img, i, j) * self.gaussian_kernel).sum()


class BilateralKernel(Kernel):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 d: int,
                 sigma_color: float,
                 sigma_space: float,
                 border_type: int = cv2.BORDER_REFLECT_101):
        super().__init__(size, border_type)

        if sigma_color <= 0:
            sigma_color = 1
        if sigma_space <= 0:
            sigma_space = 1
        if d <= 0:
            self.radius = np.rint(sigma_space*1.5)
        else:
            self.radius = d // 2.0
        self.radius = np.maximum(self.radius, 1.0)
        d = self.radius * 2 + 1
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.gauss_color_coeff = -0.5 / (sigma_color ** 2)
        self.gauss_space_coeff = -0.5 / (sigma_space ** 2)
        self.distance_mx = self._get_distance_mx()

    def _get_distance_mx(self) -> np.ndarray:
        mx = np.zeros(shape=self.size, dtype=np.float32)
        center_idx = (self.size[0] // 2, self.size[1] // 2)
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                mx[y, x] = np.sqrt((center_idx[0]-y)**2 + (center_idx[1]-x)**2)
        return mx

    def apply_to_region(self, img, i, j):
        region = self.get_img_region(img, i, j)
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
                gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
                w = gi * gs
                i_filtered += source[neighbour_x][neighbour_y] * w
                Wp += w
        i_filtered = i_filtered / Wp
        filtered_image[x][y] = int(round(i_filtered))




# https://github.com/anlcnydn/bilateral/blob/master/bilateral_filter.py
def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(round(i_filtered))

