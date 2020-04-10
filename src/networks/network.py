from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import tensorflow as tf

from src.dataset import Dataset


class Network(ABC):

    def __init__(self,
                 model: tf.keras.models.Model):
        self.model = model

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

    @abstractmethod
    def train(self, dataset_x: Dataset, dataset_y: Dataset, loss_func: Callable[[np.ndarray, np.ndarray], float], epochs: int, learning_rate: float):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: The batch of LR input images.
        :return:
        """
        return self.model(x).numpy()

    @staticmethod
    def evaluate(y: np.ndarray, y_pred: np.ndarray):
        """
        :param y: The batch of HR images.
        :param y_pred: The corresponding batch of predicted images.
        :return:
        """
        _len = len(y_pred)
        assert _len == len(y)
        ssim_list = []
        total_variation_list = []
        psnr_list = []
        mean_squared_error_list = []
        mean_absolute_error_list = []
        for i in range(_len):
            y_i_tensor = tf.convert_to_tensor(y[i], dtype=tf.float32)
            y_pred_i_tensor = tf.convert_to_tensor(y_pred[i], dtype=tf.float32)
            ssim_list.append(tf.image.ssim(y_i_tensor, y_pred_i_tensor, max_val=1).numpy())
            total_variation_list.append(tf.image.total_variation(y_pred_i_tensor).numpy())
            psnr_list.append(tf.image.psnr(y_i_tensor, y_pred_i_tensor, max_val=1).numpy())
            mean_squared_error_list.append(np.sum(tf.losses.mean_squared_error(y_i_tensor, y_pred_i_tensor).numpy()))
            mean_absolute_error_list.append(np.sum(tf.losses.mean_absolute_error(y_i_tensor, y_pred_i_tensor).numpy()))

        ssim_list = np.array(ssim_list)
        total_variation_list = np.array(total_variation_list)
        psnr_list = np.array(psnr_list)
        mean_squared_error_list = np.array(mean_squared_error_list)
        mean_absolute_error_list = np.array(mean_absolute_error_list)

        per_image_results = [
            {"ssim": ssim, "total_variation": tv, "psnr": psnr, "mean_squared_error": mse, "mean_absolute_error": mae}
            for ssim, tv, psnr, mse, mae
            in zip(ssim_list, total_variation_list, psnr_list, mean_squared_error_list, mean_absolute_error_list)]

        max = {"ssim": np.amax(ssim_list),
               "total_variation": np.amax(total_variation_list),
               "psnr": np.amax(psnr_list),
               "mean_squared_error": np.amax(mean_squared_error_list),
               "mean_absolute_error": np.amax(mean_absolute_error_list)}

        min = {"ssim": np.amin(ssim_list),
               "total_variation": np.amin(total_variation_list),
               "psnr": np.amin(psnr_list),
               "mean_squared_error": np.amin(mean_squared_error_list),
               "mean_absolute_error": np.amin(mean_absolute_error_list)}

        avg = {"ssim": np.average(ssim_list),
               "total_variation": np.average(total_variation_list),
               "psnr": np.average(psnr_list),
               "mean_squared_error": np.average(mean_squared_error_list),
               "mean_absolute_error": np.average(mean_absolute_error_list)}

        print("SSIM")
        print(f"max: [{np.where(ssim_list == max['ssim'])[0][0]}] {max['ssim']:.2f}    "
              f"min: [{np.where(ssim_list == min['ssim'])[0][0]}] {min['ssim']:.2f}    "
              f"avg: {avg['ssim']:.2f}")
        print("TOTAL_VARIATION")
        print(f"max: [{np.where(total_variation_list == max['total_variation'])[0][0]}] {max['total_variation']:.2f}    "
              f"min: [{np.where(total_variation_list == min['total_variation'])[0][0]}] {min['total_variation']:.2f}    "
              f"avg: {avg['total_variation']:.2f}")
        print("PSNR")
        print(f"max: [{np.where(psnr_list == max['psnr'])[0][0]}] {max['psnr']:.2f}    "
              f"min: [{np.where(psnr_list == min['psnr'])[0][0]}] {min['psnr']:.2f}    "
              f"avg: {avg['psnr']:.2f}")
        print("MEAN_SQUARED_ERROR")
        print(f"max: [{np.where(mean_squared_error_list == max['mean_squared_error'])[0][0]}] {max['mean_squared_error']:.2f}    "
              f"min: [{np.where(mean_squared_error_list == min['mean_squared_error'])[0][0]}] {min['mean_squared_error']:.2f}    "
              f"avg: {avg['mean_squared_error']:.2f}")
        print("MEAN_ABSOLUTE_ERROR")
        print(f"max: [{np.where(mean_absolute_error_list == max['mean_absolute_error'])[0][0]}] {max['mean_absolute_error']:.2f}    "
              f"min: [{np.where(mean_absolute_error_list == min['mean_absolute_error'])[0][0]}] {min['mean_absolute_error']:.2f}    "
              f"avg: {avg['mean_absolute_error']:.2f}")

        return per_image_results


