from src.networks.reinforced.actions import *


class State:

    @staticmethod
    def update(img_batch: np.ndarray,
               action_batch: np.ndarray) -> np.ndarray:
        """
        :param img_batch: the bach of input images in NHWC format
        :param action_batch: action img.shape-d mx with values 0-(N_ACTIONS-1)
        :return: the modified image batch
        """
        assert img_batch.shape == action_batch.shape
        img_batch_new = np.zeros_like(img_batch)
        for n in range(img_batch.shape[0]):
            img = img_batch[n]
            actions = action_batch[n]
            img_alternatives = {
                0: AlterByValue(img, -1.0 / 255.0),
                1: AlterByValue(img, 0.0),
                2: AlterByValue(img, 1.0 / 255.0),
                3: GaussianBlur(img, ksize=(5, 5), sigmaX=0.5),
                4: BilaterFilter(img, d=5, sigma_color=0.1, sigma_space=5),
                5: MedianBlur(img, ksize=5),
                6: GaussianBlur(img, ksize=(5, 5), sigmaX=1.5),
                7: BilaterFilter(img, d=5, sigma_color=1.0, sigma_space=5),
                8: BoxFilter(img, ddepth=-1, ksize=(5, 5))
            }
            for k, v in img_alternatives.items():
                img_batch_new[n] = np.where(actions == k, v.processed_img, img_batch_new[n])

        return img_batch_new
