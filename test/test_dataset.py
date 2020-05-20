import unittest
from itertools import zip_longest

from src.dataset import *
import numpy as np


class TestHDFDataset(unittest.TestCase):


    def test_reshuffle(self):
        ds = HDFDataset("bsd500_35_35_gray.h5").shuffle(reshuffle_each_iteration=True)
        l = []
        with ds:
            for _ in range(2):
                l.append(list(ds)[0])
        self.assertEqual(len(l[0]), 500)
        self.assertEqual(len(l[1]), 500)
        self.assertFalse(np.array_equal(l[0], l[1]))

    def test_shuffle_equality_while(self):
        seed = 1111
        ds_1 = HDFDataset("bsd500_35_35_gray.h5").shuffle(seed)
        ds_2 = HDFDataset("bsd500_35_35_gray.h5").shuffle(seed)

        with ds_1, ds_2:
            ds_1_iter = iter(ds_1)
            ds_2_iter = iter(ds_2)

            equality = []
            i = 0
            ii = 0
            while i < 3:
                try:
                    x_1 = next(ds_1_iter)
                    x_2 = next(ds_2_iter)
                    # print(f"{self._testMethodName}: iteration: {ii}.")
                    ii += 1
                except StopIteration:
                    # print(f"{self._testMethodName}: Dataset end reached.")
                    ds_1_iter = iter(ds_1)
                    ds_2_iter = iter(ds_2)
                    equality.append(np.array_equal(x_1, x_2))
                    i += 1
                    ii = 0
        self.assertTrue(all(equality))

    def test_batch_shuffle_equality_while(self):
        seed = 1111
        ds_1 = HDFDataset("bsd500_35_35_gray.h5").batch(100).shuffle(seed)
        ds_2 = HDFDataset("bsd500_35_35_gray.h5").batch(100).shuffle(seed)

        with ds_1, ds_2:
            ds_1_iter = iter(ds_1)
            ds_2_iter = iter(ds_2)

            equality = []
            i = 0
            ii = 0
            while i < 3:
                try:
                    x_1 = next(ds_1_iter)
                    x_2 = next(ds_2_iter)
                    # print(f"{self._testMethodName}: iteration: {ii}.")
                    ii += 1
                except StopIteration:
                    # print(f"{self._testMethodName}: Dataset end reached.")
                    ds_1_iter = iter(ds_1)
                    ds_2_iter = iter(ds_2)
                    equality.append(np.array_equal(x_1, x_2))
                    i += 1
                    ii = 0
        self.assertTrue(all(equality))
        
    def test_shuffle_equality_for(self):
        seed = 2
        ds_1 = HDFDataset("bsd500_35_35_gray.h5").shuffle(seed)
        ds_2 = HDFDataset("bsd500_35_35_gray.h5").shuffle(seed)

        iteratoins = 3
        with ds_1, ds_2:
            for i in range(iteratoins):
                # print(f"{self._testMethodName}: iteration: {i}.")
                for x, y in zip(ds_1, ds_2):
                    self.assertTrue(np.array_equal(x,y), f"{self._testMethodName}: Datasets are out of sync.")
                    # print(f"{self._testMethodName}: Datasets are in sync.")

    def test_shuffle_equality_for_batch(self):
        seed = 2
        ds_1 = HDFDataset("bsd500_35_35_gray.h5").batch(100).shuffle(seed)
        ds_2 = HDFDataset("bsd500_35_35_gray.h5").batch(100).shuffle(seed)

        iteratoins = 3
        with ds_1, ds_2:
            for i in range(iteratoins):
                # print(f"{self._testMethodName}: iteration: {i}.")
                for x, y in zip(ds_1, ds_2):
                    self.assertTrue(np.array_equal(x,y), f"{self._testMethodName}: Datasets are out of sync.")
                    # print(f"{self._testMethodName}: Datasets are in sync.")


    def test_shuffle_inequality(self):
        seed = 2
        ds_1 = HDFDataset("bsd500_35_35_gray.h5").batch(10).shuffle(seed)
        ds_2 = HDFDataset("bsd500_35_35_gray.h5").batch(10).shuffle(seed * 2)  # out of sync dataset

        iteratoins = 3
        with ds_1, ds_2:
            for i in range(iteratoins):
                # print(f"{self._testMethodName}: iteration: {i}.")
                for x, y in zip(ds_1, ds_2):
                    self.assertFalse(np.array_equal(x, y), f"{self._testMethodName}: Datasets are in sync.")
                    # print(f"{self._testMethodName}: Datasets are out of sync.")

    def test_sperate_dataset_order(self):
        """Tests that the images in different datasets are in teh same order."""
        ds_1 = HDFDataset("bsd500_35_35_color.h5")
        ds_2 = HDFDataset("bsd500_70_70_color.h5")
        equal = []
        with ds_1, ds_2:
            for lr, hr in zip(next(ds_1), next(ds_2)):
                hr_2 = cv2.resize(lr, (70, 70), interpolation=cv2.INTER_CUBIC)  # upscale the low-res images
                eq = np.allclose(a=hr_2, b=hr, atol=5)  # compare them with tolerance ~2%
                equal.append(eq)

        self.assertTrue(all(equal))  # assert that all the equality checks resulted True

    def test_sperate_dataset_order_shuffled(self):
        """Tests that paired images from different datasets keep their correct pairs while shuffled."""
        seed = 2
        ds_1 = HDFDataset("bsd500_35_35_color.h5").batch(1).shuffle(seed)
        ds_2 = HDFDataset("bsd500_70_70_color.h5").batch(1).shuffle(seed)
        equal = []
        with ds_1, ds_2:
            for lr, hr in zip(next(ds_1), next(ds_2)):
                hr_2 = cv2.resize(lr, (70, 70), interpolation=cv2.INTER_CUBIC)  # upscale the low-res images
                eq = np.allclose(a=hr_2, b=hr, atol=5)  # compare them with tolerance ~2%
                equal.append(eq)

        self.assertTrue(all(equal))  # assert that all the equality checks resulted True

class TestDirectoryDataset(unittest.TestCase):

    def test_reshuffle(self):
        ds = DirectoryDataset("set14_35_35_color").shuffle(reshuffle_each_iteration=True)
        l = []
        with ds:
            for _ in range(2):
                l.append(list(ds)[0])
        self.assertEqual(len(l[0]), 14)
        self.assertEqual(len(l[1]), 14)
        self.assertFalse(np.array_equal(l[0], l[1]))

    def test_shuffle_equality_while(self):
        seed = 1111
        ds_1 = DirectoryDataset("set14_35_35_color").shuffle(seed)
        ds_2 = DirectoryDataset("set14_35_35_color").shuffle(seed)

        with ds_1, ds_2:
            ds_1_iter = iter(ds_1)
            ds_2_iter = iter(ds_2)

            equality = []
            i = 0
            ii = 0
            while i < 3:
                try:
                    x_1 = next(ds_1_iter)
                    x_2 = next(ds_2_iter)
                    # print(f"{self._testMethodName}: iteration: {ii}.")
                    ii += 1
                except StopIteration:
                    # print(f"{self._testMethodName}: Dataset end reached.")
                    ds_1_iter = iter(ds_1)
                    ds_2_iter = iter(ds_2)
                    equality.append(np.array_equal(x_1, x_2))
                    i += 1
                    ii = 0
        self.assertTrue(all(equality))

    def test_batch_shuffle_equality_while(self):
        seed = 1111
        ds_1 = DirectoryDataset("set14_35_35_color").batch(100).shuffle(seed)
        ds_2 = DirectoryDataset("set14_35_35_color").batch(100).shuffle(seed)

        with ds_1, ds_2:
            ds_1_iter = iter(ds_1)
            ds_2_iter = iter(ds_2)

            equality = []
            i = 0
            ii = 0
            while i < 3:
                try:
                    x_1 = next(ds_1_iter)
                    x_2 = next(ds_2_iter)
                    # print(f"{self._testMethodName}: iteration: {ii}.")
                    ii += 1
                except StopIteration:
                    # print(f"{self._testMethodName}: Dataset end reached.")
                    ds_1_iter = iter(ds_1)
                    ds_2_iter = iter(ds_2)
                    equality.append(np.array_equal(x_1, x_2))
                    i += 1
                    ii = 0
        self.assertTrue(all(equality))

    def test_shuffle_equality_for(self):
        seed = 2
        ds_1 = DirectoryDataset("set14_35_35_color").shuffle(seed)
        ds_2 = DirectoryDataset("set14_35_35_color").shuffle(seed)

        iteratoins = 3
        with ds_1, ds_2:
            for i in range(iteratoins):
                # print(f"{self._testMethodName}: iteration: {i}.")
                for x, y in zip(ds_1, ds_2):
                    self.assertTrue(np.array_equal(x, y), f"{self._testMethodName}: Datasets are out of sync.")
                    # print(f"{self._testMethodName}: Datasets are in sync.")

    def test_shuffle_equality_for_batch(self):
        seed = 2
        ds_1 = DirectoryDataset("set14_35_35_color").batch(100).shuffle(seed)
        ds_2 = DirectoryDataset("set14_35_35_color").batch(100).shuffle(seed)

        iteratoins = 3
        with ds_1, ds_2:
            for i in range(iteratoins):
                # print(f"{self._testMethodName}: iteration: {i}.")
                for x, y in zip(ds_1, ds_2):
                    self.assertTrue(np.array_equal(x, y), f"{self._testMethodName}: Datasets are out of sync.")
                    # print(f"{self._testMethodName}: Datasets are in sync.")

    def test_shuffle_inequality(self):
        seed = 2
        ds_1 = DirectoryDataset("set14_35_35_color").batch(10).shuffle(seed)
        ds_2 = DirectoryDataset("set14_35_35_color").batch(10).shuffle(seed * 2)  # out of sync dataset

        iteratoins = 3
        with ds_1, ds_2:
            for i in range(iteratoins):
                # print(f"{self._testMethodName}: iteration: {i}.")
                for x, y in zip(ds_1, ds_2):
                    self.assertFalse(np.array_equal(x, y), f"{self._testMethodName}: Datasets are in sync.")
                    # print(f"{self._testMethodName}: Datasets are out of sync.")

    def test_sperate_dataset_order(self):
        """Tests that the images in different datasets are in teh same order."""
        ds_1 = DirectoryDataset("set14_35_35_color")
        ds_2 = DirectoryDataset("set14_70_70_color")
        equal = []
        with ds_1, ds_2:
            for lr, hr in zip(next(ds_1), next(ds_2)):
                hr_2 = cv2.resize(lr, (70, 70), interpolation=cv2.INTER_CUBIC)  # upscale the low-res images
                eq = np.allclose(a=hr_2, b=hr, atol=5)  # compare them with tolerance ~2%
                equal.append(eq)

        self.assertTrue(all(equal))  # assert that all the equality checks resulted True

    def test_sperate_dataset_order_shuffled(self):
        """Tests that paired images from different datasets keep their correct pairs while shuffled."""
        seed = 2
        ds_1 = DirectoryDataset("set14_35_35_color").batch(1).shuffle(seed)
        ds_2 = DirectoryDataset("set14_70_70_color").batch(1).shuffle(seed)
        equal = []
        with ds_1, ds_2:
            for lr, hr in zip(next(ds_1), next(ds_2)):
                hr_2 = cv2.resize(lr, (70, 70), interpolation=cv2.INTER_CUBIC)  # upscale the low-res images
                eq = np.allclose(a=hr_2, b=hr, atol=5)  # compare them with tolerance ~2%
                equal.append(eq)

        self.assertTrue(all(equal))  # assert that all the equality checks resulted True


if __name__ == '__main__':
    unittest.main()
