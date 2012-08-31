import unittest
import pysal
from pysal.spatial_dynamics import util
import numpy as np


class ShuffleMatrix_Tester(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(16)
        self.X.shape = (4, 4)

    def test_shuffle_matrix(self):
        np.random.seed(10)
        obs = util.shuffle_matrix(self.X, range(4)).flatten().tolist()
        exp = [10, 8, 11, 9, 2, 0, 3, 1, 14, 12, 15, 13, 6, 4, 7, 5]
        for i in range(16):
            self.assertEqual(exp[i], obs[i])


class GetLower_Tester(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(16)
        self.X.shape = (4, 4)

    def test_get_lower(self):
        np.random.seed(10)
        obs = util.get_lower(self.X).flatten().tolist()
        exp = [4, 8, 9, 12, 13, 14]
        for i in range(6):
            self.assertEqual(exp[i], obs[i])


suite = unittest.TestSuite()
test_classes = [ShuffleMatrix_Tester, GetLower_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
