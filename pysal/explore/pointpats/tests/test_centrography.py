# TODO: skyum, dtot, weighted_mean_center, manhattan_median
import unittest
import numpy as np

from ..centrography import *

from pysal.lib.common import RTOL


class TestCentrography(unittest.TestCase):

    def setUp(self):
        self.points = [[66.22, 32.54], [22.52, 22.39], [31.01, 81.21],
                       [9.47, 31.02], [30.78, 60.10], [75.21, 58.93],
                       [79.26,  7.68], [8.23, 39.93], [98.73, 77.17],
                       [89.78, 42.53], [65.19, 92.08], [54.46, 8.48]]

    def test_centrography_mbr(self):
        min_x, min_y, max_x, max_y = mbr(self.points)
        np.testing.assert_allclose(min_x, 8.2300000000000004, RTOL)
        np.testing.assert_allclose(min_y, 7.6799999999999997, RTOL)
        np.testing.assert_allclose(max_x, 98.730000000000004, RTOL)
        np.testing.assert_allclose(max_y, 92.079999999999998, RTOL)

    def test_centrography_hull(self):
        hull_array = hull(self.points)
        res = np.array([[31.01, 81.21], [8.23, 39.93], [9.47, 31.02],
                        [22.52, 22.39], [54.46, 8.48], [79.26, 7.68],
                        [89.78, 42.53], [98.73, 77.17], [65.19, 92.08]])
        np.testing.assert_array_equal(hull_array, res)

    def test_centrography_mean_center(self):
        res = np.array([52.57166667, 46.17166667])
        np.testing.assert_array_almost_equal(mean_center(self.points), res)

    def test_centrography_std_distance(self):
        std = std_distance(self.points)
        np.testing.assert_allclose(std, 40.149806489086714, RTOL)

    def test_centrography_ellipse(self):
        res = ellipse(self.points)
        np.testing.assert_allclose(res[0], 39.623867886462982, RTOL)
        np.testing.assert_allclose(res[1], 42.753818949026815, RTOL)
        np.testing.assert_allclose(res[2], 1.1039268428650906, RTOL)

    def test_centrography_euclidean_median(self):
        euclidean = euclidean_median(self.points)
        res = np.array([54.16770671, 44.4242589])
        np.testing.assert_array_almost_equal(euclidean, res)
