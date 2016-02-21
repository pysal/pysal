import unittest
import numpy as np

from pysal.contrib.points.pointpattern import PointPattern
from pysal.common import RTOL


class TestPointPattern(unittest.TestCase):

    def setUp(self):
        points = [[66.22, 32.54], [22.52, 22.39], [31.01, 81.21],
                  [9.47, 31.02], [30.78, 60.10], [75.21, 58.93],
                  [79.26,  7.68], [8.23, 39.93], [98.73, 77.17],
                  [89.78, 42.53], [65.19, 92.08], [54.46, 8.48]]
        self.pp = PointPattern(points)

    def test_point_pattern_n(self):
        self.assertEqual(self.pp.n, 12)

    def test_point_pattern_mean_nnd(self):
        np.testing.assert_allclose(self.pp.mean_nnd, 21.612139802089246, RTOL)

    def test_point_pattern_lambda_mbb(self):
        np.testing.assert_allclose(self.pp.lambda_mbb,
                                   0.0015710507711240867, RTOL)

    def test_point_pattern_lambda_hull(self):
        np.testing.assert_allclose(self.pp.lambda_hull,
                                   0.0022667153468973137, RTOL)

    def test_point_pattern_hull_area(self):
        np.testing.assert_allclose(self.pp.hull_area, 5294.0039500000003, RTOL)

    def test_point_pattern_mbb_area(self):
        np.testing.assert_allclose(self.pp.mbb_area, 7638.2000000000007, RTOL)

    def test_point_pattern_min_nnd(self):
        np.testing.assert_allclose(self.pp.min_nnd, 8.9958712752017522, RTOL)

    def test_point_pattern_max_nnd(self):
        np.testing.assert_allclose(self.pp.max_nnd, 34.63124167568931, RTOL)

    def test_point_pattern_find_pairs(self):
        self.assertEqual(self.pp.find_pairs(10), {(3, 7)})
        self.assertEqual(self.pp.find_pairs(20), {(3, 7), (1, 3)})

    def test_point_pattern_knn_other(self):
        knn = self.pp.knn_other(self.pp)
        nn = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        nnd = np.zeros(12)
        np.testing.assert_array_equal(knn[0], nn)
        np.testing.assert_array_equal(knn[1], nnd)

        knn = self.pp.knn_other([0, 0], k=12)
        nn = np.array([1, 3, 7, 11, 4, 0, 6, 2, 5, 9, 10, 8])
        nnd = np.array([31.75629859, 32.43333625, 40.76932425, 55.11625894,
                       67.52346555, 73.78306039, 79.63121247, 86.92919072,
                       95.54731289, 99.34409545, 112.82048794, 125.31090056])
        np.testing.assert_array_equal(knn[0], nn)
        np.testing.assert_array_almost_equal(knn[1], nnd)
