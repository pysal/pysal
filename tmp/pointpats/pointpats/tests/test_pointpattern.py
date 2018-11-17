import unittest
import numpy as np

from ..pointpattern import PointPattern
from libpysal.common import RTOL


class TestPointPattern(unittest.TestCase):

    def setUp(self):
        self.points = [[66.22, 32.54], [22.52, 22.39], [31.01, 81.21],
                       [9.47, 31.02], [30.78, 60.10], [75.21, 58.93],
                       [79.26,  7.68], [8.23, 39.93], [98.73, 77.17],
                       [89.78, 42.53], [65.19, 92.08], [54.46, 8.48]]
        self.pp = PointPattern(self.points)
        self.assertEqual(len(self.pp), 12)
        self.assertTrue([66.22, 32.54] in self.pp)

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

    def test_point_pattern_knn(self):
        knn = self.pp.knn(1)
        nn = np.array([[9], [3], [4], [7], [2], [9], [11], [3], [5], [5], [5],
                      [6]])
        nnd = np.array([[25.59050019], [15.64542745], [21.11125292],
                        [8.99587128], [21.11125292], [21.93729473],
                        [24.81289987], [8.99587128], [29.76387072],
                        [21.93729473], [34.63124168], [24.81289987]])
        np.testing.assert_array_equal(knn[0], nn)
        np.testing.assert_array_almost_equal(knn[1], nnd)

    def test_point_pattern_knn_error(self):
        self.assertRaises(ValueError, self.pp.knn, k=0)

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

    def test_point_pattern_knn_other_error(self):
        knn_other = self.pp.knn_other
        self.assertRaises(ValueError, knn_other, self.pp, k=0)

    def test_point_pattern_explode(self):
        explosion = self.pp.explode('x')
        for ppattern in explosion:
            np.testing.assert_array_equal(ppattern.df.iloc[0],
                                          self.pp.df.loc[ppattern.df.index[0]])

    def test_point_pattern_flip_coordinates(self):
        pp_flip = PointPattern(self.points, coord_names=['The x coordinate',
                                                        'The y coordinate'])
        coord = pp_flip.coord_names
        x_coord, y_coord = pp_flip._x, pp_flip._y
        # Flip the coordinates
        pp_flip.flip_coordinates()

        coord_flipped = pp_flip.coord_names
        x_coord_flipped, y_coord_flipped = pp_flip._x, pp_flip._y
        self.assertEqual(x_coord, y_coord_flipped)
        self.assertEqual(y_coord, x_coord_flipped)
        self.assertEqual(coord, coord_flipped)

        # Flip the coordinates again, they should return to the intial values.
        pp_flip.flip_coordinates()

        coord_again = pp_flip.coord_names
        x_coord_again, y_coord_again = pp_flip._x, pp_flip._y
        self.assertEqual(x_coord, x_coord_again)
        self.assertEqual(y_coord, y_coord_again)
        self.assertEqual(coord, coord_again)
