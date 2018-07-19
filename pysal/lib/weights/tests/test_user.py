import os
import unittest
import numpy as np
from ..util import neighbor_equality
from ..weights import W
from .. import user
from ... import examples as pysal_examples


class Testuser(unittest.TestCase):
    def setUp(self):
        self.wq = user.queen_from_shapefile(
            pysal_examples.get_path("columbus.shp"))
        self.wr = user.rook_from_shapefile(
            pysal_examples.get_path("columbus.shp"))

    def test_queen_from_shapefile(self):
        self.assertAlmostEqual(self.wq.pct_nonzero, 9.82923781757601)

    def test_rook_from_shapefile(self):
        self.assertAlmostEqual(self.wr.pct_nonzero, 8.329862557267806)

    def test_knnW_from_array(self):
        import numpy as np
        x, y = np.indices((5, 5))
        x.shape = (25, 1)
        y.shape = (25, 1)
        data = np.hstack([x, y])
        wnn2 = user.knnW_from_array(data, k=2)
        wnn4 = user.knnW_from_array(data, k=4)
        self.assertEqual(set(wnn4.neighbors[0]), set([1, 5, 6, 2]))
        self.assertEqual(set(wnn4.neighbors[5]), set([0, 6, 10, 1]))
        self.assertEqual(set(wnn2.neighbors[0]), set([1, 5]))
        self.assertEqual(set(wnn2.neighbors[5]), set([0, 6]))
        self.assertAlmostEqual(wnn2.pct_nonzero, 8.0)
        self.assertAlmostEqual(wnn4.pct_nonzero, 16.0)
        wnn4 = user.knnW_from_array(data, k=4)
        self.assertEqual(set(wnn4.neighbors[0]), set([1, 5, 6, 2]))
        '''
        wnn3e = pysal.knnW(data, p=2, k=3)
        self.assertEquals(set(wnn3e.neighbors[0]),set([1, 5, 6]))
        wnn3m = pysal.knnW(data, p=1, k=3)
        self.assertEquals(set(wnn3m.neighbors[0]), set([1, 5, 2]))
        '''

    def test_knnW_from_shapefile(self):
        wc = user.knnW_from_shapefile(pysal_examples.get_path("columbus.shp"))
        self.assertAlmostEqual(wc.pct_nonzero, 4.081632653061225)
        wc3 = user.knnW_from_shapefile(pysal_examples.get_path(
            "columbus.shp"), k=3)
        self.assertEqual(wc3.weights[1], [1, 1, 1])
        self.assertEqual(set(wc3.neighbors[1]), set([3, 0, 7]))
        self.assertEqual(set(wc.neighbors[0]), set([2, 1]))
        w = user.knnW_from_shapefile(pysal_examples.get_path('juvenile.shp'))
        self.assertAlmostEqual(w.pct_nonzero, 1.1904761904761904)
        w1 = user.knnW_from_shapefile(
            pysal_examples.get_path('juvenile.shp'), k=1)
        self.assertAlmostEqual(w1.pct_nonzero, 0.5952380952380952)

    def test_threshold_binaryW_from_array(self):
        points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        w = user.threshold_binaryW_from_array(points, threshold=11.2)
        self.assertEqual(w.weights, {0: [1, 1], 1: [1, 1], 2: [],
                                      3: [1, 1], 4: [1], 5: [1]})
        self.assertTrue(neighbor_equality(w, W({0: [1, 3], 1: [0, 3],
                                                        2: [ ], 3: [0, 1],
                                                        4: [5], 5: [4]})))

    def test_threshold_binaryW_from_shapefile(self):
        w = user.threshold_binaryW_from_shapefile(pysal_examples.get_path(
            "columbus.shp"), 0.62, idVariable="POLYID")
        self.assertEqual(w.weights[1], [1, 1])

    def test_threshold_continuousW_from_array(self):
        points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        wid = user.threshold_continuousW_from_array(points, 11.2)
        wds = {wid.neighbors[0][i]: v for i, v in enumerate(wid.weights[0])}
        self.assertEqual(wds, {1: 0.10000000000000001,
                               3: 0.089442719099991588})
        wid2 = user.threshold_continuousW_from_array(points, 11.2, alpha=-2.0)
        wds = {wid2.neighbors[0][i]: v for i, v in enumerate(wid2.weights[0])}
        self.assertEqual(wid2.weights[0], [0.01, 0.0079999999999999984])

    def test_threshold_continuousW_from_shapefile(self):
        w = user.threshold_continuousW_from_shapefile(pysal_examples.get_path(
            "columbus.shp"), 0.62, idVariable="POLYID")
        wds = {w.neighbors[1][i]:v for i,v in enumerate(w.weights[1])}
        self.assertEqual(wds, {2:1.6702346893743334, 3:1.7250729841938093})

    def test_kernelW(self):
        points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        kw = user.kernelW(points)
        wds = {kw.neighbors[0][i]:v for i,v in enumerate(kw.weights[0])}
        self.assertEqual(wds, {0:1.0, 1:0.50000004999999503,
                                          3:0.44098306152674649})
        np.testing.assert_array_almost_equal(
            kw.bandwidth, np.array([[20.000002],
                                    [20.000002],
                                    [20.000002],
                                    [20.000002],
                                    [20.000002],
                                    [20.000002]]))

    def test_min_threshold_dist_from_shapefile(self):
        f = pysal_examples.get_path('columbus.shp')
        min_d = user.min_threshold_dist_from_shapefile(f)
        self.assertAlmostEqual(min_d, 0.61886415807685413)

    def test_kernelW_from_shapefile(self):
        kw = user.kernelW_from_shapefile(pysal_examples.get_path(
            'columbus.shp'), idVariable='POLYID')
        self.assertEqual(set(kw.weights[1]), set([0.0070787731484506233,
                                         0.2052478782400463,
                                         0.23051223027663237,
                                         1.0
                                         ]))
        np.testing.assert_array_almost_equal(
            kw.bandwidth[:3], np.array([[0.75333961], [0.75333961],
                                        [0.75333961]]))

    def test_adaptive_kernelW(self):
        points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        bw = [25.0, 15.0, 25.0, 16.0, 14.5, 25.0]
        kwa = user.adaptive_kernelW(points, bandwidths=bw)
        wds = {kwa.neighbors[0][i]: v for i, v in enumerate(kwa.weights[0])}
        self.assertEqual(wds, {0: 1.0, 1: 0.59999999999999998,
                               3: 0.55278640450004202,
                               4: 0.10557280900008403})
        np.testing.assert_array_almost_equal(kwa.bandwidth,
                                             np.array([[25.], [15.], [25.],
                                                      [16.], [14.5], [25.]]))

        kweag = user.adaptive_kernelW(points, function='gaussian')
        wds = {kweag.neighbors[0][i]: v for i, v in enumerate(kweag.weights[0])}
        self.assertEqual(wds, {0: 0.3989422804014327, 1: 0.26741902915776961,
                               3: 0.24197074871621341})
        np.testing.assert_array_almost_equal(kweag.bandwidth,
                                             np.array([[11.18034101],
                                                       [11.18034101],
                                                       [20.000002],
                                                       [11.18034101],
                                                       [14.14213704],
                                                       [18.02775818]]))

    def test_adaptive_kernelW_from_shapefile(self):
        kwa = user.adaptive_kernelW_from_shapefile(
            pysal_examples.get_path('columbus.shp'))
        wds = {kwa.neighbors[0][i]: v for i, v in enumerate(kwa.weights[0])}
        self.assertEqual(wds, {0: 1.0, 2: 0.03178906767736345,
                                1: 9.9999990066379496e-08})
        np.testing.assert_array_almost_equal(kwa.bandwidth[:3],
                                             np.array([[0.59871832],
                                                       [0.59871832],
                                                       [0.56095647]]))

    def test_build_lattice_shapefile(self):
        of = "lattice.shp"
        user.build_lattice_shapefile(20, 20, of)
        w = user.rook_from_shapefile(of)
        self.assertEqual(w.n, 400)
        os.remove('lattice.shp')
        os.remove('lattice.shx')

    def test_voronoiW(self):
        np.random.seed(12345)
        points = np.random.random((5,2))*10 + 10
        w = user.voronoiW(points)
        self.assertEqual(w.n, 5)
        self.assertEqual(w.neighbors, {0: [1, 2, 3, 4],
                                        1: [0, 2], 2: [0, 1, 4],
                                        3: [0, 4], 4: [0, 2, 3]})


suite = unittest.TestLoader().loadTestsFromTestCase(Testuser)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
