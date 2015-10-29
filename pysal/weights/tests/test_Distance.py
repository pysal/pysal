import os
import unittest
import pysal
import numpy as np


class TestDistanceWeights(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        self.polyShp = pysal.examples.get_path('columbus.shp')
        self.arcShp = pysal.examples.get_path('stl_hom.shp')
        self.points = [(
            10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    def test_knnW(self):
        kd = pysal.KDTree(np.array(self.points), distance_metric='euclidean')
        wnn2 = pysal.knnW(kd, 2)
        self.assertEqual(wnn2.neighbors[0], [1,3])

        wnn3 = pysal.knnW(np.array(self.points), 2) # this form is to be deprecated
        self.assertEqual(wnn2.neighbors[0], [1,3])

        pts = [i.centroid for i in pysal.open(self.polyShp)]
        kd = pysal.cg.kdtree.KDTree(pts)
        wnn4 = pysal.knnW(kd, 4)
        self.assertEqual(wnn4.neighbors[0], [2,1,3,7])
        self.assertEqual(wnn4.neighbors[7], [3,6,12,11])

    def test_knnW_arc(self):
        pts = [x.centroid for x in pysal.open(self.arcShp)]
        dist = pysal.cg.sphere.arcdist  # default radius is Earth KM
        full = np.matrix([[dist(pts[i], pts[j]) for j in xrange(
            len(pts))] for i in xrange(len(pts))])

        kd = pysal.cg.kdtree.KDTree(pts, distance_metric='Arc',
                                    radius=pysal.cg.sphere.RADIUS_EARTH_KM)
        w = pysal.knnW(kd, 4)
        self.assertEqual(set(w.neighbors[4]), set([1,3,9,12]))
        self.assertEqual(set(w.neighbors[40]), set([31,38,45,49]))
        #self.assertTrue((full.argsort()[:, 1:5] == np.array(
        #    [w.neighbors[x] for x in range(len(pts))])).all())

    def test_Kernel(self):
        kw = pysal.Kernel(pysal.KDTree(np.array(self.points)))
        self.assertEqual(kw.weights[0], [1.0, 0.50000004999999503,
                                         0.44098306152674649])
        kw = pysal.Kernel(np.array(self.points)) # to be deprecated
        self.assertEqual(kw.weights[0], [1.0, 0.50000004999999503,
                                         0.44098306152674649])
        kw15 = pysal.Kernel(pysal.KDTree(self.points), bandwidth=15.0)
        self.assertEqual(kw15[0], {0: 1.0, 1: 0.33333333333333337,
                                   3: 0.2546440075000701})
        self.assertEqual(kw15.bandwidth[0], 15.)
        self.assertEqual(kw15.bandwidth[-1], 15.)
        bw = [25.0, 15.0, 25.0, 16.0, 14.5, 25.0]
        kwa = pysal.Kernel(pysal.KDTree(self.points), bandwidth=bw)
        self.assertEqual(kwa.weights[0], [1.0, 0.59999999999999998,
                                          0.55278640450004202,
                                          0.10557280900008403])
        self.assertEqual(kwa.neighbors[0], [0, 1, 3, 4])
        self.assertEqual(kwa.bandwidth[0], 25.)
        self.assertEqual(kwa.bandwidth[1], 15.)
        self.assertEqual(kwa.bandwidth[2], 25.)
        self.assertEqual(kwa.bandwidth[3], 16.)
        self.assertEqual(kwa.bandwidth[4], 14.5)
        self.assertEqual(kwa.bandwidth[5], 25.)
        kwea = pysal.Kernel(pysal.KDTree(self.points), fixed=False)
        self.assertEqual(kwea.weights[0], [1.0, 0.10557289844279438,
                                           9.9999990066379496e-08])
        l = kwea.bandwidth.tolist()
        self.assertEqual(l, [[11.180341005532938], [11.180341005532938],
                             [20.000002000000002], [11.180341005532938],
                             [14.142137037944515], [18.027758180095585]])
        kweag = pysal.Kernel(pysal.KDTree(self.points), fixed=False, function='gaussian')
        self.assertEqual(kweag.weights[0], [0.3989422804014327,
                                            0.26741902915776961,
                                            0.24197074871621341])
        l = kweag.bandwidth.tolist()
        self.assertEqual(l, [[11.180341005532938], [11.180341005532938],
                            [20.000002000000002], [11.180341005532938],
                            [14.142137037944515], [18.027758180095585]])

        kw = pysal.kernelW_from_shapefile(self.polyShp, idVariable='POLYID')
        self.assertEqual(set(kw.weights[1]), set([0.0070787731484506233,
                                         0.2052478782400463,
                                         0.23051223027663237,
                                         1.0
                                         ]))
        kwa = pysal.adaptive_kernelW_from_shapefile(self.polyShp)
        self.assertEqual(kwa.weights[0], [1.0, 0.03178906767736345,
                                          9.9999990066379496e-08])

    def test_threshold(self):
        md = pysal.min_threshold_dist_from_shapefile(self.polyShp)
        self.assertEqual(md, 0.61886415807685413)
        wid = pysal.threshold_continuousW_from_array(pysal.KDTree(self.points), 11.2)
        self.assertEqual(wid.weights[0], [0.10000000000000001,
                                          0.089442719099991588])
        wid = pysal.threshold_continuousW_from_array(np.array(self.points), 11.2) #to be deprecated
        self.assertEqual(wid.weights[0], [0.10000000000000001,
                                          0.089442719099991588])
        wid2 = pysal.threshold_continuousW_from_array(
            pysal.KDTree(self.points), 11.2, alpha=-2.0)
        self.assertEqual(wid2.weights[0], [0.01, 0.0079999999999999984])
        w = pysal.threshold_continuousW_from_shapefile(
            self.polyShp, 0.62, idVariable="POLYID")
        self.assertEqual(w.weights[1], [1.6702346893743334,
                                        1.7250729841938093])

    def test_DistanceBand(self):
        """ see issue #126 """
        w = pysal.rook_from_shapefile(
            pysal.examples.get_path("lattice10x10.shp"))
        polygons = pysal.open(
            pysal.examples.get_path("lattice10x10.shp"), "r").read()
        points1 = [poly.centroid for poly in polygons]
        kd = pysal.KDTree(np.array(points1), distance_metric='euclidean')
        w1 = pysal.DistanceBand(kd, 1)
        for k in range(w.n):
            self.assertEqual(w[k], w1[k])

        w1 = pysal.DistanceBand(np.array(points1), 1)
        for k in range(w.n):
            self.assertEqual(w[k], w1[k])

    def test_DistanceBand_ints(self):
        """ see issue #126 """
        """ we now require scipy 0.11, which is after the fix of this bug
            https://github.com/scipy/scipy/issues/1898"""
        w = pysal.rook_from_shapefile(
            pysal.examples.get_path("lattice10x10.shp"))
        polygons = pysal.open(
            pysal.examples.get_path("lattice10x10.shp"), "r").read()
        points2 = [tuple(map(int, poly.vertices[0])) for poly in polygons]
        w2 = pysal.DistanceBand(np.array(points2), 1)
        for k in range(w.n):
            self.assertEqual(w[k], w2[k])

    def test_DistanceBand_arc(self):
        pts = [x.centroid for x in pysal.open(self.arcShp)]
        dist = pysal.cg.sphere.arcdist  # default radius is Earth KM
        full = np.matrix([[dist(pts[i], pts[j]) for j in xrange(
            len(pts))] for i in xrange(len(pts))])

        kd = pysal.cg.kdtree.KDTree(np.array(pts), distance_metric='Arc',
                                    radius=pysal.cg.sphere.RADIUS_EARTH_KM)
        w = pysal.DistanceBand(kd, full.max(), binary=False, alpha=1.0)
        self.assertTrue((w.sparse.todense() == full).all())
        #np.allclose(w.sparse.todense(), full)


suite = unittest.TestLoader().loadTestsFromTestCase(TestDistanceWeights)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
