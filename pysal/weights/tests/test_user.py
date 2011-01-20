

import os
import unittest
import pysal
import numpy as np

class _Testuser(unittest.TestCase):
    def setUp(self):
        self.wq = pysal.queen_from_shapefile("../../examples/columbus.shp")
        self.wr = pysal.rook_from_shapefile("../../examples/columbus.shp")

    def test_queen_from_shapefile(self):
        self.assertAlmostEquals(self.wq.pct_nonzero, 0.098292378175760101)

    def test_rook_from_shapefile(self):
        self.assertAlmostEquals(self.wr.pct_nonzero,0.083298625572678045)

    def test_knnW_from_array(self):
        import numpy as np
        x,y = np.indices((5,5))
        x.shape = (25,1)
        y.shape = (25,1)
        data = np.hstack([x,y])
        wnn2 = pysal.knnW_from_array(data,k = 2)
        wnn4 = pysal.knnW_from_array(data,k = 4)
        self.assertEquals(wnn4.neighbors[0], [1, 5, 6, 2])
        self.assertEquals(wnn4.neighbors[5], [0, 6, 10, 1])
        self.assertEquals(wnn2.neighbors[0], [1, 5])
        self.assertEquals(wnn2.neighbors[5], [0, 6])
        self.assertAlmostEquals(wnn2.pct_nonzero, 0.080000000000000002)
        self.assertAlmostEquals(wnn4.pct_nonzero,  0.16)
        wnn4 = pysal.knnW_from_array(data,k = 4)
        self.assertEquals(wnn4.neighbors[0], [1, 5, 6, 2])
        wnn3e = pysal.knnW(data,p = 2,k = 3)
        self.assertEquals(wnn3e.neighbors[0], [1, 5, 6])
        wnn3m = pysal.knnW(data,p = 1,k = 3)
        self.assertEquals(wnn3m.neighbors[0], [1, 5, 2])

    def test_knnW_from_shapefile(self):
        wc=pysal.knnW_from_shapefile('../../examples/columbus.shp')
        self.assertAlmostEquals(wc.pct_nonzero, 0.040816326530612242)
        wc3=pysal.knnW_from_shapefile('../../examples/columbus.shp',k=3,idVariable="POLYID")
        self.assertEquals(wc3.weights[1], [1, 1, 1])
        self.assertEquals(wc3.neighbors[1], [3, 2, 4])
        self.assertEquals(wc.neighbors[0], [2, 1])
        w = pysal.knnW_from_shapefile('../../examples/juvenile.shp')
        self.assertAlmostEquals(w.pct_nonzero, 0.011904761904761904)
        w1=pysal.knnW_from_shapefile('../../examples/juvenile.shp',k=1)
        self.assertAlmostEquals(w1.pct_nonzero, 0.0059523809523809521)

    def test_threshold_binarW_from_array(self):
        points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        w= pysal.threshold_binaryW_from_array(points,threshold=11.2)
        self.assertEquals(w.weights, {0: [1, 1], 1: [1, 1], 2: [], 3: [1, 1],
                                      4: [1], 5: [1]})
        self.assertEquals(w.neighbors, {0: [1, 3], 1: [0, 3], 2: [],
                                        3: [0, 1], 4: [5], 5: [4]})

    def test_threshold_binaryW_from_shapefile(self):

        w = pysal.threshold_binaryW_from_shapefile('../../examples/columbus.shp',0.62,idVariable="POLYID")
        self.assertEquals(w.weights[1], [1, 1])
        

    def test_threshold_continousW_from_array(self):
        points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        wid = pysal.threshold_continuousW_from_array(points,11.2)
        self.assertEquals(wid.weights[0], [0.10000000000000001,
                                           0.089442719099991588])
        wid2 = pysal.threshold_continuousW_from_array(points,11.2,alpha=-2.0)
        self.assertEquals(wid2.weights[0], [0.01, 0.0079999999999999984])

    def test_threshold_continuousW_from_shapefile(self):
        w = pysal.threshold_continuousW_from_shapefile('../../examples/columbus.shp',
                                                       0.62,idVariable="POLYID")
        self.assertEquals(w.weights[1], [1.6702346893743276,
                                               1.7250729841938044])
    def test_kernelW(self):
        points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        kw = pysal.kernelW(points)
        self.assertEquals(kw.weights[0], [1.0, 0.50000004999999503,
                                          0.44098306152674649])
        self.assertEquals(kw.neighbors[0], [0, 1, 3])
        np.testing.assert_array_almost_equal(kw.bandwidth, np.array([[ 20.000002],
                                                     [ 20.000002],
                                                     [ 20.000002],
                                                     [ 20.000002],
                                                     [ 20.000002],
                                                     [ 20.000002]]))

        
suite = unittest.TestLoader().loadTestsFromTestCase(_Testuser)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
