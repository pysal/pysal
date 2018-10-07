from .. import sphere
from ...io.fileio import FileIO as psopen
from ... import examples as pysal_examples
import math
import unittest
import numpy as np


class Sphere(unittest.TestCase):

    def setUp(self):
        self.pt0 = (0, 0)
        self.pt1 = (180, 0)
        f = psopen(pysal_examples.get_path('stl_hom.shp'), 'r')
        self.shapes = f.read()
        self.p0 = (-87.893517, 41.981417)
        self.p1 = (-87.519295, 41.657498)
        self.p3 = (41.981417, -87.893517)
        self.p4 = (41.657498, -87.519295)

    def test_arcdist(self):
        d = sphere.arcdist(self.pt0, self.pt1, sphere.RADIUS_EARTH_MILES)
        self.assertEqual(d, math.pi * sphere.RADIUS_EARTH_MILES)

    def test_arcdist2linear(self):
        d = sphere.arcdist(self.pt0, self.pt1, sphere.RADIUS_EARTH_MILES)
        ld = sphere.arcdist2linear(d, sphere.RADIUS_EARTH_MILES)
        self.assertEqual(ld, 2.0)

    def test_radangle(self):
        p0 = (-87.893517, 41.981417)
        p1 = (-87.519295, 41.657498)
        self.assertAlmostEqual(sphere.radangle(p0,p1), 0.007460167953189258)

    def test_linear2arcdist(self):
        d = sphere.arcdist(self.pt0, self.pt1, sphere.RADIUS_EARTH_MILES)
        ad = sphere.linear2arcdist(2.0, radius=sphere.RADIUS_EARTH_MILES)
        self.assertEqual(d, ad)

    def test_harcdist(self):
        d1 = sphere.harcdist(self.p0, self.p1,
                             radius=sphere.RADIUS_EARTH_MILES)
        self.assertAlmostEqual(d1, 29.532983644123796)
        d1 = sphere.harcdist(self.p3, self.p4,
                             radius=sphere.RADIUS_EARTH_MILES)
        self.assertAlmostEqual(d1, 25.871647470233675)

    def test_geointerpolate(self):
        pn1 = sphere.geointerpolate(self.p0, self.p1, 0.1)
        self.assertAlmostEqual(pn1, (-87.85592403438788, 41.949079912574796))
        pn2 = sphere.geointerpolate(self.p3, self.p4, 0.1, lonx=False)
        self.assertAlmostEqual(pn2, (41.949079912574796, -87.85592403438788))

    def test_geogrid(self):
        grid = [(42.023768, -87.946389), (42.02393997819538,
                                          -87.80562679358316),
                (42.02393997819538, -87.66486420641684), (42.023768,
                                                          -87.524102),
                (41.897317, -87.94638900000001), (41.8974888973743,
                                                  -87.80562679296166),
                (41.8974888973743, -87.66486420703835), (41.897317,
                                                         -87.524102),
                (41.770866000000005, -87.94638900000001), (41.77103781320412,
                                                           -87.80562679234043),
                (41.77103781320412, -87.66486420765956), (41.770866000000005,
                                                          -87.524102),
                (41.644415, -87.946389), (41.64458672568646,
                                          -87.80562679171955),
                (41.64458672568646, -87.66486420828045), (41.644415,
                                                          -87.524102)]

        pup = (42.023768, -87.946389)    # Arlington Heights IL
        pdown = (41.644415, -87.524102)  # Hammond, IN
        grid1 = sphere.geogrid(pup, pdown, 3, lonx=False)
        np.testing.assert_array_almost_equal(grid, grid1)


    def test_toXYZ(self):
        w2 = {0: [2, 5, 6, 10], 1: [4, 7, 9, 14], 2: [6, 0, 3, 8],
              3: [8, 2, 12, 4], 4: [1, 9, 12, 3], 5: [11, 10, 0, 15],
              6: [2, 10, 8, 0], 7: [14, 1, 16, 9], 8: [12, 3, 19, 6],
              9: [12, 16, 4, 1], 10: [17, 6, 15, 5], 11: [15, 13, 5, 21],
              12: [8, 19, 9, 3], 13: [21, 11, 15, 28], 14:
              [7, 16, 22, 9], 15: [11, 27, 10, 26], 16: [14, 25, 9, 20], 17:
              [31, 18, 10, 26], 18: [17, 19, 23, 32], 19: [23, 20, 12, 18], 20:
              [23, 25, 19, 34], 21: [13, 28, 27, 15], 22: [30, 14, 29, 24], 23:
              [20, 19, 18, 34], 24: [30, 22, 41, 43], 25: [20, 16, 33, 34], 26:
              [31, 27, 38, 17], 27: [35, 28, 26, 21], 28: [21, 37, 27, 35], 29:
              [33, 30, 22, 25], 30: [24, 29, 43, 22], 31: [40, 26, 17, 32], 32:
              [39, 45, 31, 18], 33: [29, 25, 44, 34], 34: [36, 25, 39, 33], 35:
              [27, 37, 46, 38], 36: [39, 34, 50, 48], 37: [47, 28, 35, 46], 38:
              [51, 35, 26, 40], 39: [36, 45, 32, 34], 40: [49, 31, 38, 45], 41:
              [52, 43, 30, 53], 42: [43, 44, 33, 53], 43: [42, 53, 41, 30], 44:
              [42, 33, 50, 58], 45: [48, 39, 32, 40], 46: [47, 35, 55, 37], 47:
              [46, 37, 54, 35], 48: [45, 50, 56, 39], 49: [40, 57, 51, 45], 50:
              [48, 36, 59, 44], 51: [61, 38, 55, 49], 52: [41, 53, 64, 43], 53:
              [60, 43, 52, 64], 54: [55, 47, 46, 61], 55: [54, 61, 46, 51], 56:
              [62, 66, 48, 57], 57: [49, 65, 61, 56], 58: [59, 60, 68, 44], 59:
              [58, 63, 50, 69], 60: [53, 64, 68, 58], 61: [67, 51, 55, 57], 62:
              [63, 56, 66, 48], 63: [62, 70, 69, 59], 64: [60, 53, 52, 71], 65:
              [57, 72, 66, 67], 66: [62, 56, 75, 65], 67: [61, 65, 72, 55], 68:
              [60, 58, 76, 71], 69: [73, 70, 63, 59], 70: [74, 63, 69, 77], 71:
              [68, 76, 64, 60], 72: [65, 75, 67, 66], 73: [69, 76, 77, 68], 74:
              [75, 70, 77, 66], 75: [74, 66, 72, 65], 76: [73, 68, 71, 69], 77:
              [70, 74, 73, 69]}

        pts = [shape.centroid for shape in self.shapes]
        pts = list(map(sphere.toXYZ, pts))
        self.assertAlmostEqual(sphere.brute_knn(pts, 4, 'xyz'), w2)

if __name__ == '__main__':
    unittest.main()
