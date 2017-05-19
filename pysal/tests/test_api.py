from .. import api as ps
from libpysal.common import pandas, RTOL, ATOL

import numpy as np
import unittest as ut


class Weights_Test(ut.TestCase):

    def test_queen_from_shapefile(self):
        w = ps.queen_from_shapefile(ps.get_path("NAT.shp"))
        self.assertEqual(w.n, 3085)

class Esda_Test(ut.TestCase):

    def setUp(self):
        self.w = ps.open(ps.get_path("stl.gal")).read()
        f = ps.open(ps.get_path('stl_hom.txt'))
        self.y = np.array(f.by_col['HR8893'])

    def test_moran(self):
        mi = ps.esda.moran.Moran(self.y, self.w, two_tailed=False)
        np.testing.assert_allclose(mi.I,  0.24365582621771659, rtol=RTOL, atol=ATOL)



suite = ut.TestSuite()
test_classes = [Weights_Test]
for i in test_classes:
    a = ut.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = ut.TextTestRunner()
    runner.run(suite)
