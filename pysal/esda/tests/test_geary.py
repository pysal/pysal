"""Geary Unittest."""
import unittest
import pysal
from pysal.esda import geary
import numpy as np


class Geary_Tester(unittest.TestCase):
    """Geary class for unit tests."""
    def setUp(self):
        self.w = pysal.open(pysal.examples.get_path("book.gal")).read()
        f = pysal.open(pysal.examples.get_path("book.txt"))
        self.y = np.array(f.by_col['y'])

    def test_Geary(self):
        c = geary.Geary(self.y, self.w, permutations=0)
        self.assertAlmostEquals(c.C, 0.33301083591331254)
        self.assertAlmostEquals(c.EC, 1.0)

        self.assertAlmostEquals(c.VC_norm, 0.031805300245097874)
        self.assertAlmostEquals(c.p_norm, 9.2018240680169505e-05)
        self.assertAlmostEquals(c.z_norm, -3.7399778367629564)
        self.assertAlmostEquals(c.seC_norm, 0.17834040553138225)

        self.assertAlmostEquals(c.VC_rand, 0.018437747611029367)
        self.assertAlmostEquals(c.p_rand, 4.5059156794646782e-07)
        self.assertAlmostEquals(c.z_rand, -4.9120733751216008)
        self.assertAlmostEquals(c.seC_rand, 0.13578566791465646)

        np.random.seed(12345)
        c = geary.Geary(self.y, self.w, permutations=999)
        self.assertAlmostEquals(c.C, 0.33301083591331254)
        self.assertAlmostEquals(c.EC, 1.0)

        self.assertAlmostEquals(c.VC_norm, 0.031805300245097874)
        self.assertAlmostEquals(c.p_norm, 9.2018240680169505e-05)
        self.assertAlmostEquals(c.z_norm, -3.7399778367629564)
        self.assertAlmostEquals(c.seC_norm, 0.17834040553138225)

        self.assertAlmostEquals(c.VC_rand, 0.018437747611029367)
        self.assertAlmostEquals(c.p_rand, 4.5059156794646782e-07)
        self.assertAlmostEquals(c.z_rand, -4.9120733751216008)
        self.assertAlmostEquals(c.seC_rand, 0.13578566791465646)

        self.assertAlmostEquals(c.EC_sim, 0.9980676303238214)
        self.assertAlmostEquals(c.VC_sim, 0.034430408799858946)
        self.assertAlmostEquals(c.p_sim, 0.001)
        self.assertAlmostEquals(c.p_z_sim, 0.00016908100514811952)
        self.assertAlmostEquals(c.z_sim, -3.5841621159171746)
        self.assertAlmostEquals(c.seC_sim, 0.18555432843202269)


suite = unittest.TestSuite()
test_classes = [Geary_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
