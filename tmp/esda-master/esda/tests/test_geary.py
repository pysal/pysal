"""Geary Unittest."""
import unittest

from libpysal.io import open as popen
from libpysal import examples
from libpysal.common import pandas

from .. import geary
import numpy as np

PANDAS_EXTINCT = pandas is None

class Geary_Tester(unittest.TestCase):
    """Geary class for unit tests."""
    def setUp(self):
        self.w = popen(examples.get_path("book.gal")).read()
        f = popen(examples.get_path("book.txt"))
        self.y = np.array(f.by_col['y'])

    def test_Geary(self):
        c = geary.Geary(self.y, self.w, permutations=0)
        self.assertAlmostEqual(c.C, 0.33301083591331254)
        self.assertAlmostEqual(c.EC, 1.0)

        self.assertAlmostEqual(c.VC_norm, 0.031805300245097874)
        self.assertAlmostEqual(c.p_norm, 9.2018240680169505e-05)
        self.assertAlmostEqual(c.z_norm, -3.7399778367629564)
        self.assertAlmostEqual(c.seC_norm, 0.17834040553138225)

        self.assertAlmostEquals(c.VC_rand,0.033411917666958356)
        self.assertAlmostEquals(c.p_rand,0.00013165646189214729)
        self.assertAlmostEquals(c.z_rand, -3.6489513837253944)
        self.assertAlmostEquals(c.seC_rand, 0.18278927120309429)

        np.random.seed(12345)
        c = geary.Geary(self.y, self.w, permutations=999)
        self.assertAlmostEquals(c.C, 0.33301083591331254)
        self.assertAlmostEquals(c.EC, 1.0)

        self.assertAlmostEquals(c.VC_norm, 0.031805300245097874)
        self.assertAlmostEquals(c.p_norm, 9.2018240680169505e-05)
        self.assertAlmostEquals(c.z_norm, -3.7399778367629564)
        self.assertAlmostEquals(c.seC_norm, 0.17834040553138225)

        self.assertAlmostEquals(c.VC_rand,0.033411917666958356)
        self.assertAlmostEquals(c.p_rand,0.00013165646189214729)
        self.assertAlmostEquals(c.z_rand, -3.6489513837253944)
        self.assertAlmostEquals(c.seC_rand, 0.18278927120309429)


        self.assertAlmostEquals(c.EC_sim, 0.9980676303238214)
        self.assertAlmostEquals(c.VC_sim, 0.034430408799858946)
        self.assertAlmostEquals(c.p_sim, 0.001)
        self.assertAlmostEquals(c.p_z_sim, 0.00016908100514811952)
        self.assertAlmostEquals(c.z_sim, -3.5841621159171746)
        self.assertAlmostEquals(c.seC_sim, 0.18555432843202269)

    @unittest.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_by_col(self):
        import pandas as pd
        df = pd.DataFrame(self.y, columns=['y'])
        r1 = geary.Geary.by_col(df, ['y'], w=self.w, permutations=999)
        this_geary = np.unique(r1.y_geary.values)
        this_pval = np.unique(r1.y_p_sim.values)
        np.random.seed(12345)
        c = geary.Geary(self.y, self.w, permutations=999)
        self.assertAlmostEqual(this_geary, c.C)
        self.assertAlmostEqual(this_pval, c.p_sim)



suite = unittest.TestSuite()
test_classes = [Geary_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
