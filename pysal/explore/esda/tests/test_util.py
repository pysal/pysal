import unittest
import pysal.lib
from .. import util
from .. import moran
import numpy as np

class Fdr_Tester(unittest.TestCase):
    def setUp(self):
        self.w = pysal.lib.io.open(pysal.lib.examples.get_path("stl.gal")).read()
        f = pysal.lib.io.open(pysal.lib.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col['HR8893'])

    def test_fdr(self):
        np.random.seed(10)
        lm = moran.Moran_Local(self.y, self.w, transformation="r",
                               permutations=999)
        self.assertAlmostEqual(util.fdr(lm.p_sim, 0.1),
                               0.002564102564102564)
        self.assertAlmostEqual(util.fdr(lm.p_sim, 0.05), 0.000641025641025641)


suite = unittest.TestSuite()
test_classes = [Fdr_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
