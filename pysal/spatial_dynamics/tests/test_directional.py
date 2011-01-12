import unittest
import pysal
from pysal.spatial_dynamics import directional 
import numpy as np


class Rose_Tester(unittest.TestCase):
    def setUp(self):
        result = directional.load_directional(fpath="../../examples/spi_download.csv",gpath="../../examples/states48.gal")
        self.Y = result[0]
        self.w = result[1]
    def test_rose(self):
        k = 4
        np.random.seed(100)
        r4=directional.rose(self.Y,self.w,k,permutations=999)
        exp = [0., 1.57079633, 3.14159265, 4.71238898, 6.28318531]
        obs = list(r4['cuts'])
        for i in range(k+1):
            self.assertAlmostEqual(exp[i],obs[i])
        self.assertEquals(list(r4['counts']), [32,  5,  9,  2])
        exp = [0.02, 1., 0.001, 1.]
        obs = list(r4['pvalues'])
        for i in range(k):
            self.assertAlmostEqual(exp[i],obs[i])


suite = unittest.TestSuite()
test_classes = [Rose_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)



