import unittest
import numpy as np
import pysal
from pysal.esda.gamma import Gamma


class Gamma_Tester(unittest.TestCase):
    """Unit test for Gamma Index"""
    def setUp(self):
        self.w = pysal.lat2W(4, 4)
        self.y = np.ones(16)
        self.y[0:8] = 0

    def test_Gamma(self):
        """Test method"""
        np.random.seed(12345)
        g = Gamma(self.y, self.w)
        self.assertAlmostEquals(g.g, 20.0)
        self.assertAlmostEquals(g.g_z, 3.1879280354548638)
        self.assertAlmostEquals(g.p_sim_g, 0.0030000000000000001)
        self.assertAlmostEquals(g.min_g, 0.0)
        self.assertAlmostEquals(g.max_g, 20.0)
        self.assertAlmostEquals(g.mean_g, 11.093093093093094)
        np.random.seed(12345)
        g1 = Gamma(self.y, self.w, operation='s')
        self.assertAlmostEquals(g1.g, 8.0)
        self.assertAlmostEquals(g1.g_z, -3.7057554345954791)
        self.assertAlmostEquals(g1.p_sim_g, 0.001)
        self.assertAlmostEquals(g1.min_g, 14.0)
        self.assertAlmostEquals(g1.max_g, 48.0)
        self.assertAlmostEquals(g1.mean_g, 25.623623623623622)
        np.random.seed(12345)
        g2 = Gamma(self.y, self.w, operation='a')
        self.assertAlmostEquals(g2.g, 8.0)
        self.assertAlmostEquals(g2.g_z, -3.7057554345954791)
        self.assertAlmostEquals(g2.p_sim_g, 0.001)
        self.assertAlmostEquals(g2.min_g, 14.0)
        self.assertAlmostEquals(g2.max_g, 48.0)
        self.assertAlmostEquals(g2.mean_g, 25.623623623623622)
        np.random.seed(12345)
        g3 = Gamma(self.y, self.w, standardize='y')
        self.assertAlmostEquals(g3.g, 32.0)
        self.assertAlmostEquals(g3.g_z, 3.7057554345954791)
        self.assertAlmostEquals(g3.p_sim_g, 0.001)
        self.assertAlmostEquals(g3.min_g, -48.0)
        self.assertAlmostEquals(g3.max_g, 20.0)
        self.assertAlmostEquals(g3.mean_g, -3.2472472472472473)
        np.random.seed(12345)

        def func(z, i, j):
            q = z[i] * z[j]
            return q

        g4 = Gamma(self.y, self.w, operation=func)
        self.assertAlmostEquals(g4.g, 20.0)
        self.assertAlmostEquals(g4.g_z, 3.1879280354548638)
        self.assertAlmostEquals(g4.p_sim_g, 0.0030000000000000001)


suite = unittest.TestSuite()
test_classes = [Gamma_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
