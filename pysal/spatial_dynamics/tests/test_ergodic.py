import unittest
import pysal
from pysal.spatial_dynamics import ergodic
import numpy as np


class SteadyState_Tester(unittest.TestCase):
    def setUp(self):
        self.p = np.matrix([[.5, .25, .25], [.5, 0, .5], [.25, .25, .5]])

    def test_steady_state(self):
        obs = ergodic.steady_state(self.p).tolist()
        exp = np.matrix([[0.4], [0.2], [0.4]]).tolist()
        k = self.p.shape[0]
        for i in range(k):
            self.assertAlmostEqual(exp[i][0], obs[i][0])


class Fmpt_Tester(unittest.TestCase):
    def setUp(self):
        self.p = np.matrix([[.5, .25, .25], [.5, 0, .5], [.25, .25, .5]])

    def test_fmpt(self):
        k = self.p.shape[0]
        obs = ergodic.fmpt(self.p).flatten().tolist()[0]
        exp = np.matrix([[2.5, 4., 3.33333333], [2.66666667, 5.,
                                                 2.66666667], [3.33333333, 4., 2.5]])
        exp = exp.flatten().tolist()[0]
        for i in range(k):
            self.assertAlmostEqual(exp[i], obs[i])


class VarFmpt_Tester(unittest.TestCase):
    def setUp(self):
        self.p = np.matrix([[.5, .25, .25], [.5, 0, .5], [.25, .25, .5]])

    def test_var_fmpt(self):
        k = self.p.shape[0]
        obs = ergodic.var_fmpt(self.p).flatten().tolist()[0]
        exp = np.matrix([[5.58333333, 12., 6.88888889], [6.22222222,
                                                         12., 6.22222222], [6.88888889, 12., 5.58333333]])
        exp = exp.flatten().tolist()[0]
        for i in range(k):
            self.assertAlmostEqual(exp[i], obs[i])


suite = unittest.TestSuite()
test_classes = [SteadyState_Tester, Fmpt_Tester, VarFmpt_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
