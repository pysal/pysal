import unittest
from .. import ergodic
import numpy as np


class SteadyState_Tester(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[.5, .25, .25], [.5, 0, .5], [.25, .25, .5]])

    def test_steady_state(self):
        obs = ergodic.steady_state(self.p)
        exp = np.array([0.4, 0.2, 0.4])
        np.testing.assert_array_almost_equal(exp, obs)


class Fmpt_Tester(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[.5, .25, .25], [.5, 0, .5], [.25, .25, .5]])

    def test_fmpt(self):
        obs = ergodic.fmpt(self.p)
        exp = np.array([[2.5, 4., 3.33333333], [2.66666667, 5.,
                                                 2.66666667], [3.33333333, 4., 2.5]])
        np.testing.assert_array_almost_equal(exp, obs)


class VarFmpt_Tester(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[.5, .25, .25], [.5, 0, .5], [.25, .25, .5]])

    def test_var_fmpt(self):
        obs = ergodic.var_fmpt(self.p)
        exp = np.array([[5.58333333, 12., 6.88888889], [6.22222222,
                                                         12., 6.22222222], [6.88888889, 12., 5.58333333]])
        np.testing.assert_array_almost_equal(exp, obs)


suite = unittest.TestSuite()
test_classes = [SteadyState_Tester, Fmpt_Tester, VarFmpt_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
