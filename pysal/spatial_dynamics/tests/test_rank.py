import unittest
import pysal
from pysal.spatial_dynamics import rank
import numpy as np


class Theta_Tester(unittest.TestCase):
    def setUp(self):
        f = pysal.open(pysal.examples.get_path('mexico.csv'))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        self.y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        self.regime = np.array(f.by_col['esquivel99'])

    def test_Theta(self):
        np.random.seed(10)
        t = rank.Theta(self.y, self.regime, 999)
        k = self.y.shape[1]
        obs = t.theta.tolist()
        exp = [[0.41538462, 0.28070175, 0.61363636, 0.62222222,
                0.33333333, 0.47222222]]
        for i in range(k - 1):
            self.assertAlmostEqual(exp[0][i], obs[0][i])
        obs = t.pvalue_left.tolist()
        exp = [0.307, 0.077, 0.823, 0.552, 0.045, 0.735]
        for i in range(k - 1):
            self.assertAlmostEqual(exp[i], obs[i])
        obs = t.total.tolist()
        exp = [130., 114., 88., 90., 90., 72.]
        for i in range(k - 1):
            self.assertAlmostEqual(exp[i], obs[i])
        self.assertEqual(t.max_total, 512)


class SpatialTau_Tester(unittest.TestCase):
    def setUp(self):
        f = pysal.open(pysal.examples.get_path('mexico.csv'))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        self.y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        regime = np.array(f.by_col['esquivel99'])
        self.w = pysal.weights.block_weights(regime)

    def test_SpatialTau(self):
        np.random.seed(12345)
        k = self.y.shape[1]
        obs = [rank.SpatialTau(self.y[:, i], self.y[:, i + 1],
                               self.w, 99) for i in range(k - 1)]
        tau_s = [0.397, 0.492, 0.651, 0.714, 0.683, 0.810]
        ev_tau_s = [0.659, 0.706, 0.772, 0.752, 0.705, 0.819]
        p_vals = [0.010, 0.010, 0.020, 0.210, 0.270, 0.280]
        for i in range(k - 1):
            self.assertAlmostEqual(tau_s[i], obs[i].tau_spatial, 3)
            self.assertAlmostEqual(ev_tau_s[i], obs[i].taus.mean(), 3)
            self.assertAlmostEqual(p_vals[i], obs[i].tau_spatial_psim, 3)


class Tau_Tester(unittest.TestCase):
    def test_Tau(self):
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        kt = rank.Tau(x1, x2)
        self.assertAlmostEqual(kt.tau, -0.47140452079103173, 5)
        self.assertAlmostEqual(kt.tau_p, 0.24821309157521476, 5)


suite = unittest.TestSuite()
test_classes = [Theta_Tester, SpatialTau_Tester, Tau_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
