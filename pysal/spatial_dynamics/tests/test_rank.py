import unittest
import pysal
from pysal.spatial_dynamics import rank
import numpy as np


class Theta_Tester(unittest.TestCase):
    def setUp(self):
        f=pysal.open("pysal/examples/mexico.csv")
        vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
        self.y=np.transpose(np.array([f.by_col[v] for v in vnames]))
        self.regime=np.array(f.by_col['esquivel99'])
    def test_Theta(self):
        np.random.seed(10)
        t=rank.Theta(self.y,self.regime,999)
        k = self.y.shape[1]
        obs = t.theta.tolist()
        exp = [[0.41538462, 0.28070175, 0.61363636, 0.62222222, 0.33333333, 0.47222222]]
        for i in range(k-1): 
            self.assertAlmostEqual(exp[0][i],obs[0][i])
        obs = t.pvalue_left.tolist()
        exp = [0.307, 0.077, 0.823, 0.552, 0.045, 0.735]
        for i in range(k-1):
            self.assertAlmostEqual(exp[i],obs[i])
        obs = t.total.tolist()
        exp = [130., 114., 88., 90., 90., 72.]
        for i in range(k-1):
            self.assertAlmostEqual(exp[i],obs[i])
        self.assertEqual(t.max_total,512)


class SpatialTau_Tester(unittest.TestCase):
    def setUp(self):
        f=pysal.open("pysal/examples/mexico.csv")
        vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
        self.y=np.transpose(np.array([f.by_col[v] for v in vnames]))
        regime=np.array(f.by_col['esquivel99'])
        self.w=pysal.weights.regime_weights(regime)
    def test_SpatialTau(self):
        np.random.seed(10)
        k = self.y.shape[1]
        obs = [rank.SpatialTau(self.y[:,i],self.y[:,i+1],self.w,99) for i in range(k-1)]
        exp_wnc = [44., 47., 52., 54., 53., 57.]
        exp_evwnc = [52.354, 53.576, 55.747, 55.556, 53.384, 57.566]
        exp_prandwnc = [0.000, 0.006, 0.031, 0.212, 0.436, 0.390]
        for i in range(k-1):
            self.assertAlmostEqual(exp_wnc[i],obs[i].wnc,3)
            self.assertAlmostEqual(exp_evwnc[i],obs[i].ev_wnc,3)
            self.assertAlmostEqual(exp_prandwnc[i],obs[i].p_rand_wnc,3)


suite = unittest.TestSuite()
test_classes = [Theta_Tester, SpatialTau_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
