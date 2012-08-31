import unittest
from pysal.weights.Distance import DistanceBand
from pysal.esda import getisord
import numpy as np

POINTS = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
W = DistanceBand(POINTS, threshold=15)
Y = np.array([2, 3, 3.2, 5, 8, 7])


class G_Tester(unittest.TestCase):

    def setUp(self):
        self.w = W
        self.y = Y
        np.random.seed(10)

    def test_G(self):
        g = getisord.G(self.y, self.w)
        self.assertAlmostEquals(g.G, 0.55709779, places=8)
        self.assertAlmostEquals(g.p_norm, 0.1729, places=4)


class G_Local_Tester(unittest.TestCase):

    def setUp(self):
        self.w = W
        self.y = Y
        np.random.seed(10)

    def test_G_Local_Binary(self):
        lg = getisord.G_Local(self.y, self.w, transform='B')
        self.assertAlmostEquals(lg.Zs[0], -1.0136729, places=7)
        self.assertAlmostEquals(lg.p_sim[0], 0.10100000000000001, places=7)

    def test_G_Local_Row_Standardized(self):
        lg = getisord.G_Local(self.y, self.w, transform='R')
        self.assertAlmostEquals(lg.Zs[0], -0.62074534, places=7)
        self.assertAlmostEquals(lg.p_sim[0], 0.10100000000000001, places=7)

    def test_G_star_Local_Binary(self):
        lg = getisord.G_Local(self.y, self.w, transform='B', star=True)
        self.assertAlmostEquals(lg.Zs[0], -1.39727626, places=8)
        self.assertAlmostEquals(lg.p_sim[0], 0.10100000000000001, places=7)

    def test_G_star_Row_Standardized(self):
        lg = getisord.G_Local(self.y, self.w, transform='R', star=True)
        self.assertAlmostEquals(lg.Zs[0], -0.62488094, places=8)
        self.assertAlmostEquals(lg.p_sim[0], 0.10100000000000001, places=7)

suite = unittest.TestSuite()
test_classes = [G_Tester, G_Local_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
