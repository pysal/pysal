import unittest
import numpy as np

from .. import getisord
from ...weights.Distance import DistanceBand
from ...common import pandas

POINTS = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
W = DistanceBand(POINTS, threshold=15)
Y = np.array([2, 3, 3.2, 5, 8, 7])

PANDAS_EXTINCT = pandas is None

class G_Tester(unittest.TestCase):

    def setUp(self):
        self.w = W
        self.y = Y
        np.random.seed(10)

    def test_G(self):
        g = getisord.G(self.y, self.w)
        self.assertAlmostEquals(g.G, 0.55709779, places=8)
        self.assertAlmostEquals(g.p_norm, 0.1729, places=4)

    @unittest.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_by_col(self):
        import pandas as pd
        df = pd.DataFrame(self.y, columns=['y'])
        np.random.seed(12345)
        r1 = getisord.G.by_col(df, ['y'], w=self.w)
        this_getisord = np.unique(r1.y_g.values)
        this_pval = np.unique(r1.y_p_sim.values)
        np.random.seed(12345)
        stat = getisord.G(self.y, self.w)
        self.assertAlmostEquals(this_getisord, stat._statistic)
        self.assertAlmostEquals(this_pval, stat.p_sim)

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
    
    @unittest.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_by_col(self):
        import pandas as pd
        df = pd.DataFrame(self.y, columns=['y'])
        np.random.seed(12345)
        r1 = getisord.G_Local.by_col(df, ['y'], w=self.w)
        np.random.seed(12345)
        stat = getisord.G_Local(self.y, self.w)
        np.testing.assert_allclose(r1.y_g_local.values, stat.Gs)
        np.testing.assert_allclose(r1.y_p_sim, stat.p_sim)

suite = unittest.TestSuite()
test_classes = [G_Tester, G_Local_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
