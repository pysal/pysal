import unittest
import numpy as np
import libpysal
from inequality.gini import Gini, Gini_Spatial


class Gini_Tester(unittest.TestCase):

    def setUp(self):
        f = libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        self.y = y[:, 0]
        regimes = np.array(f.by_col('hanson98'))
        self.w = libpysal.weights.block_weights(regimes)

    def test_Gini(self):
        g = Gini(self.y)
        np.testing.assert_almost_equal(g.g, 0.35372371173452849)

    def test_Gini_Spatial(self):
        np.random.seed(12345)
        g = Gini_Spatial(self.y, self.w)
        np.testing.assert_almost_equal(g.g, 0.35372371173452849)
        np.testing.assert_almost_equal(g.wg, 884130.0)
        np.testing.assert_almost_equal(g.wcg, 4353856.0)
        np.testing.assert_almost_equal(g.p_sim, 0.040)
        np.testing.assert_almost_equal(g.e_wcg, 4170356.7474747472)


if __name__ == '__main__':
    unittest.main()
