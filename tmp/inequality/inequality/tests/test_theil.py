import unittest
import libpysal
import numpy as np
from inequality.theil import *


class test_Theil(unittest.TestCase):

    def test___init__(self):
        # theil = Theil(y)
        f = libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        theil_y = Theil(y)
        np.testing.assert_almost_equal(theil_y.T, np.array([0.20894344, 0.15222451, 0.10472941, 0.10194725, 0.09560113, 0.10511256, 0.10660832]))


class test_TheilD(unittest.TestCase):
    def test___init__(self):
        # theil_d = TheilD(y, partition)
        f = libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        regimes = np.array(f.by_col('hanson98'))
        theil_d = TheilD(y, regimes)
        np.testing.assert_almost_equal(theil_d.bg, np.array([0.0345889, 0.02816853, 0.05260921, 0.05931219, 0.03205257, 0.02963731, 0.03635872]))


class test_TheilDSim(unittest.TestCase):
    def test___init__(self):
        f = libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        regimes = np.array(f.by_col('hanson98'))
        np.random.seed(10)
        theil_ds = TheilDSim(y, regimes, 999)
        np.testing.assert_almost_equal(theil_ds.bg_pvalue, np.array(
            [0.4, 0.344, 0.001, 0.001, 0.034, 0.072, 0.032]))


if __name__ == '__main__':
    unittest.main()
