import unittest
import pysal
from pysal.esda import moran
import numpy as np


class Moran_Tester(unittest.TestCase):
    def setUp(self):
        self.w = pysal.open(pysal.examples.get_path("stl.gal")).read()
        f = pysal.open(pysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col['HR8893'])

    def test_moran(self):
        mi = moran.Moran(self.y, self.w)
        self.assertAlmostEquals(mi.I, 0.24365582621771659, 7)
        self.assertAlmostEquals(mi.p_norm, 0.00027147862770937614)

    def test_sids(self):
        w = pysal.open(pysal.examples.get_path("sids2.gal")).read()
        f = pysal.open(pysal.examples.get_path("sids2.dbf"))
        SIDR = np.array(f.by_col("SIDR74"))
        mi = pysal.Moran(SIDR, w)
        self.assertAlmostEquals(mi.I, 0.24772519320480135)
        self.assertAlmostEquals(mi.p_norm, 0.00011583330781)


class Moran_Rate_Tester(unittest.TestCase):
    def setUp(self):
        self.w = pysal.open(pysal.examples.get_path("sids2.gal")).read()
        f = pysal.open(pysal.examples.get_path("sids2.dbf"))
        self.e = np.array(f.by_col['SID79'])
        self.b = np.array(f.by_col['BIR79'])

    def test_moran_rate(self):
        mi = moran.Moran_Rate(self.e, self.b, self.w)
        self.assertAlmostEquals(mi.I, 0.16622343552567395, 7)
        self.assertAlmostEquals(mi.p_norm, 0.0083829990097843421)


class Moran_BV_matrix_Tester(unittest.TestCase):
    def setUp(self):
        f = pysal.open(pysal.examples.get_path("sids2.dbf"))
        varnames = ['SIDR74', 'SIDR79', 'NWR74', 'NWR79']
        self.names = varnames
        vars = [np.array(f.by_col[var]) for var in varnames]
        self.vars = vars
        self.w = pysal.open(pysal.examples.get_path("sids2.gal")).read()

    def test_Moran_BV_matrix(self):
        res = moran.Moran_BV_matrix(self.vars, self.w, varnames=self.names)
        self.assertAlmostEquals(res[(0, 1)].I, 0.19362610652874668)
        self.assertAlmostEquals(res[(3, 0)].I, 0.37701382542927858)


class Moran_Local_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.w = pysal.open(pysal.examples.get_path("desmith.gal")).read()
        f = pysal.open(pysal.examples.get_path("desmith.txt"))
        self.y = np.array(f.by_col['z'])

    def test_Moran_Local(self):
        lm = moran.Moran_Local(
            self.y, self.w, transformation="r", permutations=99)
        self.assertAlmostEquals(lm.z_sim[0], -0.081383956359666748)
        self.assertAlmostEquals(lm.p_z_sim[0], 0.46756830387716064)
        self.assertAlmostEquals(lm.VI_sim, 0.2067126047680822)


class Moran_Local_Rate_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.w = pysal.open(pysal.examples.get_path("sids2.gal")).read()
        f = pysal.open(pysal.examples.get_path("sids2.dbf"))
        self.e = np.array(f.by_col['SID79'])
        self.b = np.array(f.by_col['BIR79'])

    def test_moran_rate(self):
        lm = moran.Moran_Local_Rate(self.e, self.b, self.w,
                                    transformation="r", permutations=99)
        self.assertAlmostEquals(lm.z_sim[0], -0.27099998923550017)
        self.assertAlmostEquals(lm.p_z_sim[0], 0.39319552026912641)
        self.assertAlmostEquals(lm.VI_sim, 0.21879403675396222)


suite = unittest.TestSuite()
test_classes = [Moran_Tester, Moran_BV_matrix_Tester, Moran_Local_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
