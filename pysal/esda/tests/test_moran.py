import unittest
import pysal
from .. import moran
from ...common import pandas, RTOL, ATOL 
import numpy as np

PANDAS_EXTINCT = pandas is None

class Moran_Tester(unittest.TestCase):
    def setUp(self):
        self.w = pysal.open(pysal.examples.get_path("stl.gal")).read()
        f = pysal.open(pysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col['HR8893'])

    def test_moran(self):
        mi = moran.Moran(self.y, self.w, two_tailed=False)
        np.testing.assert_allclose(mi.I,  0.24365582621771659, rtol=RTOL, atol=ATOL)
        self.assertAlmostEquals(mi.p_norm, 0.00013573931385468807)

    def test_sids(self):
        w = pysal.open(pysal.examples.get_path("sids2.gal")).read()
        f = pysal.open(pysal.examples.get_path("sids2.dbf"))
        SIDR = np.array(f.by_col("SIDR74"))
        mi = pysal.Moran(SIDR, w, two_tailed=False)
        np.testing.assert_allclose(mi.I, 0.24772519320480135, atol=ATOL, rtol=RTOL)
        self.assertAlmostEquals(mi.p_norm,  5.7916539074498452e-05)

    @unittest.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_by_col(self):
        import pysal.contrib.pdio as pdio
        df = pdio.read_files(pysal.examples.get_path('sids2.dbf'))
        w = pysal.open(pysal.examples.get_path("sids2.gal")).read()
        mi = moran.Moran.by_col(df, ['SIDR74'], w=w, two_tailed=False)
        sidr = np.unique(mi.SIDR74_moran.values)
        pval = np.unique(mi.SIDR74_p_sim.values)
        np.testing.assert_allclose(sidr, 0.24772519320480135, atol=ATOL, rtol=RTOL)
        self.assertAlmostEquals(pval, 0.001)


class Moran_Rate_Tester(unittest.TestCase):
    def setUp(self):
        self.w = pysal.open(pysal.examples.get_path("sids2.gal")).read()
        f = pysal.open(pysal.examples.get_path("sids2.dbf"))
        self.e = np.array(f.by_col['SID79'])
        self.b = np.array(f.by_col['BIR79'])

    def test_moran_rate(self):
        mi = moran.Moran_Rate(self.e, self.b, self.w, two_tailed=False)
        np.testing.assert_allclose(mi.I, 0.16622343552567395, rtol=RTOL, atol=ATOL)
        self.assertAlmostEquals(mi.p_norm, 0.004191499504892171)

    @unittest.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_by_col(self):
        import pysal.contrib.pdio as pdio
        df = pdio.read_files(pysal.examples.get_path('sids2.dbf'))
        mi = moran.Moran_Rate.by_col(df, ['SID79'], ['BIR79'], w=self.w, two_tailed=False)
        sidr = np.unique(mi["SID79-BIR79_moran_rate"].values)
        pval = np.unique(mi["SID79-BIR79_p_sim"].values)
        np.testing.assert_allclose(sidr, 0.16622343552567395, rtol=RTOL, atol=ATOL)
        self.assertAlmostEquals(pval, 0.009)



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
        self.assertAlmostEquals(lm.z_sim[0], -0.68493799168603808)
        self.assertAlmostEquals(lm.p_z_sim[0],  0.24669152541631179)

    @unittest.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_by_col(self):
        import pandas as pd
        df = pd.DataFrame(self.y, columns =['z'])
        lm = moran.Moran_Local.by_col(df, ['z'], w=self.w, transformation='r',
                permutations=99, outvals=['z_sim', 'p_z_sim'])
        self.assertAlmostEquals(lm.z_z_sim[0], -0.68493799168603808)
        self.assertAlmostEquals(lm.z_p_z_sim[0],  0.24669152541631179)


class Moran_Local_BV_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.w = pysal.open(pysal.examples.get_path("sids2.gal")).read()
        f = pysal.open(pysal.examples.get_path("sids2.dbf"))
        self.x = np.array(f.by_col['SIDR79'])
        self.y = np.array(f.by_col['SIDR74'])

    def test_Moran_Local_BV(self):
        lm = moran.Moran_Local_BV(self.x, self.y, self.w,
                                  transformation="r", permutations=99)
        self.assertAlmostEquals(lm.Is[0], 1.4649221250620736)
        self.assertAlmostEquals(lm.z_sim[0],  1.5816540860500772)
        self.assertAlmostEquals(lm.p_z_sim[0], 0.056864279811026153)

    @unittest.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_by_col(self):
        import pysal.contrib.pdio as pdio
        df = pdio.read_files(pysal.examples.get_path('sids2.dbf'))
        np.random.seed(12345)
        moran.Moran_Local_BV.by_col(df, ['SIDR74', 'SIDR79'], w=self.w,
                                    inplace=True, outvals=['z_sim', 'p_z_sim'],
                                    transformation='r', permutations=99)
        bvstats = df['SIDR79-SIDR74_moran_local_bv'].values
        bvz = df['SIDR79-SIDR74_z_sim'].values
        bvzp = df['SIDR79-SIDR74_p_z_sim'].values
        self.assertAlmostEquals(bvstats[0], 1.4649221250620736)
        self.assertAlmostEquals(bvz[0],  1.657427, 5)
        self.assertAlmostEquals(bvzp[0], 0.048717, 5)


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
        self.assertAlmostEquals(lm.z_sim[0], -0.13699844503985936, 7)
        self.assertAlmostEquals(lm.p_z_sim[0], 0.44551601210081715)

    @unittest.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_by_col(self):
        import pysal.contrib.pdio as pdio
        df = pdio.read_files(pysal.examples.get_path('sids2.dbf'))
        lm = moran.Moran_Local_Rate.by_col(df, ['SID79'], ['BIR79'], w=self.w,
                                           outvals=['p_z_sim', 'z_sim'],
                                           transformation='r', permutations=99)
        self.assertAlmostEquals(lm['SID79-BIR79_z_sim'][0],  -0.13699844503985936, 7)
        self.assertAlmostEquals(lm['SID79-BIR79_p_z_sim'][0], 0.44551601210081715)



suite = unittest.TestSuite()
test_classes = [Moran_Tester, Moran_Rate_Tester,
                Moran_BV_matrix_Tester, Moran_Local_Tester,
                Moran_Local_BV_Tester, Moran_Local_Rate_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
