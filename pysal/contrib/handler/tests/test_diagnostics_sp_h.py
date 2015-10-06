import unittest
import numpy as np
import pysal
from pysal.contrib.handler import Model
from functools import partial
from pysal.spreg import diagnostics
#from pysal.spreg.ols import OLS as OLS
OLS = Model
#from pysal.spreg.twosls import TSLS as TSLS
TSLS = partial(Model, mtype='TSLS')
#from pysal.spreg.twosls_sp import GM_Lag
GM_Lag = partial(Model, mtype='GM_Lag')
from pysal.spreg.diagnostics_sp import LMtests, MoranRes, spDcache, AKtest


class TestLMtests(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        X = np.array(X).T
        self.y = y
        self.X = X
        ols = OLS(self.y, self.X)
        self.ols = ols
        w = pysal.open(pysal.examples.get_path('columbus.gal'), 'r').read()
        w.transform='r'
        self.w = w

    def test_lm_err(self):
        lms = LMtests(self.ols, self.w)
        lme = np.array([3.097094,  0.078432])
        np.testing.assert_array_almost_equal(lms.lme, lme, decimal=6)

    def test_lm_lag(self):
        lms = LMtests(self.ols, self.w)
        lml = np.array([ 0.981552,  0.321816])
        np.testing.assert_array_almost_equal(lms.lml, lml, decimal=6)

    def test_rlm_err(self):
        lms = LMtests(self.ols, self.w)
        rlme = np.array([ 3.209187,  0.073226])
        np.testing.assert_array_almost_equal(lms.rlme, rlme, decimal=6)

    def test_rlm_lag(self):
        lms = LMtests(self.ols, self.w)
        rlml = np.array([ 1.093645,  0.295665])
        np.testing.assert_array_almost_equal(lms.rlml, rlml, decimal=6)

    def test_lm_sarma(self):
        lms = LMtests(self.ols, self.w)
        sarma = np.array([ 4.190739,  0.123025])
        np.testing.assert_array_almost_equal(lms.sarma, sarma, decimal=6)


class TestMoranRes(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        X = np.array(X).T
        self.y = y
        self.X = X
        ols = OLS(self.y, self.X)
        self.ols = ols
        w = pysal.open(pysal.examples.get_path('columbus.gal'), 'r').read()
        w.transform='r'
        self.w = w
    
    def test_get_m_i(self):
        m = MoranRes(self.ols, self.w, z=True)
        np.testing.assert_array_almost_equal(m.I, 0.17130999999999999, decimal=6)

    def test_get_v_i(self):
        m = MoranRes(self.ols, self.w, z=True)
        np.testing.assert_array_almost_equal(m.vI, 0.0081300000000000001, decimal=6)

    def test_get_e_i(self):
        m = MoranRes(self.ols, self.w, z=True)
        np.testing.assert_array_almost_equal(m.eI, -0.034522999999999998, decimal=6)

    def test_get_z_i(self):
        m = MoranRes(self.ols, self.w, z=True)
        np.testing.assert_array_almost_equal(m.zI, 2.2827389999999999, decimal=6)


class TestAKTest(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
        y = np.array(db.by_col("CRIME"))
        y = np.reshape(y, (49,1))
        self.y = y
        X = []
        X.append(db.by_col("INC"))
        X = np.array(X).T
        self.X = X
        yd = []
        yd.append(db.by_col("HOVAL"))
        yd = np.array(yd).T
        self.yd = yd
        q = []
        q.append(db.by_col("DISCBD"))
        q = np.array(q).T
        self.q = q
        reg = TSLS(y, X, yd, q=q)
        self.reg = reg
        w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        w.transform = 'r'
        self.w = w

    def test_gen_mi(self):
        ak = AKtest(self.reg, self.w)
        np.testing.assert_array_almost_equal(ak.mi, 0.2232672865437263, decimal=6)

    def test_gen_ak(self):
        ak = AKtest(self.reg, self.w)
        np.testing.assert_array_almost_equal(ak.ak, 4.6428948758930852, decimal=6)

    def test_gen_p(self):
        ak = AKtest(self.reg, self.w)
        np.testing.assert_array_almost_equal(ak.p, 0.031182360054340875, decimal=6)

    def test_sp_mi(self):
        ak = AKtest(self.reg, self.w, case='gen')
        np.testing.assert_array_almost_equal(ak.mi, 0.2232672865437263, decimal=6)

    def test_sp_ak(self):
        ak = AKtest(self.reg, self.w,case='gen')
        np.testing.assert_array_almost_equal(ak.ak, 1.1575928784397795, decimal=6)

    def test_sp_p(self):
        ak = AKtest(self.reg, self.w, case='gen')
        np.testing.assert_array_almost_equal(ak.p, 0.28196531619791054, decimal=6)

class TestSpDcache(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        X = np.array(X).T
        self.y = y
        self.X = X
        ols = OLS(self.y, self.X)
        self.ols = ols
        w = pysal.open(pysal.examples.get_path('columbus.gal'), 'r').read()
        w.transform='r'
        self.w = w

    def test_j(self):
        cache = spDcache(self.ols, self.w)
        np.testing.assert_array_almost_equal(cache.j[0][0], 0.62330311259039439, decimal=6)

    def test_t(self):
        cache = spDcache(self.ols, self.w)
        np.testing.assert_array_almost_equal(cache.t, 22.751186696900984, decimal=6)

    def test_trA(self):
        cache = spDcache(self.ols, self.w)
        np.testing.assert_array_almost_equal(cache.trA, 1.5880426389276328, decimal=6)

    def test_utwuDs(self):
        cache = spDcache(self.ols, self.w)
        np.testing.assert_array_almost_equal(cache.utwuDs[0][0], 8.3941977502916068, decimal=6)

    def test_utwyDs(self):
        cache = spDcache(self.ols, self.w)
        np.testing.assert_array_almost_equal(cache.utwyDs[0][0], 5.475255215067957, decimal=6)

    def test_wu(self):
        cache = spDcache(self.ols, self.w)
        np.testing.assert_array_almost_equal(cache.wu[0][0], -10.681344941514411, decimal=6)


if __name__ == '__main__':
    unittest.main()
