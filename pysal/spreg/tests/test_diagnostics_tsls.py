import unittest
import numpy as np
import pysal
import pysal.spreg.diagnostics_tsls as diagnostics_tsls
import pysal.spreg.diagnostics as diagnostics
from pysal.spreg.ols import OLS as OLS
from pysal.spreg.twosls import TSLS as TSLS
from pysal.spreg.twosls_sp import GM_Lag
from scipy.stats import pearsonr
from pysal.common import RTOL

# create regression object used by the apatial tests
db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
y = np.array(db.by_col("CRIME"))
y = np.reshape(y, (49,1))
X = []
X.append(db.by_col("INC"))
X = np.array(X).T    
yd = []
yd.append(db.by_col("HOVAL"))
yd = np.array(yd).T
q = []
q.append(db.by_col("DISCBD"))
q = np.array(q).T
reg = TSLS(y, X, yd, q)

# create regression object for spatial test
db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
y = np.array(db.by_col("HOVAL"))
y = np.reshape(y, (49,1))
X = np.array(db.by_col("INC"))
X = np.reshape(X, (49,1))
yd = np.array(db.by_col("CRIME"))
yd = np.reshape(yd, (49,1))
q = np.array(db.by_col("DISCBD"))
q = np.reshape(q, (49,1))
w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp")) 
w.transform = 'r'
regsp = GM_Lag(y, X, w=w, yend=yd, q=q, w_lags=2)


class TestTStat(unittest.TestCase):
    def test_t_stat(self):
        obs = diagnostics_tsls.t_stat(reg)
        exp = [(5.8452644704588588, 4.9369075950019865e-07),
               (0.36760156683572748, 0.71485634049075841),
               (-1.9946891307832111, 0.052021795864651159)]
        np.testing.assert_allclose(obs, exp, RTOL)

class TestPr2Aspatial(unittest.TestCase):
    def test_pr2_aspatial(self):
        obs = diagnostics_tsls.pr2_aspatial(reg)
        exp = 0.2793613712817381
        np.testing.assert_allclose(obs,exp, RTOL)

class TestPr2Spatial(unittest.TestCase):
    def test_pr2_spatial(self):
        obs = diagnostics_tsls.pr2_spatial(regsp)
        exp = 0.29964855438065163
        np.testing.assert_allclose(obs,exp, RTOL)


if __name__ == '__main__':
    unittest.main()
