import unittest
import numpy as np
import pysal
from pysal.spreg import diagnostics
from pysal.spreg.ols import BaseOLS as OLS
from pysal.spreg.diagnostics_sp import LMtests, MoranRes, spDcache, spDcache

class Test_diagnostics_sp(unittest.TestCase):
    """ setUp is called before each test function execution """
    def setUp(self):
        db=pysal.open("pysal/examples/columbus.dbf","r")
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
        w = pysal.open('pysal/examples/columbus.gal', 'r').read()
        w.transform='r'
        self.w = w

    def test_LMtests(self):
        lms = LMtests(self.ols, self.w)
        lme = np.array([3.097094,  0.078432])
        np.testing.assert_array_almost_equal(lms.lme, lme, decimal=6)
        lml = np.array([ 0.981552,  0.321816])
        np.testing.assert_array_almost_equal(lms.lml, lml, decimal=6)
        rlme = np.array([ 3.209187,  0.073226])
        np.testing.assert_array_almost_equal(lms.rlme, rlme, decimal=6)
        rlml = np.array([ 1.093645,  0.295665])
        np.testing.assert_array_almost_equal(lms.rlml, rlml, decimal=6)
        sarma = np.array([ 4.190739,  0.123025])
        np.testing.assert_array_almost_equal(lms.sarma, sarma, decimal=6)
 
    def test_MoranRes(self):
        m = MoranRes(self.ols, self.w, z=True)
        np.testing.assert_array_almost_equal(m.I, 0.17130999999999999, decimal=6)
        np.testing.assert_array_almost_equal(m.eI, -0.034522999999999998, decimal=6)
        np.testing.assert_array_almost_equal(m.vI, 0.0081300000000000001, decimal=6)
        np.testing.assert_array_almost_equal(m.zI, 2.2827389999999999, decimal=6)

    def test_spDcache(self):
        cache = spDcache(self.ols, self.w)
        np.testing.assert_array_almost_equal(cache.j[0][0], 0.62330311259039439, decimal=6)
        np.testing.assert_array_almost_equal(cache.wu[0][0], -10.681344941514411, decimal=6)
        np.testing.assert_array_almost_equal(cache.utwuDs[0][0], 8.3941977502916068, decimal=6)
        np.testing.assert_array_almost_equal(cache.utwyDs[0][0], 5.475255215067957, decimal=6)
        np.testing.assert_array_almost_equal(cache.t, 22.751186696900984, decimal=6)
        np.testing.assert_array_almost_equal(cache.trA, 1.5880426389276328, decimal=6)

suite = unittest.TestLoader().loadTestsFromTestCase(Test_diagnostics_sp)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
