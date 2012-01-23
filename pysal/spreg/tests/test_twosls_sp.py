import unittest
import numpy as np
import pysal
import pysal.spreg.diagnostics as D

class TestBaseGMLag(unittest.TestCase):
    def setUp(self):
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
        self.db = pysal.open(pysal.examples.get_path("columbus.dbf"), 'r')
        y = np.array(self.db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        
    def test___init__(self):
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        base_gm_lag = pysal.spreg.twosls_sp.BaseGM_Lag(self.y, self.X, w=self.w, w_lags=2)
        tbetas = np.array([[  4.53017056e+01], [  6.20888617e-01], [ -4.80723451e-01], [  2.83622122e-02]])
        np.testing.assert_array_almost_equal(base_gm_lag.betas, tbetas) 

        dbetas = D.se_betas(base_gm_lag)
        se_betas = np.array([ 17.91278862, 0.52486082, 0.1822815, 0.31740089 ])
        np.testing.assert_array_almost_equal(dbetas, se_betas)

    def test_init_white_(self):
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        base_gm_lag = pysal.spreg.twosls_sp.BaseGM_Lag(self.y, self.X, w=self.w, w_lags=2, robust='white')
        tbetas = np.array([[  4.53017056e+01], [  6.20888617e-01], [ -4.80723451e-01], [  2.83622122e-02]])
        np.testing.assert_array_almost_equal(base_gm_lag.betas, tbetas) 

        dbetas = D.se_betas(base_gm_lag)
        se_betas = np.array([ 20.47077481, 0.50613931, 0.20138425, 0.38028295 ])
        np.testing.assert_array_almost_equal(dbetas, se_betas)


    def test_init_discbd(self):
        X = np.array(self.db.by_col("INC"))
        X = np.reshape(X, (49,1))
        yd = np.array(self.db.by_col("CRIME"))
        yd = np.reshape(yd, (49,1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49,1))
        reg = pysal.spreg.twosls_sp.BaseGM_Lag(self.y, X, w=self.w, yend=yd, q=q, w_lags=2)
        tbetas = np.array([[ 100.79359082], [  -0.50215501], [  -1.14881711], [  -0.38235022]])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)

        dbetas = D.se_betas(reg)
        se_betas = np.array([ 53.0829123 ,   1.02511494,   0.57589064,   0.59891744 ])
        np.testing.assert_array_almost_equal(dbetas, se_betas)

        


        

class TestGMLag(unittest.TestCase):
    def test___init__(self):
        # g_m__lag = GM_Lag(y, x, yend, q, w, w_lags, lag_q, robust, gwk, sig2n_k, spat_diag, vm, name_y, name_x, name_yend, name_q, name_w, name_gwk, name_ds)
        #assert False # TODO: implement your test here
        pass

if __name__ == '__main__':
    unittest.main()
