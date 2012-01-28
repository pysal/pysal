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
        reg = pysal.spreg.twosls_sp.BaseGM_Lag(self.y, self.X, w=self.w, w_lags=2)
        betas = np.array([[  4.53017056e+01], [  6.20888617e-01], [ -4.80723451e-01], [  2.83622122e-02]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        h_0 = np.array([  1.        ,  19.531     ,  15.72598   ,  18.594     ,
                            24.7142675 ,  13.72216667,  27.82929567])
        np.testing.assert_array_almost_equal(reg.h[0], h_0)
        hth = np.  array([   49.        ,   704.371999  ,  1721.312371  ,   724.7435916 ,
                             1707.35412945,   711.31248483,  1729.63201243])
        np.testing.assert_array_almost_equal(reg.hth[0], hth, 7)
        hthi = np.array([  7.33701328e+00,   2.27764882e-02,   2.18153588e-02,
                           -5.11035447e-02,   1.22515181e-03,  -2.38079378e-01,
                           -1.20149133e-01])
        np.testing.assert_array_almost_equal(reg.hthi[0], hthi, 7)
        self.assertEqual(reg.k, 4)
        self.assertEqual(reg.kstar, 1)
        self.assertAlmostEqual(reg.mean_y, 38.436224469387746, 7)
        self.assertEqual(reg.n, 49)
        pfora1a2 = np.array([ 80.5588479 ,  -1.06625281,  -0.61703759,  -1.10071931]) 
        np.testing.assert_array_almost_equal(reg.pfora1a2[0], pfora1a2, 7)
        predy_5 = np.array([[ 50.87411532],[ 50.76969931],[ 41.77223722],[ 33.44262382],[ 28.77418036]])
        np.testing.assert_array_almost_equal(reg.predy[0:5], predy_5, 7)
        q_5 = np.array([ 18.594     ,  24.7142675 ,  13.72216667,  27.82929567])
        np.testing.assert_array_almost_equal(reg.q[0], q_5)
        self.assertAlmostEqual(reg.sig2n_k, 234.54258763039289, 7)
        self.assertAlmostEqual(reg.sig2n, 215.39625394627919, 7)
        self.assertAlmostEqual(reg.sig2, 215.39625394627919, 7)
        self.assertAlmostEqual(reg.std_y, 18.466069465206047, 7)
        u_5 = np.array( [[ 29.59288768], [ -6.20269831], [-15.42223722], [ -0.24262282], [ -5.54918036]])
        np.testing.assert_array_almost_equal(reg.u[0:5], u_5, 7)
        self.assertAlmostEqual(reg.utu, 10554.41644336768, 7)
        varb = np.array( [[  1.48966377e+00, -2.28698061e-02, -1.20217386e-02, -1.85763498e-02],
                          [ -2.28698061e-02,  1.27893998e-03,  2.74600023e-04, -1.33497705e-04],
                          [ -1.20217386e-02,  2.74600023e-04,  1.54257766e-04,  6.86851184e-05],
                          [ -1.85763498e-02, -1.33497705e-04,  6.86851184e-05,  4.67711582e-04]])
        np.testing.assert_array_almost_equal(reg.varb, varb, 7)
        vm = np.array([[  3.20867996e+02, -4.92607057e+00, -2.58943746e+00, -4.00127615e+00],
                       [ -4.92607057e+00,  2.75478880e-01,  5.91478163e-02, -2.87549056e-02],
                       [ -2.58943746e+00,  5.91478163e-02,  3.32265449e-02,  1.47945172e-02],
                       [ -4.00127615e+00, -2.87549056e-02,  1.47945172e-02,  1.00743323e-01]])
        np.testing.assert_array_almost_equal(reg.vm, vm, 6)
        x_0 = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0], x_0, 7)
        y_5 = np.array( [[ 80.467003], [ 44.567001], [ 26.35    ], [ 33.200001], [ 23.225   ]])
        np.testing.assert_array_almost_equal(reg.y[0:5], y_5, 7)
        yend_5 = np.array( [[ 35.4585005 ], [ 46.67233467], [ 45.36475125], [ 32.81675025], [ 30.81785714]])
        np.testing.assert_array_almost_equal(reg.yend[0:5], yend_5, 7)
        z_0 = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005]) 
        np.testing.assert_array_almost_equal(reg.z[0], z_0, 7)
        zthhthi = np.array( [[  1.00000000e+00, -2.22044605e-16, -2.22044605e-16 , 2.22044605e-16,
                                4.44089210e-16,  0.00000000e+00, -8.88178420e-16],
                             [  0.00000000e+00,  1.00000000e+00, -3.55271368e-15 , 3.55271368e-15,
                               -7.10542736e-15,  7.10542736e-14,  0.00000000e+00],
                             [  1.81898940e-12,  2.84217094e-14,  1.00000000e+00 , 0.00000000e+00,
                               -2.84217094e-14,  5.68434189e-14,  5.68434189e-14],
                             [ -8.31133940e+00, -3.76104678e-01, -2.07028208e-01 , 1.32618931e+00,
                               -8.04284562e-01,  1.30527047e+00,  1.39136816e+00]])
        np.testing.assert_array_almost_equal(reg.zthhthi, zthhthi, 7)

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

    def test_init_hac_(self):
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        gwk = pysal.kernelW_from_shapefile(pysal.examples.get_path('columbus.shp'),k=15,function='triangular', fixed=False)        
        base_gm_lag = pysal.spreg.twosls_sp.BaseGM_Lag(self.y, self.X, w=self.w, w_lags=2, robust='hac', gwk=gwk)
        tbetas = np.array([[  4.53017056e+01], [  6.20888617e-01], [ -4.80723451e-01], [  2.83622122e-02]])
        np.testing.assert_array_almost_equal(base_gm_lag.betas, tbetas) 
        dbetas = D.se_betas(base_gm_lag)
        se_betas = np.array([ 19.08513569,   0.51769543,   0.18244862,   0.35460553])
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

    def test_n_k(self):
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        reg = pysal.spreg.twosls_sp.BaseGM_Lag(self.y, self.X, w=self.w, w_lags=2, sig2n_k=True)
        betas = np.  array([[  4.53017056e+01], [  6.20888617e-01], [ -4.80723451e-01], [  2.83622122e-02]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        vm = np.array( [[  3.49389596e+02, -5.36394351e+00, -2.81960968e+00, -4.35694515e+00],
                         [ -5.36394351e+00,  2.99965892e-01,  6.44054000e-02, -3.13108972e-02],
                         [ -2.81960968e+00,  6.44054000e-02,  3.61800155e-02,  1.61095854e-02],
                         [ -4.35694515e+00, -3.13108972e-02,  1.61095854e-02,  1.09698285e-01]])
        np.testing.assert_array_almost_equal(reg.vm, vm, 7)

    def test_lag_q(self):
        X = np.array(self.db.by_col("INC"))
        X = np.reshape(X, (49,1))
        yd = np.array(self.db.by_col("CRIME"))
        yd = np.reshape(yd, (49,1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49,1))
        reg = pysal.spreg.twosls_sp.BaseGM_Lag(self.y, X, w=self.w, yend=yd, q=q, w_lags=2, lag_q=False)
        tbetas = np.array( [[ 108.83261383], [  -0.48041099], [  -1.18950006], [  -0.56140186]])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)
        dbetas = D.se_betas(reg)
        se_betas = np.array([ 58.33203837,   1.09100446,   0.62315167,   0.68088777])
        np.testing.assert_array_almost_equal(dbetas, se_betas)



class TestGMLag(unittest.TestCase):
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
        reg = pysal.spreg.twosls_sp.GM_Lag(self.y, self.X, w=self.w, w_lags=2)
        betas = np.array([[  4.53017056e+01], [  6.20888617e-01], [ -4.80723451e-01], [  2.83622122e-02]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        h_0 = np.array([  1.        ,  19.531     ,  15.72598   ,  18.594     ,
                            24.7142675 ,  13.72216667,  27.82929567])
        np.testing.assert_array_almost_equal(reg.h[0], h_0)
        hth = np.  array([   49.        ,   704.371999  ,  1721.312371  ,   724.7435916 ,
                             1707.35412945,   711.31248483,  1729.63201243])
        np.testing.assert_array_almost_equal(reg.hth[0], hth, 7)
        hthi = np.array([  7.33701328e+00,   2.27764882e-02,   2.18153588e-02,
                           -5.11035447e-02,   1.22515181e-03,  -2.38079378e-01,
                           -1.20149133e-01])
        np.testing.assert_array_almost_equal(reg.hthi[0], hthi, 7)
        self.assertEqual(reg.k, 4)
        self.assertEqual(reg.kstar, 1)
        self.assertAlmostEqual(reg.mean_y, 38.436224469387746, 7)
        self.assertEqual(reg.n, 49)
        pfora1a2 = np.array([ 80.5588479 ,  -1.06625281,  -0.61703759,  -1.10071931]) 
        np.testing.assert_array_almost_equal(reg.pfora1a2[0], pfora1a2, 7)
        predy_5 = np.array([[ 50.87411532],[ 50.76969931],[ 41.77223722],[ 33.44262382],[ 28.77418036]])
        np.testing.assert_array_almost_equal(reg.predy[0:5], predy_5, 7)
        q_5 = np.array([ 18.594     ,  24.7142675 ,  13.72216667,  27.82929567])
        np.testing.assert_array_almost_equal(reg.q[0], q_5)
        self.assertAlmostEqual(reg.sig2n_k, 234.54258763039289, 7)
        self.assertAlmostEqual(reg.sig2n, 215.39625394627919, 7)
        self.assertAlmostEqual(reg.sig2, 215.39625394627919, 7)
        self.assertAlmostEqual(reg.std_y, 18.466069465206047, 7)
        u_5 = np.array( [[ 29.59288768], [ -6.20269831], [-15.42223722], [ -0.24262282], [ -5.54918036]])
        np.testing.assert_array_almost_equal(reg.u[0:5], u_5, 7)
        self.assertAlmostEqual(reg.utu, 10554.41644336768, 7)
        varb = np.array( [[  1.48966377e+00, -2.28698061e-02, -1.20217386e-02, -1.85763498e-02],
                          [ -2.28698061e-02,  1.27893998e-03,  2.74600023e-04, -1.33497705e-04],
                          [ -1.20217386e-02,  2.74600023e-04,  1.54257766e-04,  6.86851184e-05],
                          [ -1.85763498e-02, -1.33497705e-04,  6.86851184e-05,  4.67711582e-04]])
        np.testing.assert_array_almost_equal(reg.varb, varb, 7)
        vm = np.array([[  3.20867996e+02, -4.92607057e+00, -2.58943746e+00, -4.00127615e+00],
                       [ -4.92607057e+00,  2.75478880e-01,  5.91478163e-02, -2.87549056e-02],
                       [ -2.58943746e+00,  5.91478163e-02,  3.32265449e-02,  1.47945172e-02],
                       [ -4.00127615e+00, -2.87549056e-02,  1.47945172e-02,  1.00743323e-01]])
        np.testing.assert_array_almost_equal(reg.vm, vm, 6)
        x_0 = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0], x_0, 7)
        y_5 = np.array( [[ 80.467003], [ 44.567001], [ 26.35    ], [ 33.200001], [ 23.225   ]])
        np.testing.assert_array_almost_equal(reg.y[0:5], y_5, 7)
        yend_5 = np.array( [[ 35.4585005 ], [ 46.67233467], [ 45.36475125], [ 32.81675025], [ 30.81785714]])
        np.testing.assert_array_almost_equal(reg.yend[0:5], yend_5, 7)
        z_0 = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005]) 
        np.testing.assert_array_almost_equal(reg.z[0], z_0, 7)
        zthhthi = np.array( [[  1.00000000e+00, -2.22044605e-16, -2.22044605e-16 , 2.22044605e-16,
                                4.44089210e-16,  0.00000000e+00, -8.88178420e-16],
                             [  0.00000000e+00,  1.00000000e+00, -3.55271368e-15 , 3.55271368e-15,
                               -7.10542736e-15,  7.10542736e-14,  0.00000000e+00],
                             [  1.81898940e-12,  2.84217094e-14,  1.00000000e+00 , 0.00000000e+00,
                               -2.84217094e-14,  5.68434189e-14,  5.68434189e-14],
                             [ -8.31133940e+00, -3.76104678e-01, -2.07028208e-01 , 1.32618931e+00,
                               -8.04284562e-01,  1.30527047e+00,  1.39136816e+00]])
        np.testing.assert_array_almost_equal(reg.zthhthi, zthhthi, 7)

    def test_init_white_(self):
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        base_gm_lag = pysal.spreg.twosls_sp.GM_Lag(self.y, self.X, w=self.w, w_lags=2, robust='white')
        tbetas = np.array([[  4.53017056e+01], [  6.20888617e-01], [ -4.80723451e-01], [  2.83622122e-02]])
        np.testing.assert_array_almost_equal(base_gm_lag.betas, tbetas) 
        dbetas = D.se_betas(base_gm_lag)
        se_betas = np.array([ 20.47077481, 0.50613931, 0.20138425, 0.38028295 ])
        np.testing.assert_array_almost_equal(dbetas, se_betas)

    def test_init_hac_(self):
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        gwk = pysal.kernelW_from_shapefile(pysal.examples.get_path('columbus.shp'),k=15,function='triangular', fixed=False)        
        base_gm_lag = pysal.spreg.twosls_sp.GM_Lag(self.y, self.X, w=self.w, w_lags=2, robust='hac', gwk=gwk)
        tbetas = np.array([[  4.53017056e+01], [  6.20888617e-01], [ -4.80723451e-01], [  2.83622122e-02]])
        np.testing.assert_array_almost_equal(base_gm_lag.betas, tbetas) 
        dbetas = D.se_betas(base_gm_lag)
        se_betas = np.array([ 19.08513569,   0.51769543,   0.18244862,   0.35460553])
        np.testing.assert_array_almost_equal(dbetas, se_betas)

    def test_init_discbd(self):
        X = np.array(self.db.by_col("INC"))
        X = np.reshape(X, (49,1))
        yd = np.array(self.db.by_col("CRIME"))
        yd = np.reshape(yd, (49,1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49,1))
        reg = pysal.spreg.twosls_sp.GM_Lag(self.y, X, w=self.w, yend=yd, q=q, w_lags=2)
        tbetas = np.array([[ 100.79359082], [  -0.50215501], [  -1.14881711], [  -0.38235022]])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)
        dbetas = D.se_betas(reg)
        se_betas = np.array([ 53.0829123 ,   1.02511494,   0.57589064,   0.59891744 ])
        np.testing.assert_array_almost_equal(dbetas, se_betas)

    def test_n_k(self):
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        reg = pysal.spreg.twosls_sp.GM_Lag(self.y, self.X, w=self.w, w_lags=2, sig2n_k=True)
        betas = np.  array([[  4.53017056e+01], [  6.20888617e-01], [ -4.80723451e-01], [  2.83622122e-02]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        vm = np.array( [[  3.49389596e+02, -5.36394351e+00, -2.81960968e+00, -4.35694515e+00],
                         [ -5.36394351e+00,  2.99965892e-01,  6.44054000e-02, -3.13108972e-02],
                         [ -2.81960968e+00,  6.44054000e-02,  3.61800155e-02,  1.61095854e-02],
                         [ -4.35694515e+00, -3.13108972e-02,  1.61095854e-02,  1.09698285e-01]])
        np.testing.assert_array_almost_equal(reg.vm, vm, 7)

    def test_lag_q(self):
        X = np.array(self.db.by_col("INC"))
        X = np.reshape(X, (49,1))
        yd = np.array(self.db.by_col("CRIME"))
        yd = np.reshape(yd, (49,1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49,1))
        reg = pysal.spreg.twosls_sp.GM_Lag(self.y, X, w=self.w, yend=yd, q=q, w_lags=2, lag_q=False)
        tbetas = np.array( [[ 108.83261383], [  -0.48041099], [  -1.18950006], [  -0.56140186]])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)
        dbetas = D.se_betas(reg)
        se_betas = np.array([ 58.33203837,   1.09100446,   0.62315167,   0.68088777])
        np.testing.assert_array_almost_equal(dbetas, se_betas)




if __name__ == '__main__':
    unittest.main()
