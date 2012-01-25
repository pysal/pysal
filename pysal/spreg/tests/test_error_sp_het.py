import unittest
import pysal
import numpy as np
from pysal.spreg import error_sp_het as HET

class TestBaseGMErrorHet(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = HET.BaseGM_Error_Het(self.y, self.X, self.w, step1c=True)
        betas = np.array([[ 47.99626638], [  0.71048989], [ -0.55876126], [  0.41178776]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        u = np.array([ 27.38122697])
        np.testing.assert_array_almost_equal(reg.u[0],u,7)
        predy = np.array([ 53.08577603])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,7)
        n = 49
        self.assertEqual(reg.n,n,7)
        k = 3
        self.assertEqual(reg.k,k,7)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,7)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        self.assertEqual(reg.iteration,its,7)
        my = 38.436224469387746
        self.assertEqual(reg.mean_y,my,7)
        stdy = 18.466069465206047
        self.assertEqual(reg.std_y,stdy,7)
        vm = np.array([[  1.31767529e+02,  -3.58368748e+00,  -1.65090647e+00,
              0.00000000e+00],
           [ -3.58368748e+00,   1.35513711e-01,   3.77539055e-02,
              0.00000000e+00],
           [ -1.65090647e+00,   3.77539055e-02,   2.61042702e-02,
              0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              2.82398517e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02,   1.72131237e+03],
           [  7.04371999e+02,   1.16866734e+04,   2.15575320e+04],
           [  1.72131237e+03,   2.15575320e+04,   7.39058986e+04]])
        np.testing.assert_array_almost_equal(reg.xtx,xtx,4)
             
class TestGMErrorHet(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = HET.GM_Error_Het(self.y, self.X, self.w, step1c=True)
        betas = np.array([[ 47.99626638], [  0.71048989], [ -0.55876126], [  0.41178776]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        u = np.array([ 27.38122697])
        np.testing.assert_array_almost_equal(reg.u[0],u,7)
        predy = np.array([ 53.08577603])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,7)
        n = 49
        self.assertEqual(reg.n,n,7)
        k = 3
        self.assertEqual(reg.k,k,7)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,7)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        self.assertEqual(reg.iteration,its,7)
        my = 38.436224469387746
        self.assertEqual(reg.mean_y,my,7)
        stdy = 18.466069465206047
        self.assertEqual(reg.std_y,stdy,7)
        vm = np.array([[  1.31767529e+02,  -3.58368748e+00,  -1.65090647e+00,
              0.00000000e+00],
           [ -3.58368748e+00,   1.35513711e-01,   3.77539055e-02,
              0.00000000e+00],
           [ -1.65090647e+00,   3.77539055e-02,   2.61042702e-02,
              0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              2.82398517e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02,   1.72131237e+03],
           [  7.04371999e+02,   1.16866734e+04,   2.15575320e+04],
           [  1.72131237e+03,   2.15575320e+04,   7.39058986e+04]])
        np.testing.assert_array_almost_equal(reg.xtx,xtx,4)

class TestBaseGMEndogErrorHet(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        yd = []
        yd.append(db.by_col("CRIME"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = HET.BaseGM_Endog_Error_Het(self.y, self.X, self.yd, self.q, self.w, step1c=True)
        betas = np.array([[ 55.39707924], [  0.46563046], [ -0.67038326], [  0.41135023]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        u = np.array([ 26.51812895])
        np.testing.assert_array_almost_equal(reg.u[0],u,7)
        predy = np.array([ 53.94887405])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,7)
        n = 49
        self.assertEqual(reg.n,n,7)
        k = 3
        self.assertEqual(reg.k,k,7)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,7)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        yend = np.array([ 15.72598])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,7)
        q = np.array([ 5.03])
        np.testing.assert_array_almost_equal(reg.q[0],q,7)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.z[0],z,7)
        h = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_array_almost_equal(reg.h[0],h,7)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        self.assertEqual(reg.iteration,its,7)
        my = 38.436224469387746
        self.assertEqual(reg.mean_y,my,7)
        stdy = 18.466069465206047
        self.assertEqual(reg.std_y,stdy,7)
        vm = np.array([[  8.34637805e+02,  -2.16932259e+01,  -1.33327894e+01,
                  1.65840848e+00],
               [ -2.16932259e+01,   5.97683070e-01,   3.39503523e-01,
                 -3.90111107e-02],
               [ -1.33327894e+01,   3.39503523e-01,   2.19008080e-01,
                 -2.81929695e-02],
               [  1.65840848e+00,  -3.90111107e-02,  -2.81929695e-02,
                  3.15686105e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        hth = np.array([[    49.        ,    704.371999  ,    139.75      ],
               [   704.371999  ,  11686.67338121,   2246.12800625],
               [   139.75      ,   2246.12800625,    498.5851    ]])
        np.testing.assert_array_almost_equal(reg.hth,hth,6)
        
class TestGMEndogErrorHet(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        yd = []
        yd.append(db.by_col("CRIME"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = HET.GM_Endog_Error_Het(self.y, self.X, self.yd, self.q, self.w, step1c=True)
        betas = np.array([[ 55.39707924], [  0.46563046], [ -0.67038326], [  0.41135023]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        u = np.array([ 26.51812895])
        np.testing.assert_array_almost_equal(reg.u[0],u,7)
        predy = np.array([ 53.94887405])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,7)
        n = 49
        self.assertEqual(reg.n,n,7)
        k = 3
        self.assertEqual(reg.k,k,7)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,7)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        yend = np.array([ 15.72598])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,7)
        q = np.array([ 5.03])
        np.testing.assert_array_almost_equal(reg.q[0],q,7)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.z[0],z,7)
        h = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_array_almost_equal(reg.h[0],h,7)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        self.assertEqual(reg.iteration,its,7)
        my = 38.436224469387746
        self.assertEqual(reg.mean_y,my,7)
        stdy = 18.466069465206047
        self.assertEqual(reg.std_y,stdy,7)
        vm = np.array([[  8.34637805e+02,  -2.16932259e+01,  -1.33327894e+01,
                  1.65840848e+00],
               [ -2.16932259e+01,   5.97683070e-01,   3.39503523e-01,
                 -3.90111107e-02],
               [ -1.33327894e+01,   3.39503523e-01,   2.19008080e-01,
                 -2.81929695e-02],
               [  1.65840848e+00,  -3.90111107e-02,  -2.81929695e-02,
                  3.15686105e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        hth = np.array([[    49.        ,    704.371999  ,    139.75      ],
               [   704.371999  ,  11686.67338121,   2246.12800625],
               [   139.75      ,   2246.12800625,    498.5851    ]])
        np.testing.assert_array_almost_equal(reg.hth,hth,6)
 
class TestBaseGMComboHet(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        # Only spatial lag
        reg = HET.BaseGM_Combo_Het(self.y, self.X, w=self.w, step1c=True)
        betas = np.array([[ 57.7778574 ], [  0.73034922], [ -0.59257362], [ -0.2230231 ], [  0.56636724]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        u = np.array([ 25.65156033])
        np.testing.assert_array_almost_equal(reg.u[0],u,7)
        predy = np.array([ 54.81544267])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,7)
        n = 49
        self.assertEqual(reg.n,n,7)
        k = 4
        self.assertEqual(reg.k,k,7)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,7)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        yend = np.array([ 35.4585005])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,7)
        q = np.array([ 18.594    ,  24.7142675])
        np.testing.assert_array_almost_equal(reg.q[0],q,7)
        z = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005])
        np.testing.assert_array_almost_equal(reg.z[0],z,7)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        self.assertEqual(reg.iteration,its,7)
        my = 38.436224469387746
        self.assertEqual(reg.mean_y,my,7)
        stdy = 18.466069465206047
        self.assertEqual(reg.std_y,stdy,7)
        vm = np.array([[  4.86218274e+02,  -2.77268729e+00,  -1.59987770e+00,
             -1.01969471e+01,   2.74302006e+00],
           [ -2.77268729e+00,   1.04680972e-01,   2.51172238e-02,
              1.95136385e-03,   3.70052723e-03],
           [ -1.59987770e+00,   2.51172238e-02,   2.15655720e-02,
              7.65868344e-03,  -7.30173070e-03],
           [ -1.01969471e+01,   1.95136385e-03,   7.65868344e-03,
              2.78273684e-01,  -6.89402590e-02],
           [  2.74302006e+00,   3.70052723e-03,  -7.30173070e-03,
             -6.89402590e-02,   7.12034037e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        hth = np.array([[  4.90000000e+01,   7.04371999e+02,   1.72131237e+03,
              7.24743592e+02,   1.70735413e+03],
           [  7.04371999e+02,   1.16866734e+04,   2.15575320e+04,
              1.10925200e+04,   2.23848036e+04],
           [  1.72131237e+03,   2.15575320e+04,   7.39058986e+04,
              2.34796298e+04,   6.70145378e+04],
           [  7.24743592e+02,   1.10925200e+04,   2.34796298e+04,
              1.16146226e+04,   2.30304624e+04],
           [  1.70735413e+03,   2.23848036e+04,   6.70145378e+04,
              2.30304624e+04,   6.69879858e+04]])
        np.testing.assert_array_almost_equal(reg.hth,hth,4)

class TestGMComboHet(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        # Only spatial lag
        reg = HET.GM_Combo_Het(self.y, self.X, w=self.w, step1c=True)
        betas = np.array([[ 57.7778574 ], [  0.73034922], [ -0.59257362], [ -0.2230231 ], [  0.56636724]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        u = np.array([ 25.65156033])
        np.testing.assert_array_almost_equal(reg.u[0],u,7)
        predy = np.array([ 54.81544267])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,7)
        n = 49
        self.assertEqual(reg.n,n,7)
        k = 4
        self.assertEqual(reg.k,k,7)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,7)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        yend = np.array([ 35.4585005])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,7)
        q = np.array([ 18.594    ,  24.7142675])
        np.testing.assert_array_almost_equal(reg.q[0],q,7)
        z = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005])
        np.testing.assert_array_almost_equal(reg.z[0],z,7)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        self.assertEqual(reg.iteration,its,7)
        my = 38.436224469387746
        self.assertEqual(reg.mean_y,my,7)
        stdy = 18.466069465206047
        self.assertEqual(reg.std_y,stdy,7)
        vm = np.array([[  4.86218274e+02,  -2.77268729e+00,  -1.59987770e+00,
             -1.01969471e+01,   2.74302006e+00],
           [ -2.77268729e+00,   1.04680972e-01,   2.51172238e-02,
              1.95136385e-03,   3.70052723e-03],
           [ -1.59987770e+00,   2.51172238e-02,   2.15655720e-02,
              7.65868344e-03,  -7.30173070e-03],
           [ -1.01969471e+01,   1.95136385e-03,   7.65868344e-03,
              2.78273684e-01,  -6.89402590e-02],
           [  2.74302006e+00,   3.70052723e-03,  -7.30173070e-03,
             -6.89402590e-02,   7.12034037e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        hth = np.array([[  4.90000000e+01,   7.04371999e+02,   1.72131237e+03,
              7.24743592e+02,   1.70735413e+03],
           [  7.04371999e+02,   1.16866734e+04,   2.15575320e+04,
              1.10925200e+04,   2.23848036e+04],
           [  1.72131237e+03,   2.15575320e+04,   7.39058986e+04,
              2.34796298e+04,   6.70145378e+04],
           [  7.24743592e+02,   1.10925200e+04,   2.34796298e+04,
              1.16146226e+04,   2.30304624e+04],
           [  1.70735413e+03,   2.23848036e+04,   6.70145378e+04,
              2.30304624e+04,   6.69879858e+04]])
        np.testing.assert_array_almost_equal(reg.hth,hth,4)

if __name__ == '__main__':
    unittest.main()
