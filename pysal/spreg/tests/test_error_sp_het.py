import unittest
import pysal
import numpy as np
from pysal.spreg import error_sp_het as HET
from pysal.common import RTOL

class TestBaseGMErrorHet(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = HET.BaseGM_Error_Het(self.y, self.X, self.w.sparse, step1c=True)
        betas = np.array([[ 47.99626638], [  0.71048989], [ -0.55876126], [  0.41178776]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 27.38122697])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        ef = np.array([ 32.29765975])
        np.testing.assert_allclose(reg.e_filtered[0],ef,RTOL)
        predy = np.array([ 53.08577603])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        k = 3
        np.testing.assert_allclose(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        np.testing.assert_allclose(reg.iteration,its,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        stdy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,stdy)
        vm = np.array([[  1.31767529e+02,  -3.58368748e+00,  -1.65090647e+00,
              0.00000000e+00],
           [ -3.58368748e+00,   1.35513711e-01,   3.77539055e-02,
              0.00000000e+00],
           [ -1.65090647e+00,   3.77539055e-02,   2.61042702e-02,
              0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              2.82398517e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02,   1.72131237e+03],
           [  7.04371999e+02,   1.16866734e+04,   2.15575320e+04],
           [  1.72131237e+03,   2.15575320e+04,   7.39058986e+04]])
        np.testing.assert_allclose(reg.xtx,xtx,RTOL)
             
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
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 27.38122697])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        ef = np.array([ 32.29765975])
        np.testing.assert_allclose(reg.e_filtered[0],ef,RTOL)
        predy = np.array([ 53.08577603])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        k = 3
        np.testing.assert_allclose(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        np.testing.assert_allclose(reg.iteration,its,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        stdy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,stdy)
        vm = np.array([[  1.31767529e+02,  -3.58368748e+00,  -1.65090647e+00,
              0.00000000e+00],
           [ -3.58368748e+00,   1.35513711e-01,   3.77539055e-02,
              0.00000000e+00],
           [ -1.65090647e+00,   3.77539055e-02,   2.61042702e-02,
              0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              2.82398517e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        pr2 = 0.34951013222581306
        np.testing.assert_allclose(reg.pr2,pr2)
        stde = np.array([ 11.47900385,   0.36812187,   0.16156816,   0.16804717])
        np.testing.assert_allclose(reg.std_err,stde,RTOL)
        z_stat = np.array([[  4.18122226e+00,   2.89946274e-05],
           [  1.93003988e+00,   5.36018970e-02],
           [ -3.45836247e+00,   5.43469673e-04],
           [  2.45042960e+00,   1.42685863e-02]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02,   1.72131237e+03],
           [  7.04371999e+02,   1.16866734e+04,   2.15575320e+04],
           [  1.72131237e+03,   2.15575320e+04,   7.39058986e+04]])
        np.testing.assert_allclose(reg.xtx,xtx,RTOL)

class TestBaseGMEndogErrorHet(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        yd = []
        yd.append(db.by_col("CRIME"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = HET.BaseGM_Endog_Error_Het(self.y, self.X, self.yd, self.q, self.w.sparse, step1c=True)
        betas = np.array([[ 55.39707924], [  0.46563046], [ -0.67038326], [  0.41135023]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 26.51812895])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        ef = np.array([ 31.46604707])
        np.testing.assert_allclose(reg.e_filtered[0],ef,RTOL)
        predy = np.array([ 53.94887405])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        k = 3
        np.testing.assert_allclose(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([ 15.72598])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        q = np.array([ 5.03])
        np.testing.assert_allclose(reg.q[0],q,RTOL)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        h = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_allclose(reg.h[0],h,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        np.testing.assert_allclose(reg.iteration,its,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        stdy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,stdy)
        vm = np.array([[  8.34637805e+02,  -2.16932259e+01,  -1.33327894e+01,
                  1.65840848e+00],
               [ -2.16932259e+01,   5.97683070e-01,   3.39503523e-01,
                 -3.90111107e-02],
               [ -1.33327894e+01,   3.39503523e-01,   2.19008080e-01,
                 -2.81929695e-02],
               [  1.65840848e+00,  -3.90111107e-02,  -2.81929695e-02,
                  3.15686105e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        hth = np.array([[    49.        ,    704.371999  ,    139.75      ],
               [   704.371999  ,  11686.67338121,   2246.12800625],
               [   139.75      ,   2246.12800625,    498.5851    ]])
        np.testing.assert_allclose(reg.hth,hth,RTOL)
        
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
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 26.51812895])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 53.94887405])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        k = 3
        np.testing.assert_allclose(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([ 15.72598])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        q = np.array([ 5.03])
        np.testing.assert_allclose(reg.q[0],q,RTOL)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        h = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_allclose(reg.h[0],h,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        np.testing.assert_allclose(reg.iteration,its,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        stdy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,stdy)
        vm = np.array([[  8.34637805e+02,  -2.16932259e+01,  -1.33327894e+01,
                  1.65840848e+00],
               [ -2.16932259e+01,   5.97683070e-01,   3.39503523e-01,
                 -3.90111107e-02],
               [ -1.33327894e+01,   3.39503523e-01,   2.19008080e-01,
                 -2.81929695e-02],
               [  1.65840848e+00,  -3.90111107e-02,  -2.81929695e-02,
                  3.15686105e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        pr2 = 0.34648011338954804
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        std_err = np.array([ 28.89009873,  0.77309965,  0.46798299,
            0.17767558])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([(1.9175109006819244, 0.055173057472126787), (0.60229035155742305, 0.54698088217644414), (-1.4324949211864271, 0.15200223057569454), (2.3151759776869496, 0.020603303355572443)])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)
        hth = np.array([[    49.        ,    704.371999  ,    139.75      ],
               [   704.371999  ,  11686.67338121,   2246.12800625],
               [   139.75      ,   2246.12800625,    498.5851    ]])
        np.testing.assert_allclose(reg.hth,hth,RTOL)
 
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
        yd2, q2 = pysal.spreg.utils.set_endog(self.y, self.X, self.w, None, None, 1, True)
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        reg = HET.BaseGM_Combo_Het(self.y, self.X, yend=yd2, q=q2, w=self.w.sparse, step1c=True)
        betas = np.array([[ 57.7778574 ], [  0.73034922], [ -0.59257362], [ -0.2230231 ], [  0.56636724]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 25.65156033])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        ef = np.array([ 31.87664403])
        np.testing.assert_allclose(reg.e_filtered[0],ef,RTOL)
        predy = np.array([ 54.81544267])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        k = 4
        np.testing.assert_allclose(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([ 35.4585005])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        q = np.array([ 18.594    ,  24.7142675])
        np.testing.assert_allclose(reg.q[0],q,RTOL)
        z = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        np.testing.assert_allclose(reg.iteration,its,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        stdy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,stdy,RTOL)
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
        np.testing.assert_allclose(reg.vm,vm,RTOL)
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
        np.testing.assert_allclose(reg.hth,hth,RTOL)

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
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 25.65156033])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        ef = np.array([ 31.87664403])
        np.testing.assert_allclose(reg.e_filtered[0],ef,RTOL)
        ep = np.array([ 28.30648145])
        np.testing.assert_allclose(reg.e_pred[0],ep,RTOL)
        pe = np.array([ 52.16052155])
        np.testing.assert_allclose(reg.predy_e[0],pe,RTOL)
        predy = np.array([ 54.81544267])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        k = 4
        np.testing.assert_allclose(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([ 35.4585005])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        q = np.array([ 18.594    ,  24.7142675])
        np.testing.assert_allclose(reg.q[0],q,RTOL)
        z = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        np.testing.assert_allclose(reg.iteration,its,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        stdy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,stdy)
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
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        pr2 = 0.3001582877472412
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        pr2_e = 0.35613102283621967
        np.testing.assert_allclose(reg.pr2_e,pr2_e,RTOL)
        std_err = np.array([ 22.05035768,  0.32354439,  0.14685221,  0.52751653,  0.26683966])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([(2.6202684885795335, 0.00878605635338265), (2.2573385444145524, 0.023986928627746887), (-4.0351698589183433, 5.456281036278686e-05), (-0.42277935292121521, 0.67245625315942159), (2.1225002455741895, 0.033795752094112265)])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)
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
        np.testing.assert_allclose(reg.hth,hth,RTOL)

if __name__ == '__main__':
    unittest.main()
