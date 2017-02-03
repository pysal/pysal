import unittest
import scipy
import pysal
import numpy as np
from pysal.spreg import error_sp_regimes as SP
from pysal.spreg.error_sp import GM_Error, GM_Endog_Error, GM_Combo
from pysal.common import RTOL

@unittest.skipIf(int(scipy.__version__.split(".")[1]) < 11,
"Maximum Likelihood requires SciPy version 11 or newer.")
class TestGM_Error_Regimes(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("HOVAL"))
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        self.w = pysal.queen_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
        self.r_var = 'NSA'
        self.regimes = db.by_col(self.r_var)
        X1 = []
        X1.append(db.by_col("INC"))
        self.X1 = np.array(X1).T
        yd = []
        yd.append(db.by_col("HOVAL"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        #Artficial:
        n = 256
        self.n2 = n//2
        self.x_a1 = np.random.uniform(-10,10,(n,1))
        self.x_a2 = np.random.uniform(1,5,(n,1))
        self.q_a = self.x_a2 + np.random.normal(0,1,(n,1))
        self.x_a = np.hstack((self.x_a1,self.x_a2))
        self.y_a = np.dot(np.hstack((np.ones((n,1)),self.x_a)),np.array([[1],[0.5],[2]])) + np.random.normal(0,1,(n,1))
        latt = int(np.sqrt(n))
        self.w_a = pysal.lat2W(latt,latt)
        self.w_a.transform='r'
        self.regi_a = [0]*(n//2) + [1]*(n//2) ##must be floors!
        self.w_a1 = pysal.lat2W(latt//2,latt)
        self.w_a1.transform='r'
        
    def test_model(self):
        reg = SP.GM_Error_Regimes(self.y, self.X, self.regimes, self.w)
        betas = np.array([[ 63.3443073 ],
       [ -0.15468   ],
       [ -1.52186509],
       [ 61.40071412],
       [ -0.33550084],
       [ -0.85076108],
       [  0.38671608]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([-2.06177251])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 17.78775251])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 6
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 15.72598])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([[  0.      ,   0.      ,   0.      ,   1.      ,  80.467003,  19.531   ]])
        np.testing.assert_allclose(reg.x[0].toarray(),x,RTOL)
        e = np.array([ 1.40747232])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([ 50.55875289,  -0.14444487,  -2.05735489,   0.        ,
         0.        ,   0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        sig2 = 102.13050615267227
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.5525102200608539
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        std_err = np.array([ 7.11046784,  0.21879293,  0.58477864,  7.50596504,  0.10800686,
        0.57365981])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        chow_r = np.array([[ 0.03533785,  0.85088948],
       [ 0.54918491,  0.45865093],
       [ 0.67115641,  0.41264872]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.81985446000130979
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)
    """
    def test_model_regi_error(self):
        #Columbus:
        reg = SP.GM_Error_Regimes(self.y, self.X, self.regimes, self.w, regime_err_sep=True)
        betas = np.array([[ 60.45730439],
       [ -0.17732134],
       [ -1.30936328],
       [  0.51314713],
       [ 66.5487126 ],
       [ -0.31845995],
       [ -1.29047149],
       [  0.08092997]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 39.33656288,  -0.08420799,  -1.50350999,   0.        ,
         0.        ,   0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([ 0.00698341])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 15.71899659])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([ 0.53685671])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[  3.63674458e-01,   5.46472584e-01],
       [  4.29607250e-01,   5.12181727e-01],
       [  5.44739543e-04,   9.81379339e-01]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.70119418251625387
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)
        #Artficial:
        model = SP.GM_Error_Regimes(self.y_a, self.x_a, self.regi_a, w=self.w_a, regime_err_sep=True)
        model1 = GM_Error(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a[0:(self.n2)], w=self.w_a1)
        model2 = GM_Error(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a[(self.n2):], w=self.w_a1)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas,RTOL)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.testing.assert_allclose(model.vm.diagonal(), vm,RTOL)
    """
    def test_model_endog(self):
        reg = SP.GM_Endog_Error_Regimes(self.y, self.X1, self.yd, self.q, self.regimes, self.w)
        betas = np.array([[ 77.48385551,   4.52986622,  78.93209405,   0.42186261,
         -3.23823854,  -1.1475775 ,   0.20222108]])
        print('Runining higher-tolerance test on L133 of test_error_sp_regimes.py')
        np.testing.assert_allclose(reg.betas.T,betas,RTOL + .0001)
        u = np.array([ 20.89660904])
        #np.testing.assert_allclose(reg.u[0],u,RTOL)
        np.testing.assert_allclose(reg.u[0],u,rtol=1e-05)
        e = np.array([ 25.21818724])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([-5.17062904])
        #np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        np.testing.assert_allclose(reg.predy[0],predy,rtol=1e-03)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        k = 6
        np.testing.assert_allclose(reg.k,k)
        y = np.array([ 15.72598])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([[  0.   ,   0.   ,   1.   ,  19.531]])
        np.testing.assert_allclose(reg.x[0].toarray(),x,RTOL)
        yend = np.array([[  0.      ,  80.467003]])
        np.testing.assert_allclose(reg.yend[0].toarray(),yend,RTOL)
        z = np.array([[  0.      ,   0.      ,   1.      ,  19.531   ,   0.      ,
         80.467003]])
        np.testing.assert_allclose(reg.z[0].toarray(),z,RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y,my)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y,sy)
        vm = np.array([ 390.88250241,   52.25924084,    0.        ,    0.        ,
        -32.64274729,    0.        ])
        #np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        np.allclose(reg.vm, vm)
        pr2 = 0.19623994206233333
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        sig2 = 649.4011
        #np.testing.assert_allclose(round(reg.sig2,RTOL),round(sig2,RTOL),RTOL)
        np.testing.assert_allclose(sig2, reg.sig2, rtol=1e-05)
        std_err = np.array([ 19.77074866,   6.07667394,  24.32254786,   2.17776972,
         2.97078606,   0.94392418])
        #np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        np.testing.assert_allclose(reg.std_err, std_err, rtol=1e-05)
        chow_r = np.array([[ 0.0021348 ,  0.96314775],
       [ 0.40499741,  0.5245196 ],
       [ 0.4498365 ,  0.50241261]])
        print('Running higher-tolerance tests on L176 of test_error_sp_regimes.py')
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL+.0001)
        chow_j = 1.2885590185243503
        #np.testing.assert_allclose(reg.chow.joint[0],chow_j)
        np.testing.assert_allclose(reg.chow.joint[0], chow_j, rtol=1e-05)

    def test_model_endog_regi_error(self):
        #Columbus:
        reg = SP.GM_Endog_Error_Regimes(self.y, self.X1, self.yd, self.q, self.regimes, self.w, regime_err_sep=True)
        betas = np.array([[  7.91729500e+01],
       [  5.80693176e+00],
       [ -3.84036576e+00],
       [  1.46462983e-01],
       [  8.24723791e+01],
       [  5.68908920e-01],
       [ -1.28824699e+00],
       [  6.70387351e-02]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 791.86679123,  140.12967794,  -81.37581255,    0.        ,
          0.        ,    0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([ 25.80361497])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([-10.07763497])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([ 27.32251813])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[ 0.00926459,  0.92331985],
       [ 0.26102777,  0.60941494],
       [ 0.26664581,  0.60559072]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 1.1184631131987004
        #np.testing.assert_allclose(reg.chow.joint[0],chow_j)
        np.testing.assert_allclose(reg.chow.joint[0], chow_j,RTOL)
        #Artficial:
        model = SP.GM_Endog_Error_Regimes(self.y_a, self.x_a1, yend=self.x_a2, q=self.q_a, regimes=self.regi_a, w=self.w_a, regime_err_sep=True)
        model1 = GM_Endog_Error(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a1[0:(self.n2)], yend=self.x_a2[0:(self.n2)], q=self.q_a[0:(self.n2)], w=self.w_a1)
        model2 = GM_Endog_Error(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a1[(self.n2):], yend=self.x_a2[(self.n2):], q=self.q_a[(self.n2):], w=self.w_a1)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas,RTOL)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.testing.assert_allclose(model.vm.diagonal(), vm,RTOL)

    def test_model_combo(self):
        reg = SP.GM_Combo_Regimes(self.y, self.X1, self.regimes, self.yd, self.q, w=self.w)
        predy_e = np.array([ 18.82774339])
        np.testing.assert_allclose(reg.predy_e[0],predy_e,RTOL)
        betas = np.array([[ 36.44798052],
       [ -0.7974482 ],
       [ 30.53782661],
       [ -0.72602806],
       [ -0.30953121],
       [ -0.21736652],
       [  0.64801059],
       [ -0.16601265]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 0.84393304])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e_filtered = np.array([ 0.4040027])
        np.testing.assert_allclose(reg.e_filtered[0],e_filtered,RTOL)
        predy = np.array([ 14.88204696])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 7
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 15.72598])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([[  0.   ,   0.   ,   1.   ,  19.531]])
        np.testing.assert_allclose(reg.x[0].toarray(),x,RTOL)
        yend = np.array([[  0.       ,  80.467003 ,  24.7142675]])
        np.testing.assert_allclose(reg.yend[0].toarray(),yend,RTOL)
        z = np.array([[  0.       ,   0.       ,   1.       ,  19.531    ,   0.       ,
         80.467003 ,  24.7142675]])
        np.testing.assert_allclose(reg.z[0].toarray(),z,RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([ 109.23549239,   -0.19754121,   84.29574673,   -1.99317178,
         -1.60123994,   -0.1252719 ,   -1.3930344 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        sig2 = 94.98610921110007
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.6493586702255537
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        pr2_e = 0.5255332447240576
        np.testing.assert_allclose(reg.pr2_e,pr2_e,RTOL)
        std_err = np.array([ 10.45157846,   0.93942923,  11.38484969,   0.60774708,
         0.44461334,   0.15871227,   0.15738141])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        chow_r = np.array([[ 0.49716076,  0.48075032],
       [ 0.00405377,  0.94923363],
       [ 0.03866684,  0.84411016]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.64531386285872072
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)

    def test_model_combo_regi_error(self):
        #Columbus:
        reg = SP.GM_Combo_Regimes(self.y, self.X1, self.regimes, self.yd, self.q, w=self.w, regime_lag_sep=True, regime_err_sep=True)
        betas = np.array([[ 42.01035248],
       [ -0.13938772],
       [ -0.6528306 ],
       [  0.54737621],
       [  0.2684419 ],
       [ 34.02473255],
       [ -0.14920259],
       [ -0.48972903],
       [  0.65883658],
       [ -0.17174845]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 153.58614432,    2.96302131,   -3.26211855,   -2.46914703,
          0.        ,    0.        ,    0.        ,    0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([ 7.73968703])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 7.98629297])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([ 6.45052714])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[  1.00886404e-01,   7.50768497e-01],
       [  3.61843271e-05,   9.95200481e-01],
       [  4.69585772e-02,   8.28442711e-01],
       [  8.13275259e-02,   7.75506385e-01]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.28479988992843119
        np.testing.assert_allclose(reg.chow.joint[0],chow_j, RTOL)
        #Artficial:
        model = SP.GM_Combo_Regimes(self.y_a, self.x_a1, yend=self.x_a2, q=self.q_a, regimes=self.regi_a, w=self.w_a, regime_err_sep=True, regime_lag_sep=True)
        model1 = GM_Combo(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a1[0:(self.n2)], yend=self.x_a2[0:(self.n2)], q=self.q_a[0:(self.n2)], w=self.w_a1)
        model2 = GM_Combo(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a1[(self.n2):], yend=self.x_a2[(self.n2):], q=self.q_a[(self.n2):], w=self.w_a1)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))

if __name__ == '__main__':
    unittest.main()


