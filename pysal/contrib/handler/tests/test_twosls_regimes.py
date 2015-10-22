import unittest
import numpy as np
import pysal
#from pysal.spreg.twosls_regimes import TSLS_Regimes
#from pysal.spreg.twosls import TSLS

from pysal.contrib.handler import Model
from functools import partial

TSLS_Regimes = partial(Model, mtype='TSLS_Regimes')
TSLS = partial(Model, mtype='TSLS')

class TestTSLS(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
        self.y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(self.y, (49,1))
        self.x = []
        self.x.append(db.by_col("INC"))
        self.x = np.array(self.x).T
        self.yd = []
        self.yd.append(db.by_col("HOVAL"))
        self.yd = np.array(self.yd).T
        self.q = []
        self.q.append(db.by_col("DISCBD"))
        self.q = np.array(self.q).T
        self.r_var = 'NSA'
        self.regimes = db.by_col(self.r_var)

    def test_basic(self):
        reg = TSLS_Regimes(self.y, self.x, self.yd, self.q, self.regimes, regime_err_sep=False)        
        betas = np.array([[ 80.23408166],[  5.48218125],[ 82.98396737],[  0.49775429],[ -3.72663211],[ -1.27451485]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        h_0 = np.array([[  0.   ,   0.   ,   1.   ,  19.531,   0.   ,   5.03 ]])
        np.testing.assert_array_almost_equal(reg.h[0]*np.eye(6), h_0)
        hth = np.array([[   25.        ,   416.378999  ,     0.        ,     0.        ,\
           76.03      ,     0.        ],\
       [  416.378999  ,  7831.05477839,     0.        ,     0.        ,\
         1418.65422625,     0.        ],\
       [    0.        ,     0.        ,    24.        ,   287.993     ,\
            0.        ,    63.72      ],\
       [    0.        ,     0.        ,   287.993     ,  3855.61860282,\
            0.        ,   827.47378   ],\
       [   76.03      ,  1418.65422625,     0.        ,     0.        ,\
          291.9749    ,     0.        ],\
       [    0.        ,     0.        ,    63.72      ,   827.47378   ,\
            0.        ,   206.6102    ]])
        np.testing.assert_array_almost_equal(reg.hth, hth, 7)
        hthi = np.array([[ 0.3507855 , -0.0175615 ,  0.        ,  0.        , -0.00601601,\
         0.        ],\
       [-0.0175615 ,  0.00194521, -0.        , -0.        , -0.00487844,\
        -0.        ],\
       [ 0.        ,  0.        ,  0.42327489, -0.02563036,  0.        ,\
        -0.02789128],\
       [-0.        , -0.        , -0.02563036,  0.00339841, -0.        ,\
        -0.00570605],\
       [-0.00601601, -0.00487844,  0.        ,  0.        ,  0.02869498,\
         0.        ],\
       [ 0.        ,  0.        , -0.02789128, -0.00570605,  0.        ,\
         0.03629464]])
        np.testing.assert_array_almost_equal(reg.hthi, hthi, 7)
        self.assertEqual(reg.k, 6)
        self.assertEqual(reg.kstar, 2)
        self.assertAlmostEqual(reg.mean_y, 35.128823897959187, 7)
        np.testing.assert_equal(reg.kf, 0)
        np.testing.assert_equal(reg.kr, 3)
        np.testing.assert_equal(reg.n, 49)
        np.testing.assert_equal(reg.nr, 2)
        pfora1a2 = np.array([[ 17.80208995,  -0.46997739,   0.        ,   0.        ,\
         -0.21344994,   0.        ],\
       [ -0.36293902,   0.41200496,   0.        ,   0.        ,\
         -0.17308863,   0.        ],\
       [  0.        ,   0.        ,  23.8584271 ,  -0.96035493,\
          0.        ,  -0.26149141],\
       [  0.        ,   0.        ,  -0.61800983,   0.2269828 ,\
          0.        ,  -0.05349643],\
       [ -3.22151864,  -2.10181214,   0.        ,   0.        ,\
          1.01810757,   0.        ],\
       [  0.        ,   0.        ,  -5.42403871,  -0.6641704 ,\
          0.        ,   0.34027606]]) 
        np.testing.assert_array_almost_equal(reg.pfora1a2, pfora1a2, 7)
        predy_5 = np.array([[ -9.85078372],[ 36.75098196],[ 57.34266859],[ 42.89851907],[ 58.9840913 ]]) 
        np.testing.assert_array_almost_equal(reg.predy[0:5], predy_5, 7)
        q_5 = np.array([ 5.03,  4.27,  3.89,  3.7 ,  2.83])
        np.testing.assert_array_equal((reg.q[0:5].T*np.eye(5))[1,:], q_5)
        self.assertAlmostEqual(reg.sig2n_k, 990.00750983736714, 7)
        self.assertAlmostEqual(reg.sig2n, 868.78210046952631, 7)
        self.assertAlmostEqual(reg.sig2, 990.00750983736714, 7)
        self.assertAlmostEqual(reg.std_y, 16.732092091229699, 7)
        u_5 = np.array([[ 25.57676372],[-17.94922796],[-26.71588759],[-10.51075907],[ -8.2525813 ]]) 
        np.testing.assert_array_almost_equal(reg.u[0:5], u_5, 7)
        self.assertAlmostEqual(reg.utu, 42570.322923006788, 7)
        varb = np.array([[ 0.50015831,  0.07969376,  0.        ,  0.        , -0.04760541,\
         0.        ],\
       [ 0.07969376,  0.06523527,  0.        ,  0.        , -0.03105915,\
         0.        ],\
       [ 0.        ,  0.        ,  0.73944792,  0.01132445,  0.        ,\
        -0.02117969],\
       [ 0.        ,  0.        ,  0.01132445,  0.00756336,  0.        ,\
        -0.00259344],\
       [-0.04760541, -0.03105915, -0.        , -0.        ,  0.0150449 ,\
        -0.        ],\
       [-0.        , -0.        , -0.02117969, -0.00259344, -0.        ,\
         0.0013287 ]]) 
        np.testing.assert_array_almost_equal(reg.varb, varb, 7)
        vm = np.array([[ 495.16048523,   78.89742341,    0.        ,    0.        ,\
         -47.12971066,    0.        ],\
       [  78.89742341,   64.58341083,    0.        ,    0.        ,\
         -30.74878934,    0.        ],\
       [   0.        ,    0.        ,  732.05899155,   11.21128921,\
           0.        ,  -20.96804956],\
       [   0.        ,    0.        ,   11.21128921,    7.48778398,\
           0.        ,   -2.56752553],\
       [ -47.12971066,  -30.74878934,    0.        ,    0.        ,\
          14.89456384,    0.        ],\
       [   0.        ,    0.        ,  -20.96804956,   -2.56752553,\
           0.        ,    1.3154267 ]]) 
        np.testing.assert_array_almost_equal(reg.vm, vm, 7)
        x_0 = np.array([[  0.   ,   0.   ,   1.   ,  19.531]])
        np.testing.assert_array_almost_equal(reg.x[0]*np.eye(4), x_0, 7)
        y_5 = np.array([[ 15.72598 ], [ 18.801754], [ 30.626781], [ 32.38776 ], [ 50.73151 ]]) 
        np.testing.assert_array_almost_equal(reg.y[0:5], y_5, 7)
        yend_3 = np.array([[  0.      ,  80.467003],[  0.      ,  44.567001],[  0.      ,  26.35    ]]) 
        np.testing.assert_array_almost_equal(reg.yend[0:3]*np.eye(2), yend_3, 7)
        z_0 = np.array([[  0.      ,   0.      ,   1.      ,  19.531   ,   0.      , 80.467003]]) 
        np.testing.assert_array_almost_equal(reg.z[0]*np.eye(6), z_0, 7)
        zthhthi = np.array([[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,\
          0.00000000e+00,  -4.44089210e-16,   0.00000000e+00],\
       [ -1.24344979e-14,   1.00000000e+00,   0.00000000e+00,\
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,\
          0.00000000e+00,   0.00000000e+00,  -1.11022302e-16],\
       [  0.00000000e+00,   0.00000000e+00,  -3.55271368e-15,\
          1.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
       [  2.87468088e+00,   1.82963841e+00,   0.00000000e+00,\
          0.00000000e+00,   1.38104644e+00,   0.00000000e+00],\
       [  0.00000000e+00,   0.00000000e+00,   1.19237474e+01,\
          1.13018165e+00,   0.00000000e+00,   5.22645427e+00]]) 
        np.testing.assert_array_almost_equal(reg.zthhthi, zthhthi, 7)
        self.assertAlmostEqual(reg.pr2, 0.17729324026706564, 7)
        z_stat = np.array([[  3.60566933e+00,   3.11349387e-04],\
       [  6.82170447e-01,   4.95131179e-01],\
       [  3.06705211e+00,   2.16181168e-03],\
       [  1.81902371e-01,   8.55659343e-01],\
       [ -9.65611937e-01,   3.34238400e-01],\
       [ -1.11124949e+00,   2.66460976e-01]])
        np.testing.assert_array_almost_equal(np.array(reg.z_stat), z_stat, 7)
        chow_regi = np.array([[ 0.00616179,  0.93743265],
       [ 0.3447218 ,  0.55711631],
       [ 0.37093662,  0.54249417]])
        np.testing.assert_array_almost_equal(reg.chow.regi, chow_regi, 7)
        self.assertAlmostEqual(reg.chow.joint[0], 1.1353790779820598, 7)
        title = 'TWO STAGE LEAST SQUARES - REGIMES'
        self.assertEqual(reg.title, title)
        
    def test_n_k(self):
        reg = TSLS_Regimes(self.y, self.x, self.yd, self.q, self.regimes, sig2n_k=True, regime_err_sep=False)
        betas = np.array([[ 80.23408166],[  5.48218125],[ 82.98396737],[  0.49775429],[ -3.72663211],[ -1.27451485]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        vm = np.array([[ 495.16048523,   78.89742341,    0.        ,    0.        ,\
         -47.12971066,    0.        ],\
       [  78.89742341,   64.58341083,    0.        ,    0.        ,\
         -30.74878934,    0.        ],\
       [   0.        ,    0.        ,  732.05899155,   11.21128921,\
           0.        ,  -20.96804956],\
       [   0.        ,    0.        ,   11.21128921,    7.48778398,\
           0.        ,   -2.56752553],\
       [ -47.12971066,  -30.74878934,    0.        ,    0.        ,\
          14.89456384,    0.        ],\
       [   0.        ,    0.        ,  -20.96804956,   -2.56752553,\
           0.        ,    1.3154267 ]]) 
        np.testing.assert_array_almost_equal(reg.vm, vm, 7)

    def test_spatial(self):
        w = pysal.queen_from_shapefile(pysal.examples.get_path('columbus.shp'))
        reg = TSLS_Regimes(self.y, self.x, self.yd, self.q, self.regimes, spat_diag=True, w=w, regime_err_sep=False)
        betas = np.array([[ 80.23408166],[  5.48218125],[ 82.98396737],[  0.49775429],[ -3.72663211],[ -1.27451485]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        vm = np.array([[ 495.16048523,   78.89742341,    0.        ,    0.        ,\
         -47.12971066,    0.        ],\
       [  78.89742341,   64.58341083,    0.        ,    0.        ,\
         -30.74878934,    0.        ],\
       [   0.        ,    0.        ,  732.05899155,   11.21128921,\
           0.        ,  -20.96804956],\
       [   0.        ,    0.        ,   11.21128921,    7.48778398,\
           0.        ,   -2.56752553],\
       [ -47.12971066,  -30.74878934,    0.        ,    0.        ,\
          14.89456384,    0.        ],\
       [   0.        ,    0.        ,  -20.96804956,   -2.56752553,\
           0.        ,    1.3154267 ]]) 
        np.testing.assert_array_almost_equal(reg.vm, vm, 7)
        ak_test = np.array([ 0.69774552,  0.40354227])
        np.testing.assert_array_almost_equal(reg.ak_test, ak_test, 7)   

    def test_names(self):
        w = pysal.queen_from_shapefile(pysal.examples.get_path('columbus.shp'))
        gwk = pysal.kernelW_from_shapefile(pysal.examples.get_path('columbus.shp'),k=5,function='triangular', fixed=False)
        name_x = ['inc']
        name_y = 'crime'
        name_yend = ['hoval']
        name_q = ['discbd']
        name_w = 'queen'
        name_gwk = 'k=5'
        name_ds = 'columbus'
        name_regimes= 'nsa'
        reg = TSLS_Regimes(self.y, self.x, self.yd, self.q, self.regimes, regime_err_sep=False,
                spat_diag=True, w=w, robust='hac', gwk=gwk, name_regimes=name_regimes,
                name_x=name_x, name_y=name_y, name_q=name_q, name_w=name_w,
                name_yend=name_yend, name_gwk=name_gwk, name_ds=name_ds)
        betas = np.array([[ 80.23408166],[  5.48218125],[ 82.98396737],[  0.49775429],[ -3.72663211],[ -1.27451485]])
        np.testing.assert_array_almost_equal(reg.betas, betas, 7)
        vm = np.array([[ 522.75813101,  120.64940697,  -15.60303241,   -0.976389  ,\
         -67.15556574,    0.64553579],\
       [ 122.83491674,  122.62303068,   -5.52270916,    0.05023488,\
         -57.89404902,    0.15750428],\
       [   0.1983661 ,   -0.03539147,  335.24731378,   17.40764168,\
          -0.26447114,  -14.3375455 ],\
       [  -0.13612426,   -0.43622084,   18.46644989,    2.70320508,\
           0.20398876,   -1.31821991],\
       [ -68.0704928 ,  -58.03685405,    2.66225388,    0.00323082,\
          27.68512974,   -0.08124602],\
       [  -0.08001296,    0.13575504,  -14.6998294 ,   -1.28225201,\
          -0.05193056,    0.79845124]])
        np.testing.assert_array_almost_equal(reg.vm, vm, 7)
        self.assertEqual(reg.name_x, ['0_CONSTANT', '0_inc', '1_CONSTANT', '1_inc'])
        self.assertEqual(reg.name_yend, ['0_hoval', '1_hoval'])
        self.assertEqual(reg.name_q, ['0_discbd', '1_discbd'])
        self.assertEqual(reg.name_y, name_y)
        self.assertEqual(reg.name_w, name_w)
        self.assertEqual(reg.name_gwk, name_gwk)
        self.assertEqual(reg.name_ds, name_ds)
        self.assertEqual(reg.name_regimes, name_regimes)
    
    def test_regi_err(self):
        #Artficial:
        n = 256
        x1 = np.random.uniform(-10,10,(n,1))
        x2 = np.random.uniform(1,5,(n,1))
        q = x2 + np.random.normal(0,1,(n,1))
        x = np.hstack((x1,x2))
        y = np.dot(np.hstack((np.ones((n,1)),x)),np.array([[1],[0.5],[2]])) + np.random.normal(0,1,(n,1))
        latt = int(np.sqrt(n))
        regi = [0]*(n/2) + [1]*(n/2)
        model = TSLS_Regimes(y, x1, regimes=regi, q=q, yend=x2, regime_err_sep=True, sig2n_k=False)
        model1 = TSLS(y[0:(n/2)].reshape((n/2),1), x1[0:(n/2)], yend=x2[0:(n/2)], q=q[0:(n/2)], sig2n_k=False)
        model2 = TSLS(y[(n/2):n].reshape((n/2),1), x1[(n/2):n], yend=x2[(n/2):n], q=q[(n/2):n], sig2n_k=False)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_array_almost_equal(model.betas,tbetas)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.testing.assert_array_almost_equal(model.vm.diagonal(), vm, 6)
        #Columbus:
        reg = TSLS_Regimes(self.y, self.x, regimes=self.regimes, yend=self.yd, q=self.q, regime_err_sep=False)
        tbetas = np.array([[ 80.23408166],
       [  5.48218125],
       [ 82.98396737],
       [  0.49775429],
       [ -3.72663211],
       [ -1.27451485],])
        np.testing.assert_array_almost_equal(tbetas, reg.betas)
        vm = np.array([ 495.16048523,   78.89742341,    0.        ,    0.        ,
        -47.12971066,    0.        ])
        np.testing.assert_array_almost_equal(reg.vm[0], vm, 6)
        u_3 = np.array([[ 25.57676372],
       [-17.94922796],
       [-26.71588759]])
        np.testing.assert_array_almost_equal(reg.u[0:3], u_3, 7)
        predy_3 = np.array([[ -9.85078372],
       [ 36.75098196],
       [ 57.34266859]])
        np.testing.assert_array_almost_equal(reg.predy[0:3], predy_3, 7)
        chow_regi = np.array([[ 0.00616179,  0.93743265],
       [ 0.3447218 ,  0.55711631],
       [ 0.37093662,  0.54249417]])
        np.testing.assert_array_almost_equal(reg.chow.regi, chow_regi, 7)
        self.assertAlmostEqual(reg.chow.joint[0], 1.1353790779821029, 7)

if __name__ == '__main__':
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)  
    unittest.main()
    np.set_printoptions(suppress=start_suppress)        
    
