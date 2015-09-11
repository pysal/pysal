import unittest
import numpy as np
import pysal
from pysal.spreg.twosls import BaseTSLS, TSLS
from pysal.common import RTOL

class TestBaseTSLS(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
        self.y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(self.y, (49,1))
        self.X = []
        self.X.append(db.by_col("INC"))
        self.X = np.array(self.X).T
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        self.yd = []
        self.yd.append(db.by_col("HOVAL"))
        self.yd = np.array(self.yd).T
        self.q = []
        self.q.append(db.by_col("DISCBD"))
        self.q = np.array(self.q).T

    def test_basic(self):
        reg = BaseTSLS(self.y, self.X, self.yd, self.q)
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        h_0 = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_allclose(reg.h[0], h_0)
        hth = np.array([[    49.        ,    704.371999  ,    139.75      ],
                        [   704.371999  ,  11686.67338121,   2246.12800625],
                        [   139.75      ,   2246.12800625,    498.5851    ]])
        np.testing.assert_allclose(reg.hth, hth,RTOL)
        hthi = np.array([[ 0.1597275 , -0.00762011, -0.01044191],
                        [-0.00762011,  0.00100135, -0.0023752 ],
                        [-0.01044191, -0.0023752 ,  0.01563276]]) 
        np.testing.assert_allclose(reg.hthi, hthi,RTOL)
        self.assertEqual(reg.k, 3)
        self.assertEqual(reg.kstar, 1)
        np.testing.assert_allclose(reg.mean_y, 35.128823897959187,RTOL)
        self.assertEqual(reg.n, 49)
        pfora1a2 = np.array([[ 9.58156106, -0.22744226, -0.13820537],
                             [ 0.02580142,  0.08226331, -0.03143731],
                             [-3.13896453, -0.33487872,  0.20690965]]) 
        np.testing.assert_allclose(reg.pfora1a2, pfora1a2,RTOL)
        predy_5 = np.array([[-28.68949467], [ 28.99484984], [ 55.07344824], [ 38.26609504], [ 57.57145851]]) 
        np.testing.assert_allclose(reg.predy[0:5], predy_5,RTOL)
        q_5 = np.array([[ 5.03], [ 4.27], [ 3.89], [ 3.7 ], [ 2.83]])
        np.testing.assert_array_equal(reg.q[0:5], q_5)
        np.testing.assert_allclose(reg.sig2n_k, 587.56797852699822,RTOL)
        np.testing.assert_allclose(reg.sig2n, 551.5944288212637,RTOL)
        np.testing.assert_allclose(reg.sig2, 551.5944288212637,RTOL)
        np.testing.assert_allclose(reg.std_y, 16.732092091229699,RTOL)
        u_5 = np.array([[ 44.41547467], [-10.19309584], [-24.44666724], [ -5.87833504], [ -6.83994851]]) 
        np.testing.assert_allclose(reg.u[0:5], u_5,RTOL)
        np.testing.assert_allclose(reg.utu, 27028.127012241919,RTOL)
        varb = np.array([[ 0.41526237,  0.01879906, -0.01730372],
                         [ 0.01879906,  0.00362823, -0.00184604],
                         [-0.01730372, -0.00184604,  0.0011406 ]]) 
        np.testing.assert_allclose(reg.varb, varb,RTOL)
        vm = np.array([[ 229.05640809,   10.36945783,   -9.54463414],
                       [  10.36945783,    2.0013142 ,   -1.01826408],
                       [  -9.54463414,   -1.01826408,    0.62914915]]) 
        np.testing.assert_allclose(reg.vm, vm,RTOL)
        x_0 = np.array([  1.   ,  19.531])
        np.testing.assert_allclose(reg.x[0], x_0,RTOL)
        y_5 = np.array([[ 15.72598 ], [ 18.801754], [ 30.626781], [ 32.38776 ], [ 50.73151 ]]) 
        np.testing.assert_allclose(reg.y[0:5], y_5,RTOL)
        yend_5 = np.array([[ 80.467003], [ 44.567001], [ 26.35    ], [ 33.200001], [ 23.225   ]]) 
        np.testing.assert_allclose(reg.yend[0:5], yend_5,RTOL)
        z_0 = np.array([  1.      ,  19.531   ,  80.467003]) 
        np.testing.assert_allclose(reg.z[0], z_0,RTOL)
        zthhthi = np.array([[  1.00000000e+00,  -1.66533454e-16,   4.44089210e-16],
                            [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00],
                            [  1.26978671e+01,   1.05598709e+00,   3.70212359e+00]]) 
        #np.testing.assert_allclose(reg.zthhthi, zthhthi, RTOL)
        np.testing.assert_array_almost_equal(reg.zthhthi, zthhthi, 7)
        
    def test_n_k(self):
        reg = BaseTSLS(self.y, self.X, self.yd, self.q, sig2n_k=True)
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        vm = np.array([[ 243.99486949,   11.04572682,  -10.16711028],
                       [  11.04572682,    2.13183469,   -1.08467261],
                       [ -10.16711028,   -1.08467261,    0.67018062]]) 
        np.testing.assert_allclose(reg.vm, vm,RTOL)

    def test_white(self):
        reg = BaseTSLS(self.y, self.X, self.yd, self.q, robust='white')
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        vm = np.array([[ 208.27139316,   15.6687805 ,  -11.53686154],
                       [  15.6687805 ,    2.26882747,   -1.30312033],
                       [ -11.53686154,   -1.30312033,    0.81940656]]) 
        np.testing.assert_allclose(reg.vm, vm,RTOL)

    def test_hac(self):
        gwk = pysal.kernelW_from_shapefile(pysal.examples.get_path('columbus.shp'),k=15,function='triangular', fixed=False)
        reg = BaseTSLS(self.y, self.X, self.yd, self.q, robust='hac', gwk=gwk)
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        vm = np.array([[ 231.07254978,   15.42050291,  -11.3941033 ],
                       [  15.01376346,    1.92422887,   -1.11865505],
                       [ -11.34381641,   -1.1279227 ,    0.72053806]]) 
        np.testing.assert_allclose(reg.vm, vm,RTOL)

class TestTSLS(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
        self.y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(self.y, (49,1))
        self.X = []
        self.X.append(db.by_col("INC"))
        self.X = np.array(self.X).T
        self.yd = []
        self.yd.append(db.by_col("HOVAL"))
        self.yd = np.array(self.yd).T
        self.q = []
        self.q.append(db.by_col("DISCBD"))
        self.q = np.array(self.q).T

    def test_basic(self):
        reg = TSLS(self.y, self.X, self.yd, self.q)
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        h_0 = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_allclose(reg.h[0], h_0)
        hth = np.array([[    49.        ,    704.371999  ,    139.75      ],
                        [   704.371999  ,  11686.67338121,   2246.12800625],
                        [   139.75      ,   2246.12800625,    498.5851    ]])
        np.testing.assert_allclose(reg.hth, hth,RTOL)
        hthi = np.array([[ 0.1597275 , -0.00762011, -0.01044191],
                        [-0.00762011,  0.00100135, -0.0023752 ],
                        [-0.01044191, -0.0023752 ,  0.01563276]]) 
        np.testing.assert_allclose(reg.hthi, hthi,RTOL)
        self.assertEqual(reg.k, 3)
        self.assertEqual(reg.kstar, 1)
        np.testing.assert_allclose(reg.mean_y, 35.128823897959187,RTOL)
        self.assertEqual(reg.n, 49)
        pfora1a2 = np.array([[ 9.58156106, -0.22744226, -0.13820537],
                             [ 0.02580142,  0.08226331, -0.03143731],
                             [-3.13896453, -0.33487872,  0.20690965]]) 
        np.testing.assert_allclose(reg.pfora1a2, pfora1a2,RTOL)
        predy_5 = np.array([[-28.68949467], [ 28.99484984], [ 55.07344824], [ 38.26609504], [ 57.57145851]]) 
        np.testing.assert_allclose(reg.predy[0:5], predy_5,RTOL)
        q_5 = np.array([[ 5.03], [ 4.27], [ 3.89], [ 3.7 ], [ 2.83]])
        np.testing.assert_array_equal(reg.q[0:5], q_5)
        np.testing.assert_allclose(reg.sig2n_k, 587.56797852699822,RTOL)
        np.testing.assert_allclose(reg.sig2n, 551.5944288212637,RTOL)
        np.testing.assert_allclose(reg.sig2, 551.5944288212637,RTOL)
        np.testing.assert_allclose(reg.std_y, 16.732092091229699,RTOL)
        u_5 = np.array([[ 44.41547467], [-10.19309584], [-24.44666724], [ -5.87833504], [ -6.83994851]]) 
        np.testing.assert_allclose(reg.u[0:5], u_5,RTOL)
        np.testing.assert_allclose(reg.utu, 27028.127012241919,RTOL)
        varb = np.array([[ 0.41526237,  0.01879906, -0.01730372],
                         [ 0.01879906,  0.00362823, -0.00184604],
                         [-0.01730372, -0.00184604,  0.0011406 ]]) 
        np.testing.assert_allclose(reg.varb, varb,RTOL)
        vm = np.array([[ 229.05640809,   10.36945783,   -9.54463414],
                       [  10.36945783,    2.0013142 ,   -1.01826408],
                       [  -9.54463414,   -1.01826408,    0.62914915]]) 
        np.testing.assert_allclose(reg.vm, vm,RTOL)
        x_0 = np.array([  1.   ,  19.531])
        np.testing.assert_allclose(reg.x[0], x_0,RTOL)
        y_5 = np.array([[ 15.72598 ], [ 18.801754], [ 30.626781], [ 32.38776 ], [ 50.73151 ]]) 
        np.testing.assert_allclose(reg.y[0:5], y_5,RTOL)
        yend_5 = np.array([[ 80.467003], [ 44.567001], [ 26.35    ], [ 33.200001], [ 23.225   ]]) 
        np.testing.assert_allclose(reg.yend[0:5], yend_5,RTOL)
        z_0 = np.array([  1.      ,  19.531   ,  80.467003]) 
        np.testing.assert_allclose(reg.z[0], z_0,RTOL)
        zthhthi = np.array([[  1.00000000e+00,  -1.66533454e-16,   4.44089210e-16],
                            [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00],
                            [  1.26978671e+01,   1.05598709e+00,   3.70212359e+00]]) 
        #np.testing.assert_allclose(reg.zthhthi, zthhthi,RTOL)
        np.testing.assert_array_almost_equal(reg.zthhthi, zthhthi, 7)
        np.testing.assert_allclose(reg.pr2, 0.27936137128173893,RTOL)
        z_stat = np.array([[  5.84526447e+00,   5.05764078e-09],
                           [  3.67601567e-01,   7.13170346e-01],
                           [ -1.99468913e+00,   4.60767956e-02]])
        np.testing.assert_allclose(reg.z_stat, z_stat,RTOL)
        title = 'TWO STAGE LEAST SQUARES'
        self.assertEqual(reg.title, title)
        
    def test_n_k(self):
        reg = TSLS(self.y, self.X, self.yd, self.q, sig2n_k=True)
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        vm = np.array([[ 243.99486949,   11.04572682,  -10.16711028],
                       [  11.04572682,    2.13183469,   -1.08467261],
                       [ -10.16711028,   -1.08467261,    0.67018062]]) 
        np.testing.assert_allclose(reg.vm, vm,RTOL)

    def test_white(self):
        reg = TSLS(self.y, self.X, self.yd, self.q, robust='white')
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        vm = np.array([[ 208.27139316,   15.6687805 ,  -11.53686154],
                       [  15.6687805 ,    2.26882747,   -1.30312033],
                       [ -11.53686154,   -1.30312033,    0.81940656]]) 
        np.testing.assert_allclose(reg.vm, vm,RTOL)
        self.assertEqual(reg.robust, 'white')

    def test_hac(self):
        gwk = pysal.kernelW_from_shapefile(pysal.examples.get_path('columbus.shp'),k=5,function='triangular', fixed=False)
        reg = TSLS(self.y, self.X, self.yd, self.q, robust='hac', gwk=gwk)
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        vm = np.array([[ 225.0795089 ,   17.11660041,  -12.22448566],
                       [  17.67097154,    2.47483461,   -1.4183641 ],
                       [ -12.45093722,   -1.40495464,    0.8700441 ]]) 
        np.testing.assert_allclose(reg.vm, vm,RTOL)
        self.assertEqual(reg.robust, 'hac')

    def test_spatial(self):
        w = pysal.queen_from_shapefile(pysal.examples.get_path('columbus.shp'))
        reg = TSLS(self.y, self.X, self.yd, self.q, spat_diag=True, w=w)
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        vm = np.array([[ 229.05640809,   10.36945783,   -9.54463414],
                       [  10.36945783,    2.0013142 ,   -1.01826408],
                       [  -9.54463414,   -1.01826408,    0.62914915]]) 
        np.testing.assert_allclose(reg.vm, vm,RTOL)
        ak_test = np.array([ 1.16816972,  0.27977763])
        np.testing.assert_allclose(reg.ak_test, ak_test,RTOL)

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
        reg = TSLS(self.y, self.X, self.yd, self.q,
                spat_diag=True, w=w, robust='hac', gwk=gwk,
                name_x=name_x, name_y=name_y, name_q=name_q, name_w=name_w,
                name_yend=name_yend, name_gwk=name_gwk, name_ds=name_ds)
        betas = np.array([[ 88.46579584], [  0.5200379 ], [ -1.58216593]])
        np.testing.assert_allclose(reg.betas, betas,RTOL)
        vm = np.array([[ 225.0795089 ,   17.11660041,  -12.22448566],
                       [  17.67097154,    2.47483461,   -1.4183641 ],
                       [ -12.45093722,   -1.40495464,    0.8700441 ]])
        np.testing.assert_allclose(reg.vm, vm,RTOL)
        self.assertListEqual(reg.name_x, ['CONSTANT']+name_x)
        self.assertListEqual(reg.name_yend, name_yend)
        self.assertListEqual(reg.name_q, name_q)
        self.assertEqual(reg.name_y, name_y)
        self.assertEqual(reg.name_w, name_w)
        self.assertEqual(reg.name_gwk, name_gwk)
        self.assertEqual(reg.name_ds, name_ds)

    


if __name__ == '__main__':
    unittest.main()
