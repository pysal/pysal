'''
Unittests for spreg.error_sp_hom module

'''
import unittest
import pysal
from pysal.spreg import error_sp_hom as HOM
import numpy as np
from pysal.common import RTOL

class BaseGM_Error_Hom_Tester(unittest.TestCase):
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
        reg = HOM.BaseGM_Error_Hom(self.y, self.X, self.w.sparse, A1='hom_sc')
        np.testing.assert_allclose(reg.y[0],np.array([80.467003]),RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        betas = np.array([[ 47.9478524 ], [  0.70633223], [ -0.55595633], [  0.41288558]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        np.testing.assert_allclose(reg.u[0],np.array([27.466734]),RTOL)
        np.testing.assert_allclose(reg.e_filtered[0],np.array([ 32.37298547]),RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        np.testing.assert_allclose(reg.predy[0],np.array([ 53.000269]),RTOL)
        np.testing.assert_allclose(reg.n,49,RTOL)
        np.testing.assert_allclose(reg.k,3,RTOL)
        sig2 = 189.94459439729718
        np.testing.assert_allclose(reg.sig2,sig2)
        vm = np.array([[  1.51340717e+02,  -5.29057506e+00,  -1.85654540e+00, -2.39139054e-03], [ -5.29057506e+00,   2.46669610e-01, 5.14259101e-02, 3.19241302e-04], [ -1.85654540e+00,   5.14259101e-02, 3.20510550e-02,  -5.95640240e-05], [ -2.39139054e-03,   3.19241302e-04, -5.95640240e-05,  3.36690159e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02, 1.72131237e+03], [  7.04371999e+02,   1.16866734e+04,   2.15575320e+04], [  1.72131237e+03,   2.15575320e+04, 7.39058986e+04]])
        np.testing.assert_allclose(reg.xtx,xtx,RTOL)

class GM_Error_Hom_Tester(unittest.TestCase):
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
        reg = HOM.GM_Error_Hom(self.y, self.X, self.w, A1='hom_sc')
        np.testing.assert_allclose(reg.y[0],np.array([80.467003]),RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        betas = np.array([[ 47.9478524 ], [  0.70633223], [ -0.55595633], [  0.41288558]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        np.testing.assert_allclose(reg.u[0],np.array([27.46673388]),RTOL)
        np.testing.assert_allclose(reg.e_filtered[0],np.array([ 32.37298547]),RTOL)
        np.testing.assert_allclose(reg.predy[0],np.array([ 53.00026912]),RTOL)
        np.testing.assert_allclose(reg.n,49,RTOL)
        np.testing.assert_allclose(reg.k,3,RTOL)
        vm = np.array([[  1.51340717e+02,  -5.29057506e+00,  -1.85654540e+00, -2.39139054e-03], [ -5.29057506e+00,   2.46669610e-01, 5.14259101e-02, 3.19241302e-04], [ -1.85654540e+00,   5.14259101e-02, 3.20510550e-02,  -5.95640240e-05], [ -2.39139054e-03,   3.19241302e-04, -5.95640240e-05,  3.36690159e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        np.testing.assert_allclose(reg.iteration,1,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        std_y = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,std_y)
        pr2 = 0.34950977055969729
        np.testing.assert_allclose(reg.pr2,pr2)
        sig2 = 189.94459439729718
        np.testing.assert_allclose(reg.sig2,sig2)
        std_err = np.array([ 12.30206149,   0.49665844,   0.17902808, 0.18349119])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([[  3.89754616e+00,   9.71723059e-05], [  1.42216900e+00,   1.54977196e-01], [ -3.10541409e+00,   1.90012806e-03], [  2.25016500e+00,   2.44384731e-02]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02, 1.72131237e+03], [  7.04371999e+02,   1.16866734e+04,   2.15575320e+04], [  1.72131237e+03,   2.15575320e+04, 7.39058986e+04]])
        np.testing.assert_allclose(reg.xtx,xtx,RTOL)


class BaseGM_Endog_Error_Hom_Tester(unittest.TestCase):
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
        reg = HOM.BaseGM_Endog_Error_Hom(self.y, self.X, self.yd, self.q, self.w.sparse, A1='hom_sc')
        np.testing.assert_allclose(reg.y[0],np.array([ 80.467003]),RTOL)
        x = np.array([  1.     ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        h = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_allclose(reg.h[0],h,RTOL)
        yend = np.array([ 15.72598])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        q = np.array([ 5.03])
        np.testing.assert_allclose(reg.q[0],q,RTOL)
        betas = np.array([[ 55.36575166], [  0.46432416], [ -0.66904404], [  0.43205526]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 26.55390939])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        np.testing.assert_allclose(reg.e_filtered[0],np.array([ 31.74114306]),RTOL)
        predy = np.array([ 53.91309361])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        np.testing.assert_allclose(reg.n,49,RTOL)
        np.testing.assert_allclose(reg.k,3,RTOL)
        sig2 = 190.59435238060928
        np.testing.assert_allclose(reg.sig2,sig2)
        vm = np.array([[  5.52064057e+02,  -1.61264555e+01,  -8.86360735e+00, 1.04251912e+00], [ -1.61264555e+01,   5.44898242e-01, 2.39518645e-01, -1.88092950e-02], [ -8.86360735e+00,   2.39518645e-01, 1.55501840e-01, -2.18638648e-02], [  1.04251912e+00, -1.88092950e-02, -2.18638648e-02, 3.71222222e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        np.testing.assert_allclose(reg.iteration,its,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        std_y = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,std_y)
        sig2 = 0
        #np.testing.assert_allclose(reg.sig2,sig2)
        hth = np.array([[    49.        ,    704.371999  ,    139.75      ], [   704.371999  ,  11686.67338121,   2246.12800625], [   139.75      ,   2246.12800625,    498.5851]])
        np.testing.assert_allclose(reg.hth,hth,RTOL)

class GM_Endog_Error_Hom_Tester(unittest.TestCase):
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
        reg = HOM.GM_Endog_Error_Hom(self.y, self.X, self.yd, self.q, self.w, A1='hom_sc')
        np.testing.assert_allclose(reg.y[0],np.array([ 80.467003]),RTOL)
        x = np.array([  1.     ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        h = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_allclose(reg.h[0],h,RTOL)
        yend = np.array([ 15.72598])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        q = np.array([ 5.03])
        np.testing.assert_allclose(reg.q[0],q,RTOL)
        betas = np.array([[ 55.36575166], [  0.46432416], [ -0.66904404], [  0.43205526]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 26.55390939])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        np.testing.assert_allclose(reg.e_filtered[0],np.array([ 31.74114306]),RTOL)
        predy = np.array([ 53.91309361])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        np.testing.assert_allclose(reg.n,49,RTOL)
        np.testing.assert_allclose(reg.k,3,RTOL)
        vm = np.array([[  5.52064057e+02,  -1.61264555e+01,  -8.86360735e+00, 1.04251912e+00], [ -1.61264555e+01,   5.44898242e-01, 2.39518645e-01, -1.88092950e-02], [ -8.86360735e+00,   2.39518645e-01, 1.55501840e-01, -2.18638648e-02], [  1.04251912e+00, -1.88092950e-02, -2.18638648e-02, 3.71222222e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        np.testing.assert_allclose(reg.iteration,its,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        std_y = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,std_y)
        pr2 = 0.34647366525657419
        np.testing.assert_allclose(reg.pr2,pr2)
        sig2 = 190.59435238060928
        np.testing.assert_allclose(reg.sig2,sig2)
        #std_err
        std_err = np.array([ 23.49604343,   0.73817223,   0.39433722, 0.19267128])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([[ 2.35638617,  0.01845372], [ 0.62901874,  0.52933679], [-1.69662923,  0.08976678], [ 2.24244556,  0.02493259]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)


class BaseGM_Combo_Hom_Tester(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
    def test_model(self):
        yd2, q2 = pysal.spreg.utils.set_endog(self.y, self.X, self.w, None, None, 1, True)
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        reg = HOM.BaseGM_Combo_Hom(self.y, self.X, yend=yd2, q=q2, w=self.w.sparse, A1='hom_sc')
        np.testing.assert_allclose(reg.y[0],np.array([80.467003]),RTOL)
        x = np.array([  1.     ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        betas = np.array([[ 10.12541428], [  1.56832263], [  0.15132076], [  0.21033397]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        np.testing.assert_allclose(reg.u[0],np.array([34.3450723]),RTOL)
        np.testing.assert_allclose(reg.e_filtered[0],np.array([ 36.6149682]),RTOL)
        np.testing.assert_allclose(reg.predy[0],np.array([ 46.1219307]),RTOL)
        np.testing.assert_allclose(reg.n,49,RTOL)
        np.testing.assert_allclose(reg.k,3,RTOL)
        vm = np.array([[  2.33694742e+02,  -6.66856869e-01,  -5.58304254e+00, 4.85488380e+00], [ -6.66856869e-01,   1.94241504e-01, -5.42327138e-02, 5.37225570e-02], [ -5.58304254e+00,  -5.42327138e-02, 1.63860721e-01, -1.44425498e-01], [  4.85488380e+00, 5.37225570e-02, -1.44425498e-01, 1.78622255e-01]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        z = np.array([  1.       ,  19.531    ,  35.4585005])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        h = np.array([  1.   ,  19.531,  18.594])
        np.testing.assert_allclose(reg.h[0],h,RTOL)
        yend = np.array([ 35.4585005])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        q = np.array([ 18.594])
        np.testing.assert_allclose(reg.q[0],q,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        its = 1
        np.testing.assert_allclose(reg.iteration,its,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        std_y = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,std_y)
        sig2 = 232.22680651270042
        #np.testing.assert_allclose(reg.sig2,sig2)
        np.testing.assert_allclose(reg.sig2,sig2)
        hth = np.array([[    49.        ,    704.371999  ,    724.7435916 ], [   704.371999  ,  11686.67338121,  11092.519988  ], [   724.7435916 ,  11092.519988  , 11614.62257048]])
        np.testing.assert_allclose(reg.hth,hth,RTOL)


class GM_Combo_Hom_Tester(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
    def test_model(self):
        reg = HOM.GM_Combo_Hom(self.y, self.X, w=self.w, A1='hom_sc')
        np.testing.assert_allclose(reg.y[0],np.array([80.467003]),RTOL)
        x = np.array([  1.     ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        betas = np.array([[ 10.12541428], [  1.56832263], [  0.15132076], [  0.21033397]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        np.testing.assert_allclose(reg.u[0],np.array([34.3450723]),RTOL)
        np.testing.assert_allclose(reg.e_filtered[0],np.array([ 36.6149682]),RTOL)
        np.testing.assert_allclose(reg.e_pred[0],np.array([ 32.90372983]),RTOL)
        np.testing.assert_allclose(reg.predy[0],np.array([ 46.1219307]),RTOL)
        np.testing.assert_allclose(reg.predy_e[0],np.array([47.56327317]),RTOL)
        np.testing.assert_allclose(reg.n,49,RTOL)
        np.testing.assert_allclose(reg.k,3,RTOL)
        z = np.array([  1.       ,  19.531    ,  35.4585005])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        h = np.array([  1.   ,  19.531,  18.594])
        np.testing.assert_allclose(reg.h[0],h,RTOL)
        yend = np.array([ 35.4585005])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        q = np.array([ 18.594])
        np.testing.assert_allclose(reg.q[0],q,RTOL)
        i_s = 'Maximum number of iterations reached.'
        np.testing.assert_string_equal(reg.iter_stop,i_s)
        np.testing.assert_allclose(reg.iteration,1,RTOL)
        my = 38.436224469387746
        np.testing.assert_allclose(reg.mean_y,my)
        std_y = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,std_y)
        pr2 = 0.28379825632694394
        np.testing.assert_allclose(reg.pr2,pr2)
        pr2_e = 0.25082892555141506
        np.testing.assert_allclose(reg.pr2_e,pr2_e)
        sig2 = 232.22680651270042
        #np.testing.assert_allclose(reg.sig2, sig2)
        np.testing.assert_allclose(reg.sig2, sig2)
        std_err = np.array([ 15.28707761,   0.44072838,   0.40479714, 0.42263726])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([[  6.62351206e-01,   5.07746167e-01], [  3.55847888e+00,   3.73008780e-04], [  3.73818749e-01,   7.08539170e-01], [  4.97670189e-01,   6.18716523e-01]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)
        vm = np.array([[  2.33694742e+02,  -6.66856869e-01,  -5.58304254e+00, 4.85488380e+00], [ -6.66856869e-01,   1.94241504e-01, -5.42327138e-02, 5.37225570e-02], [ -5.58304254e+00,  -5.42327138e-02, 1.63860721e-01, -1.44425498e-01], [  4.85488380e+00, 5.37225570e-02, -1.44425498e-01, 1.78622255e-01]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

suite = unittest.TestSuite()
test_classes = [BaseGM_Error_Hom_Tester, GM_Error_Hom_Tester,\
        BaseGM_Endog_Error_Hom_Tester, GM_Endog_Error_Hom_Tester, \
        BaseGM_Combo_Hom_Tester, GM_Combo_Hom_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)

