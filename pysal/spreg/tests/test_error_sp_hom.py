'''
Unittests for spreg.error_sp_hom module

ToDo:
    * BaseGM_Combo_Hom
    * GM_Combo_Hom
'''
import unittest
import pysal
from pysal.spreg import error_sp_hom as HOM
import numpy as np

class BaseGM_Error_Hom_Tester(unittest.TestCase):
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
        reg = HOM.BaseGM_Error_Hom(self.y, self.X, self.w, A1='hom_sc')
        np.testing.assert_array_almost_equal(reg.y[0],np.array([80.467003]),7)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        betas = np.array([[ 47.9478524 ], [  0.70633223], [ -0.55595633], [  0.41288558]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        np.testing.assert_array_almost_equal(reg.u[0],np.array([27.466734]),6)
        np.testing.assert_array_almost_equal(reg.predy[0],np.array([ 53.000269]),6)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  1.51340717e+02,  -5.29057506e+00,  -1.85654540e+00, -2.39139054e-03], [ -5.29057506e+00,   2.46669610e-01, 5.14259101e-02, 3.19241302e-04], [ -1.85654540e+00,   5.14259101e-02, 3.20510550e-02,  -5.95640240e-05], [ -2.39139054e-03,   3.19241302e-04, -5.95640240e-05,  3.36690159e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)

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
        np.testing.assert_array_almost_equal(reg.y[0],np.array([80.467003]),7)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        betas = np.array([[ 47.9478524 ], [  0.70633223], [ -0.55595633], [  0.41288558]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        np.testing.assert_array_almost_equal(reg.u[0],np.array([27.46673388]),6)
        np.testing.assert_array_almost_equal(reg.predy[0],np.array([ 53.00026912]),6)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  1.51340717e+02,  -5.29057506e+00,  -1.85654540e+00, -2.39139054e-03], [ -5.29057506e+00,   2.46669610e-01, 5.14259101e-02, 3.19241302e-04], [ -1.85654540e+00,   5.14259101e-02, 3.20510550e-02,  -5.95640240e-05], [ -2.39139054e-03,   3.19241302e-04, -5.95640240e-05,  3.36690159e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)


class BaseGM_Endog_Error_Hom_Tester(unittest.TestCase):
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
        reg = HOM.BaseGM_Endog_Error_Hom(self.y, self.X, self.yd, self.q, self.w, A1='hom_sc')
        np.testing.assert_array_almost_equal(reg.y[0],np.array([ 80.467003]),7)
        x = np.array([  1.     ,  19.531])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.z[0],z,7)
        h = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_array_almost_equal(reg.h[0],h,7)
        yend = np.array([ 15.72598])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,7)
        q = np.array([ 5.03])
        np.testing.assert_array_almost_equal(reg.q[0],q,7)
        betas = np.array([[ 55.36575166], [  0.46432416], [ -0.66904404], [  0.43205526]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 26.55390939])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        predy = np.array([ 53.91309361])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  5.52064057e+02,  -1.61264555e+01,  -8.86360735e+00, 1.04251912e+00], [ -1.61264555e+01,   5.44898242e-01, 2.39518645e-01, -1.88092950e-02], [ -8.86360735e+00,   2.39518645e-01, 1.55501840e-01, -2.18638648e-02], [  1.04251912e+00, -1.88092950e-02, -2.18638648e-02, 3.71222222e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)

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
        np.testing.assert_array_almost_equal(reg.y[0],np.array([ 80.467003]),7)
        x = np.array([  1.     ,  19.531])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.z[0],z,7)
        h = np.array([  1.   ,  19.531,   5.03 ])
        np.testing.assert_array_almost_equal(reg.h[0],h,7)
        yend = np.array([ 15.72598])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,7)
        q = np.array([ 5.03])
        np.testing.assert_array_almost_equal(reg.q[0],q,7)
        betas = np.array([[ 55.36575166], [  0.46432416], [ -0.66904404], [  0.43205526]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 26.55390939])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        predy = np.array([ 53.91309361])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  5.52064057e+02,  -1.61264555e+01,  -8.86360735e+00, 1.04251912e+00], [ -1.61264555e+01,   5.44898242e-01, 2.39518645e-01, -1.88092950e-02], [ -8.86360735e+00,   2.39518645e-01, 1.55501840e-01, -2.18638648e-02], [  1.04251912e+00, -1.88092950e-02, -2.18638648e-02, 3.71222222e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)


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
        reg = HOM.BaseGM_Combo_Hom(self.y, self.X, w=self.w, A1='hom_sc')
        np.testing.assert_array_almost_equal(reg.y[0],np.array([80.467003]),7)
        x = np.array([  1.     ,  19.531])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        betas = np.array([[ 10.12541428], [  1.56832263], [  0.15132076], [  0.21033397]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        np.testing.assert_array_almost_equal(reg.u[0],np.array([34.3450723]),7)
        np.testing.assert_array_almost_equal(reg.predy[0],np.array([ 46.1219307]),7)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  2.33694742e+02,  -6.66856869e-01,  -5.58304254e+00, 4.85488380e+00], [ -6.66856869e-01,   1.94241504e-01, -5.42327138e-02, 5.37225570e-02], [ -5.58304254e+00,  -5.42327138e-02, 1.63860721e-01, -1.44425498e-01], [  4.85488380e+00, 5.37225570e-02, -1.44425498e-01, 1.78622255e-01]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)

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
        np.testing.assert_array_almost_equal(reg.y[0],np.array([80.467003]),7)
        x = np.array([  1.     ,  19.531])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        betas = np.array([[ 10.12541428], [  1.56832263], [  0.15132076], [  0.21033397]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        np.testing.assert_array_almost_equal(reg.u[0],np.array([34.3450723]),7)
        np.testing.assert_array_almost_equal(reg.predy[0],np.array([ 46.1219307]),7)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  2.33694742e+02,  -6.66856869e-01,  -5.58304254e+00, 4.85488380e+00], [ -6.66856869e-01,   1.94241504e-01, -5.42327138e-02, 5.37225570e-02], [ -5.58304254e+00,  -5.42327138e-02, 1.63860721e-01, -1.44425498e-01], [  4.85488380e+00, 5.37225570e-02, -1.44425498e-01, 1.78622255e-01]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)

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

