'''
Unittests for spreg.error_sp_hom module

ToDo:
    * BaseGM_Combo_Hom
    * GM_Combo_Hom
'''
import unittest
import pysal
from econometrics import error_sp_hom as HOM
import numpy as np

class BaseGM_Error_Hom_Tester(unittest.TestCase):
    def setUp(self):
        db=pysal.open("../examples/columbus.dbf","r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile("../examples/columbus.shp")
        self.w.transform = 'r'
    def test_model(self):
        reg = HOM.BaseGM_Error_Hom(self.y, self.X, self.w, A1='hom_sc')
        np.testing.assert_array_almost_equal(reg.y[0],np.array([80.467003]),7)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        betas = np.array([[ 47.9478524 ], [  0.70633223], [ -0.55595633], [  0.41288558]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        np.testing.assert_array_almost_equal(reg.u[0],np.array([27.46673388]),7)
        np.testing.assert_array_almost_equal(reg.predy[0],np.array([ 53.00026912]),7)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  1.51340716e+02,  -5.29057478e+00,  -1.85654536e+00, -1.17178040e-01], [ -5.29057478e+00,   2.46669603e-01,   5.14259061e-02, 1.56428142e-02], [ -1.85654536e+00,   5.14259061e-02,   3.20510556e-02, -2.91863599e-03], [ -1.17178040e-01,   1.56428142e-02,  -2.91863599e-03, 1.64978203e+00]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)

class GM_Error_Hom_Tester(unittest.TestCase):
    def setUp(self):
        db=pysal.open("../examples/columbus.dbf","r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile("../examples/columbus.shp")
        self.w.transform = 'r'
    def test_model(self):
        reg = HOM.GM_Error_Hom(self.y, self.X, self.w, A1='hom_sc')
        np.testing.assert_array_almost_equal(reg.y[0],np.array([80.467003]),7)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,7)
        betas = np.array([[ 47.9478524 ], [  0.70633223], [ -0.55595633], [  0.41288558]])
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        np.testing.assert_array_almost_equal(reg.u[0],np.array([27.46673388]),7)
        np.testing.assert_array_almost_equal(reg.predy[0],np.array([ 53.00026912]),7)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  1.51340716e+02,  -5.29057478e+00,  -1.85654536e+00, -1.17178040e-01], [ -5.29057478e+00,   2.46669603e-01,   5.14259061e-02, 1.56428142e-02], [ -1.85654536e+00,   5.14259061e-02,   3.20510556e-02, -2.91863599e-03], [ -1.17178040e-01,   1.56428142e-02,  -2.91863599e-03, 1.64978203e+00]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)


class BaseGM_Endog_Error_Hom_Tester(unittest.TestCase):
    def setUp(self):
        db=pysal.open("../examples/columbus.dbf","r")
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
        self.w = pysal.rook_from_shapefile("../examples/columbus.shp")
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
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        u = np.array([ 26.55390939])
        np.testing.assert_array_almost_equal(reg.u[0],u,7)
        predy = np.array([ 53.91309361])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,7)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  5.52064300e+02,  -1.61264593e+01,  -8.86361243e+00, 1.04251892e+00], [ -1.61264593e+01,   5.44898297e-01,   2.39518735e-01, -1.88092866e-02], [ -8.86361243e+00,   2.39518735e-01,   1.55501950e-01, -2.18638628e-02], [  1.04251892e+00,  -1.88092866e-02,  -2.18638628e-02, 3.71222329e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)

class GM_Endog_Error_Hom_Tester(unittest.TestCase):
    def setUp(self):
        db=pysal.open("../examples/columbus.dbf","r")
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
        self.w = pysal.rook_from_shapefile("../examples/columbus.shp")
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
        np.testing.assert_array_almost_equal(reg.betas,betas,7)
        u = np.array([ 26.55390939])
        np.testing.assert_array_almost_equal(reg.u[0],u,7)
        predy = np.array([ 53.91309361])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,7)
        self.assertAlmostEquals(reg.n,49,7)
        self.assertAlmostEquals(reg.k,3,7)
        vm = np.array([[  5.52064300e+02,  -1.61264593e+01,  -8.86361243e+00, 1.04251892e+00], [ -1.61264593e+01,   5.44898297e-01,   2.39518735e-01, -1.88092866e-02], [ -8.86361243e+00,   2.39518735e-01,   1.55501950e-01, -2.18638628e-02], [  1.04251892e+00,  -1.88092866e-02,  -2.18638628e-02, 3.71222329e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)



suite = unittest.TestSuite()
test_classes = [BaseGM_Error_Hom_Tester, GM_Error_Hom_Tester,\
        BaseGM_Endog_Error_Hom_Tester, GM_Endog_Error_Hom_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)

