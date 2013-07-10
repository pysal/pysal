import unittest
import pysal
import numpy as np
from pysal.spreg import error_sp as SP

class TestBaseGMError(unittest.TestCase):
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
        reg = SP.BaseGM_Error(self.y, self.X, self.w.sparse)
        betas = np.array([[ 47.94371455], [  0.70598088], [ -0.55571746], [  0.37230161]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 27.4739775])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        predy = np.array([ 52.9930255])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n,6)
        k = 3
        self.assertAlmostEqual(reg.k,k,6)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,6)
        e = np.array([ 31.89620319])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,6)
        predy = np.array([ 52.9930255])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        my = 38.43622446938776
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 18.466069465206047
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([[  1.51884943e+02,  -5.37622793e+00,  -1.86970286e+00], [ -5.37622793e+00,   2.48972661e-01,   5.26564244e-02], [ -1.86970286e+00,   5.26564244e-02, 3.18930650e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        sig2 = 191.73716465732355
        self.assertAlmostEqual(reg.sig2,sig2,5)

class TestGMError(unittest.TestCase):
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
        reg = SP.GM_Error(self.y, self.X, self.w)
        betas = np.array([[ 47.94371455], [  0.70598088], [ -0.55571746], [  0.37230161]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 27.4739775])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        predy = np.array([ 52.9930255])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n,6)
        k = 3
        self.assertAlmostEqual(reg.k,k,6)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,6)
        e = np.array([ 31.89620319])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,6)
        predy = np.array([ 52.9930255])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        my = 38.43622446938776
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 18.466069465206047
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([[  1.51884943e+02,  -5.37622793e+00,  -1.86970286e+00], [ -5.37622793e+00,   2.48972661e-01,   5.26564244e-02], [ -1.86970286e+00,   5.26564244e-02, 3.18930650e-02]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        sig2 = 191.73716465732355
        self.assertAlmostEqual(reg.sig2,sig2,5)
        pr2 = 0.3495097406012179
        self.assertAlmostEqual(reg.pr2,pr2)
        std_err = np.array([ 12.32416094,   0.4989716 ,   0.1785863 ])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,6)
        z_stat = np.array([[  3.89022140e+00,   1.00152805e-04], [  1.41487186e+00,   1.57106070e-01], [ -3.11175868e+00,   1.85976455e-03]])
        np.testing.assert_array_almost_equal(reg.z_stat,z_stat,6)

class TestBaseGMEndogError(unittest.TestCase):
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
        reg = SP.BaseGM_Endog_Error(self.y, self.X, self.yd, self.q, self.w.sparse)
        betas = np.array([[ 55.36095292], [  0.46411479], [ -0.66883535], [  0.38989939]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 26.55951566])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e = np.array([ 31.23925425])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,6)
        predy = np.array([ 53.9074875])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n)
        k = 3
        self.assertAlmostEqual(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_array_almost_equal(reg.x[0],x,6)
        yend = np.array([  15.72598])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,6)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.z[0],z,6)
        my = 38.43622446938776
        self.assertAlmostEqual(reg.mean_y,my)
        #std_y
        sy = 18.466069465206047
        self.assertAlmostEqual(reg.std_y,sy)
        #vm
        vm = np.array([[  5.29156458e+02,  -1.57833384e+01,  -8.38016915e+00], [ -1.57833384e+01,   5.40234656e-01,   2.31119606e-01], [ -8.38016915e+00,   2.31119606e-01, 1.44976477e-01]])
        np.testing.assert_array_almost_equal(reg.vm,vm,5)
        sig2 = 192.50040382591442
        self.assertAlmostEqual(reg.sig2,sig2,5)

class TestGMEndogError(unittest.TestCase):
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
        reg = SP.GM_Endog_Error(self.y, self.X, self.yd, self.q, self.w)
        betas = np.array([[ 55.36095292], [  0.46411479], [ -0.66883535], [  0.38989939]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 26.55951566])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e = np.array([ 31.23925425])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,6)
        predy = np.array([ 53.9074875])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n)
        k = 3
        self.assertAlmostEqual(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_array_almost_equal(reg.x[0],x,6)
        yend = np.array([  15.72598])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,6)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.z[0],z,6)
        my = 38.43622446938776
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 18.466069465206047
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([[  5.29156458e+02,  -1.57833384e+01,  -8.38016915e+00], [ -1.57833384e+01,   5.40234656e-01,   2.31119606e-01], [ -8.38016915e+00,   2.31119606e-01, 1.44976477e-01]])
        np.testing.assert_array_almost_equal(reg.vm,vm,5)
        pr2 = 0.346472557570858
        self.assertAlmostEqual(reg.pr2,pr2)
        sig2 = 192.50040382591442
        self.assertAlmostEqual(reg.sig2,sig2,5)
        std_err = np.array([ 23.003401  ,   0.73500657,   0.38075777])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,6)
        z_stat = np.array([[ 2.40664208,  0.01609994], [ 0.63144305,  0.52775088], [-1.75659016,  0.07898769]])
        np.testing.assert_array_almost_equal(reg.z_stat,z_stat,6)

class TestBaseGMCombo(unittest.TestCase):
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
        reg = SP.BaseGM_Combo(self.y, self.X, yend=yd2, q=q2, w=self.w.sparse)
        betas = np.array([[ 57.61123461],[  0.73441314], [ -0.59459416], [ -0.21762921], [  0.54732051]])
        np.testing.assert_array_almost_equal(reg.betas,betas,5)
        u = np.array([ 25.57932637])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e_filtered = np.array([ 31.65374945])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e_filtered,5)
        predy = np.array([ 54.88767663])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n)
        k = 4
        self.assertAlmostEqual(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,6)
        yend = np.array([  35.4585005])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,6)
        z = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005])
        np.testing.assert_array_almost_equal(reg.z[0],z,6)
        my = 38.43622446938776
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 18.466069465206047
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([[  5.22438894e+02,  -6.07257246e+00,  -1.91428892e+00, -8.97134337e+00], [ -6.07257246e+00,   2.38012836e-01,   4.70160750e-02, 2.80964005e-02], [ -1.91428911e+00,  4.70160773e-02,  3.20924154e-02, 3.14968682e-03], [ -8.97134237e+00,  2.80964005e-02,  3.14968682e-03, 2.15753890e-01]])
        np.testing.assert_array_almost_equal(reg.vm,vm,4)
        sig2 = 181.78650186468832
        self.assertAlmostEqual(reg.sig2,sig2,4)

class TestGMCombo(unittest.TestCase):
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
        reg = SP.GM_Combo(self.y, self.X, w=self.w)
        e_reduced = np.array([ 28.18617481])
        np.testing.assert_array_almost_equal(reg.e_pred[0],e_reduced,6)
        predy_e = np.array([ 52.28082782])
        np.testing.assert_array_almost_equal(reg.predy_e[0],predy_e,6)
        betas = np.array([[ 57.61123515],[  0.73441313], [ -0.59459416], [ -0.21762921], [  0.54732051]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        u = np.array([ 25.57932637])
        np.testing.assert_array_almost_equal(reg.u[0],u,6)
        e_filtered = np.array([ 31.65374945])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e_filtered,5)
        predy = np.array([ 54.88767685])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n)
        k = 4
        self.assertAlmostEqual(reg.k,k)
        y = np.array([ 80.467003])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_array_almost_equal(reg.x[0],x,6)
        yend = np.array([  35.4585005])
        np.testing.assert_array_almost_equal(reg.yend[0],yend,6)
        z = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005])
        np.testing.assert_array_almost_equal(reg.z[0],z,6)
        my = 38.43622446938776
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 18.466069465206047
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([[ 5.22438894e+02,  -6.07257246e+00,  -1.91428892e+00, -8.97134337e+00], [ -6.07257218e+00,  2.38012839e-01,  4.70160773e-02, 2.80964005e-02], [ -1.91428911e+00,  4.70160773e-02,  3.20924154e-02, 3.14968682e-03], [ -8.97134237e+00,  2.80964005e-02,  3.14968682e-03, 2.15753890e-01]])
        np.testing.assert_array_almost_equal(reg.vm,vm,4)
        sig2 = 181.78650186468832
        self.assertAlmostEqual(reg.sig2,sig2,4)
        pr2 = 0.3018280166937799
        self.assertAlmostEqual(reg.pr2,pr2)
        pr2_e = 0.3561355587000738
        self.assertAlmostEqual(reg.pr2_e,pr2_e)
        std_err = np.array([ 22.85692222,  0.48786559,  0.17914356,  0.46449318])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,5)
        z_stat = np.array([[  2.52051597e+00,   1.17182922e-02], [  1.50535954e+00,   1.32231664e-01], [ -3.31909311e+00,   9.03103123e-04], [ -4.68530506e-01,   6.39405261e-01]])
        np.testing.assert_array_almost_equal(reg.z_stat,z_stat,6)

if __name__ == '__main__':
    unittest.main()
