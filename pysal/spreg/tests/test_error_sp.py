import unittest
import scipy
import pysal
import numpy as np
from pysal.spreg import error_sp as SP
from pysal.common import RTOL

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
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 27.4739775])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 52.9930255])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        e = np.array([ 31.89620319])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([ 52.9930255])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([[  1.51884943e+02,  -5.37622793e+00,  -1.86970286e+00], [ -5.37622793e+00,   2.48972661e-01,   5.26564244e-02], [ -1.86970286e+00,   5.26564244e-02, 3.18930650e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        sig2 = 191.73716465732355
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)

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
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 27.4739775])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 52.9930255])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        e = np.array([ 31.89620319])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([ 52.9930255])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([[  1.51884943e+02,  -5.37622793e+00,  -1.86970286e+00], [ -5.37622793e+00,   2.48972661e-01,   5.26564244e-02], [ -1.86970286e+00,   5.26564244e-02, 3.18930650e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        sig2 = 191.73716465732355
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.3495097406012179
        np.testing.assert_allclose(reg.pr2,pr2)
        std_err = np.array([ 12.32416094,   0.4989716 ,   0.1785863 ])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([[  3.89022140e+00,   1.00152805e-04], [  1.41487186e+00,   1.57106070e-01], [ -3.11175868e+00,   1.85976455e-03]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)

@unittest.skipIf(int(scipy.__version__.split(".")[1]) < 11,
"Maximum Likelihood requires SciPy version 11 or newer.")
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
        #raising a warning causes a failure...
        print('Running reduced precision test in L120 of test_error_sp.py')
        np.testing.assert_allclose(reg.betas,betas,RTOL+.0001)
        u = np.array([ 26.55951566])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e = np.array([ 31.23925425])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([ 53.9074875])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([  15.72598])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        #std_y
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        #vm
        vm = np.array([[  5.29158422e+02,  -1.57833675e+01,  -8.38021080e+00],
       [ -1.57833675e+01,   5.40235041e-01,   2.31120327e-01],
       [ -8.38021080e+00,   2.31120327e-01,   1.44977385e-01]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        sig2 = 192.50022721929574
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)

@unittest.skipIf(int(scipy.__version__.split(".")[1]) < 11,
"Maximum Likelihood requires SciPy version 11 or newer.")
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
        print('Running reduced precision test in L175 of test_error_sp.py')
        np.testing.assert_allclose(reg.betas,betas,RTOL +.0001)
        u = np.array([ 26.55951566])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e = np.array([ 31.23925425])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([ 53.9074875])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([  15.72598])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([[  5.29158422e+02,  -1.57833675e+01,  -8.38021080e+00],
       [ -1.57833675e+01,   5.40235041e-01,   2.31120327e-01],
       [ -8.38021080e+00,   2.31120327e-01,   1.44977385e-01]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        pr2 = 0.346472557570858
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        sig2 = 192.50022721929574
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        std_err = np.array([ 23.003401  ,   0.73500657,   0.38075777])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([[ 2.40664208,  0.01609994], [ 0.63144305,  0.52775088], [-1.75659016,  0.07898769]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)

@unittest.skipIf(int(scipy.__version__.split(".")[1]) < 11,
"Maximum Likelihood requires SciPy version 11 or newer.")
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
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 25.57932637])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e_filtered = np.array([ 31.65374945])
        np.testing.assert_allclose(reg.e_filtered[0],e_filtered,RTOL)
        predy = np.array([ 54.88767663])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 4
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([  35.4585005])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([  5.22438365e+02,   2.38012873e-01,   3.20924172e-02,
         2.15753599e-01])
        np.testing.assert_allclose(np.diag(reg.vm),vm,RTOL)
        sig2 = 181.78650186468832
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)

@unittest.skipIf(int(scipy.__version__.split(".")[1]) < 11,
"Maximum Likelihood requires SciPy version 11 or newer.")
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
        np.testing.assert_allclose(reg.e_pred[0],e_reduced,RTOL)
        predy_e = np.array([ 52.28082782])
        np.testing.assert_allclose(reg.predy_e[0],predy_e,RTOL)
        betas = np.array([[ 57.61123515],[  0.73441313], [ -0.59459416], [ -0.21762921], [  0.54732051]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 25.57932637])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e_filtered = np.array([ 31.65374945])
        np.testing.assert_allclose(reg.e_filtered[0],e_filtered,RTOL)
        predy = np.array([ 54.88767685])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 4
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([  35.4585005])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my)
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy)
        vm = np.array([  5.22438333e+02,   2.38012875e-01,   3.20924173e-02,
         2.15753579e-01])
        np.testing.assert_allclose(np.diag(reg.vm),vm,RTOL)
        sig2 = 181.78650186468832
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.3018280166937799
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        pr2_e = 0.3561355586759414
        np.testing.assert_allclose(reg.pr2_e,pr2_e,RTOL)
        std_err = np.array([ 22.85692222,  0.48786559,  0.17914356,  0.46449318])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([[  2.52051597e+00,   1.17182922e-02], [  1.50535954e+00,   1.32231664e-01], [ -3.31909311e+00,   9.03103123e-04], [ -4.68530506e-01,   6.39405261e-01]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)

if __name__ == '__main__':
    unittest.main()
