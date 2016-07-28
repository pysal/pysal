import unittest
import pysal
import scipy
import numpy as np
from pysal.spreg.ml_error import ML_Error
from pysal.spreg import utils
from pysal.common import RTOL, ATOL
from warnings import warn as Warn
from skip import SKIP


@unittest.skipIf(SKIP,
        "Skipping MLError Tests")
class TestMLError(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path("south.dbf"),'r')
        self.y_name = "HR90"
        self.y = np.array(db.by_col(self.y_name))
        self.y.shape = (len(self.y),1)
        self.x_names = ["RD90","PS90","UE90","DV90"]
        self.x = np.array([db.by_col(var) for var in self.x_names]).T
        ww = pysal.open(pysal.examples.get_path("south_q.gal"))
        self.w = ww.read()
        ww.close()
        self.w.transform = 'r'

    def _estimate_and_compare(self, method='FULL', RTOL=RTOL):
        reg = ML_Error(self.y,self.x,w=self.w,name_y=self.y_name,name_x=self.x_names,\
               name_w="south_q.gal", method=method)
        betas = np.array([[ 6.1492], [ 4.4024], [ 1.7784], [-0.3781], [ 0.4858], [ 0.2991]])
        Warn('Running higher-tolerance tests in test_ml_error.py')
        np.testing.assert_allclose(reg.betas,betas,RTOL + .0001)
        u = np.array([-5.97649777])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 6.92258051])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 1412
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 5
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 0.94608274])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([ 1.        , -0.39902838,  0.89645344,  6.85780705,  7.2636377 ])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        e = np.array([-4.92843327])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        my = 9.5492931620846928
        np.testing.assert_allclose(reg.mean_y,my)
        sy = 7.0388508798387219
        np.testing.assert_allclose(reg.std_y,sy)
        vm = np.array([ 1.06476526,  0.05548248,  0.04544514,  0.00614425,  0.01481356,
        0.00143001])
        np.testing.assert_allclose(reg.vm.diagonal(),vm,RTOL)
        sig2 = np.array([[ 32.40685441]])
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.3057664820364818
        np.testing.assert_allclose(reg.pr2,pr2)
        std_err = np.array([ 1.03187463,  0.23554719,  0.21317867,  0.07838525,  0.12171098,
        0.03781546])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = [(5.9592751097983534, 2.5335926307459251e-09),
 (18.690182928021841, 5.9508619446611137e-78),
 (8.3421632936950338, 7.2943630281051907e-17),
 (-4.8232686291115678, 1.4122456582517099e-06),
 (3.9913060809142995, 6.5710406838016854e-05),
 (7.9088780724028922, 2.5971882547279339e-15)]
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL, atol=ATOL)
        logll = -4471.407066887894
        np.testing.assert_allclose(reg.logll,logll,RTOL)
        aic = 8952.8141337757879
        np.testing.assert_allclose(reg.aic,aic,RTOL)
        schwarz = 8979.0779458660545
        np.testing.assert_allclose(reg.schwarz,schwarz,RTOL)


    def test_dense(self):
        self._estimate_and_compare(method='FULL')

    def test_LU(self):
        self._estimate_and_compare(method='LU', RTOL=RTOL*10)

    def test_ord(self):
        reg = ML_Error(self.y, self.x, w=self.w,
                     name_y=self.y_name, name_x=self.x_names,
                     name_w='south_q.gal',  method='ORD')
        betas = np.array([[ 6.1492], [ 4.4024], [ 1.7784], [-0.3781], [ 0.4858], [ 0.2991]])
        Warn('Running higher-tolerance tests in test_ml_error.py')
        np.testing.assert_allclose(reg.betas,betas,RTOL + .0001)
        u = np.array([-5.97649777])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 6.92258051])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 1412
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 5
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 0.94608274])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([ 1.        , -0.39902838,  0.89645344,  6.85780705,  7.2636377 ])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        e = np.array([-4.92843327])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        my = 9.5492931620846928
        np.testing.assert_allclose(reg.mean_y,my)
        sy = 7.0388508798387219
        np.testing.assert_allclose(reg.std_y,sy)
        vm = np.array([ 1.06476526,  0.05548248,  0.04544514,  0.00614425,  0.01481356,
        0.001501])
        np.testing.assert_allclose(reg.vm.diagonal(),vm,RTOL * 10)
        sig2 = np.array([[ 32.40685441]])
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.3057664820364818
        np.testing.assert_allclose(reg.pr2,pr2)
        std_err = np.array([ 1.03187463,  0.23554719,  0.21317867,  0.07838525,  0.12171098,
        0.038744])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL*10)
        z_stat = [(5.95927510, 2.5335927e-09),
                  (18.6901829, 5.9508630e-78),
                  (8.34216329, 7.2943634e-17),
                  (-4.8232686, 1.4122457e-06),
                  (3.99130608, 6.5710407e-05),
                  (7.71923784, 1.1702739e-14)]
        np.testing.assert_allclose(reg.z_stat,z_stat,rtol=RTOL, atol=ATOL)
        logll = -4471.407066887894
        np.testing.assert_allclose(reg.logll,logll,RTOL)
        aic = 8952.8141337757879
        np.testing.assert_allclose(reg.aic,aic,RTOL)
        schwarz = 8979.0779458660545
        np.testing.assert_allclose(reg.schwarz,schwarz,RTOL)

if __name__ == '__main__':
    unittest.main()
