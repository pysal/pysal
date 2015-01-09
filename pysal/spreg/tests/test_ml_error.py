import unittest
import pysal
import scipy
import numpy as np
from pysal.spreg.ml_error import ML_Error
from pysal.spreg import utils

@unittest.skipIf(int(scipy.__version__.split(".")[1]) < 11,
        "Max Likelihood requires SciPy version 11 or newer.")
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

    def test_model(self):
        reg = ML_Error(self.y,self.x,w=self.w,name_y=self.y_name,name_x=self.x_names,\
               name_w="south_q.gal")
        betas = np.array([[ 6.1492], [ 4.4024], [ 1.7784], [-0.3781], [ 0.4858], [ 0.2991]])
        np.testing.assert_array_almost_equal(reg.betas,betas,4)
        u = np.array([-5.97649777])
        np.testing.assert_array_almost_equal(reg.u[0],u,4)
        predy = np.array([ 6.92258051])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,4)
        n = 1412
        self.assertAlmostEqual(reg.n,n,4)
        k = 5
        self.assertAlmostEqual(reg.k,k,4)
        y = np.array([ 0.94608274])
        np.testing.assert_array_almost_equal(reg.y[0],y,4)
        x = np.array([ 1.        , -0.39902838,  0.89645344,  6.85780705,  7.2636377 ])
        np.testing.assert_array_almost_equal(reg.x[0],x,4)
        e = np.array([-4.92843327])
        np.testing.assert_array_almost_equal(reg.e_filtered[0],e,4)
        my = 9.5492931620846928
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 7.0388508798387219
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([ 1.06476526,  0.05548248,  0.04544514,  0.00614425,  0.01481356,
        0.00143001])
        np.testing.assert_array_almost_equal(reg.vm.diagonal(),vm,4)
        sig2 = [ 32.40685441]
        self.assertAlmostEqual(reg.sig2,sig2,4)
        pr2 = 0.3057664820364818
        self.assertAlmostEqual(reg.pr2,pr2)
        std_err = np.array([ 1.03187463,  0.23554719,  0.21317867,  0.07838525,  0.12171098,
        0.03781546])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,4)
        z_stat = [(5.9592751097983534, 2.5335926307459251e-09),
 (18.690182928021841, 5.9508619446611137e-78),
 (8.3421632936950338, 7.2943630281051907e-17),
 (-4.8232686291115678, 1.4122456582517099e-06),
 (3.9913060809142995, 6.5710406838016854e-05),
 (7.9088780724028922, 2.5971882547279339e-15)]
        np.testing.assert_array_almost_equal(reg.z_stat,z_stat,4)
        logll = -4471.407066887894
        self.assertAlmostEqual(reg.logll,logll,4)
        aic = 8952.8141337757879
        self.assertAlmostEqual(reg.aic,aic,4)
        schwarz = 8979.0779458660545
        self.assertAlmostEqual(reg.schwarz,schwarz,4)

if __name__ == '__main__':
    unittest.main()
