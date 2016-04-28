import unittest
import pysal
import scipy
import numpy as np
#from pysal.spreg.ml_lag import ML_Lag
from pysal.spreg import utils

from functools import partial
from pysal.contrib.handler import Model

ML_Lag = partial(Model, mtype='ML_Lag')

@unittest.skipIf(int(scipy.__version__.split(".")[1]) < 11,
        "Max Likelihood requires SciPy version 11 or newer.")
class TestMLError(unittest.TestCase):
    def setUp(self):
        db =  pysal.open(pysal.examples.get_path("baltim.dbf"),'r')
        self.ds_name = "baltim.dbf"
        self.y_name = "PRICE"
        self.y = np.array(db.by_col(self.y_name)).T
        self.y.shape = (len(self.y),1)
        self.x_names = ["NROOM","AGE","SQFT"]
        self.x = np.array([db.by_col(var) for var in self.x_names]).T
        ww = pysal.open(pysal.examples.get_path("baltim_q.gal"))
        self.w = ww.read()
        ww.close()
        self.w_name = "baltim_q.gal"
        self.w.transform = 'r'

    def test_model1(self):
        reg = ML_Lag(self.y,self.x,w=self.w,name_y=self.y_name,name_x=self.x_names,\
               name_w=self.w_name,name_ds=self.ds_name)
        betas = np.array([[-6.04040164],
       [ 3.48995114],
       [-0.20103955],
       [ 0.65462382],
       [ 0.62351143]])
        np.testing.assert_array_almost_equal(reg.betas,betas,4)
        u = np.array([ 47.51218398])
        np.testing.assert_array_almost_equal(reg.u[0],u,4)
        predy = np.array([-0.51218398])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,4)
        n = 211
        self.assertAlmostEqual(reg.n,n,4)
        k = 5
        self.assertAlmostEqual(reg.k,k,4)
        y = np.array([ 47.])
        np.testing.assert_array_almost_equal(reg.y[0],y,4)
        x = np.array([   1.  ,    4.  ,  148.  ,   11.25])
        np.testing.assert_array_almost_equal(reg.x[0],x,4)
        e = np.array([ 41.99251608])
        np.testing.assert_array_almost_equal(reg.e_pred[0],e,4)
        my = 44.307180094786695
        self.assertAlmostEqual(reg.mean_y,my)
        sy = 23.606076835380495
        self.assertAlmostEqual(reg.std_y,sy)
        vm = np.array([ 28.57288755,   1.42341656,   0.00288068,   0.02956392,   0.00332139])
        np.testing.assert_array_almost_equal(reg.vm.diagonal(),vm,4)
        sig2 = 216.27525647243797
        self.assertAlmostEqual(reg.sig2,sig2,4)
        pr2 = 0.6133020721559487
        self.assertAlmostEqual(reg.pr2,pr2)
        std_err = np.array([ 5.34536131,  1.19307022,  0.05367198,  0.17194162,  0.05763147])
        np.testing.assert_array_almost_equal(reg.std_err,std_err,4)
        logll = -875.92771143484833
        self.assertAlmostEqual(reg.logll,logll,4)
        aic = 1761.8554228696967
        self.assertAlmostEqual(reg.aic,aic,4)
        schwarz = 1778.614713537077
        self.assertAlmostEqual(reg.schwarz,schwarz,4)

if __name__ == '__main__':
    unittest.main()
