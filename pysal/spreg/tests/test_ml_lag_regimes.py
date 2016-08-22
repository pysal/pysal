import unittest
import scipy
import pysal
import numpy as np
from pysal.spreg.ml_lag_regimes import ML_Lag_Regimes
from pysal.spreg.ml_lag import ML_Lag
from pysal.spreg import utils
from pysal.common import RTOL
from skip import SKIP



@unittest.skipIf(SKIP,
        "Skipping MLLag Tests")
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
        self.regimes = db.by_col("CITCOU")

    def test_model1(self):
        reg = ML_Lag_Regimes(self.y,self.x,self.regimes,w=self.w,name_y=self.y_name,name_x=self.x_names,\
               name_w=self.w_name,name_ds=self.ds_name,name_regimes="CITCOU", regime_lag_sep=False)
        betas = np.array([[-15.00586577],
       [  4.49600801],
       [ -0.03180518],
       [  0.34995882],
       [ -4.54040395],
       [  3.92187578],
       [ -0.17021393],
       [  0.81941371],
       [  0.53850323]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 32.73718478])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 14.26281522])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 211
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 9
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 47.])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([[   1.  ,    4.  ,  148.  ,   11.25,    0.  ,    0.  ,    0.  ,
           0.  ]])
        np.testing.assert_allclose(reg.x[0].toarray(),x,RTOL)
        e = np.array([ 29.45407124])
        np.testing.assert_allclose(reg.e_pred[0],e,RTOL)
        my = 44.307180094786695
        np.testing.assert_allclose(reg.mean_y,my)
        sy = 23.606076835380495
        np.testing.assert_allclose(reg.std_y,sy)
        vm = np.array([ 47.42000914,   2.39526578,   0.00506895,   0.06480022,
        69.67653371,   3.20661492,   0.01156766,   0.04862014,   0.00400775])
        np.testing.assert_allclose(reg.vm.diagonal(),vm,RTOL)
        sig2 = 200.04433357145007
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.6404460298085746
        np.testing.assert_allclose(reg.pr2,pr2)
        std_err = np.array([ 6.88621878,  1.54766462,  0.07119654,  0.25455888,  8.34724707,
        1.79070235,  0.10755305,  0.22049975,  0.0633068 ])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        logll = -864.98505596489736
        np.testing.assert_allclose(reg.logll,logll,RTOL)
        aic = 1747.9701119297947
        np.testing.assert_allclose(reg.aic,aic,RTOL)
        schwarz = 1778.1368351310794
        np.testing.assert_allclose(reg.schwarz,schwarz,RTOL)
        chow_r = np.array([[ 1.00180776,  0.31687348],
       [ 0.05904944,  0.8080047 ],
       [ 1.16987812,  0.27942629],
       [ 1.95931177,  0.16158694]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 21.648337464039283
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)

    def test_model2(self):
        reg = ML_Lag_Regimes(self.y,self.x,self.regimes,w=self.w,name_y=self.y_name,name_x=self.x_names,\
               name_w=self.w_name,name_ds=self.ds_name,name_regimes="CITCOU", regime_lag_sep=True)
        betas = np.array([[-0.71589799],
       [ 4.40910538],
       [-0.08652467],
       [ 0.46266265],
       [ 0.1627765 ],
       [-5.00594358],
       [ 2.91060349],
       [-0.18207394],
       [ 0.71129227],
       [ 0.66753263]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 55.3593679 ,  -7.22927797,  -0.19487326,   0.6030953 ,
        -0.52249569,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([ 34.03630518])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 12.96369482])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([ 32.46466912])
        np.testing.assert_allclose(reg.e_pred[0],e,RTOL)
        chow_r = np.array([[  0.15654726,   0.69235548],
       [  0.43533847,   0.509381  ],
       [  0.60552514,   0.43647766],
       [  0.59214981,   0.441589  ],
       [ 11.69437282,   0.00062689]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 21.978012275873063
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)

if __name__ == '__main__':
    unittest.main()
