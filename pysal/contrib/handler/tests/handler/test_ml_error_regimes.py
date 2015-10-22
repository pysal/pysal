import unittest
import scipy
import pysal
import numpy as np
#from pysal.spreg.ml_error_regimes import ML_Error_Regimes
#from pysal.spreg.ml_error import ML_Error
from pysal.spreg import utils

from functools import partial
from pysal.contrib.handler import Model

ML_Error_Regimes = partial(Model, mtype='ML_Error_Regimes')
ML_Error = partial(Model, mtype='ML_Error')

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
        self.regimes = db.by_col("CITCOU")
        #Artficial:
        n = 256
        self.n2 = n/2
        self.x_a1 = np.random.uniform(-10,10,(n,1))
        self.x_a2 = np.random.uniform(1,5,(n,1))
        self.q_a = self.x_a2 + np.random.normal(0,1,(n,1))
        self.x_a = np.hstack((self.x_a1,self.x_a2))
        self.y_a = np.dot(np.hstack((np.ones((n,1)),self.x_a)),np.array([[1],[0.5],[2]])) + np.random.normal(0,1,(n,1))
        latt = int(np.sqrt(n))
        self.w_a = pysal.lat2W(latt,latt)
        self.w_a.transform='r'
        self.regi_a = [0]*(n/2) + [1]*(n/2)
        self.w_a1 = pysal.lat2W(latt/2,latt)
        self.w_a1.transform='r'

    def test_model1(self):
        reg = ML_Error_Regimes(self.y,self.x,self.regimes,w=self.w,name_y=self.y_name,name_x=self.x_names,\
               name_w=self.w_name,name_ds=self.ds_name,name_regimes="CITCOU", regime_err_sep=False)
        betas = np.array([[ -2.39491278],
       [  4.873757  ],
       [ -0.02911854],
       [  0.33275008],
       [ 31.79618475],
       [  2.98102401],
       [ -0.23710892],
       [  0.80581127],
       [  0.61770744]])
        np.allclose(reg.betas,betas,4)
        u = np.array([ 30.46599009])
        np.allclose(reg.u[0],u,4)
        predy = np.array([ 16.53400991])
        np.allclose(reg.predy[0],predy,4)
        n = 211
        np.allclose(reg.n,n,4)
        k = 8
        np.allclose(reg.k,k,4)
        y = np.array([ 47.])
        np.allclose(reg.y[0],y,4)
        x = np.array([   1.  ,    4.  ,  148.  ,   11.25,    0.  ,    0.  ,    0.  ,    0.  ])
        np.allclose(reg.x[0],x,4)
        e = np.array([ 34.69181334])
        np.allclose(reg.e_filtered[0],e,4)
        my = 44.307180094786695
        np.allclose(reg.mean_y,my)
        sy = 23.606076835380495
        np.allclose(reg.std_y,sy)
        vm = np.array([ 58.50551173,   2.42952002,   0.00721525,   0.06391736,
        80.59249161,   3.1610047 ,   0.0119782 ,   0.0499432 ,   0.00502785])
        np.allclose(reg.vm.diagonal(),vm,4)
        sig2 = np.array([[ 209.60639741]])
        np.allclose(reg.sig2,sig2,4)
        pr2 = 0.43600837301477025
        np.allclose(reg.pr2,pr2)
        std_err = np.array([ 7.64888957,  1.55869177,  0.08494262,  0.25281882,  8.9773321 ,
        1.77792146,  0.10944497,  0.22347975,  0.07090735])
        np.allclose(reg.std_err,std_err,4)
        logll = -870.3331059537576
        np.allclose(reg.logll,logll,4)
        aic = 1756.6662119075154
        np.allclose(reg.aic,aic,4)
        schwarz = 1783.481076975324
        np.allclose(reg.schwarz,schwarz,4)
        chow_r = np.array([[ 8.40437046,  0.0037432 ],
       [ 0.64080535,  0.42341932],
       [ 2.25389396,  0.13327865],
       [ 1.96544702,  0.16093197]])
        np.allclose(reg.chow.regi,chow_r,4)
        chow_j = 25.367913028011799
        np.allclose(reg.chow.joint[0],chow_j,4)

    def test_model2(self):
        reg = ML_Error_Regimes(self.y,self.x,self.regimes,w=self.w,name_y=self.y_name,name_x=self.x_names,\
               name_w=self.w_name,name_ds=self.ds_name,name_regimes="CITCOU", regime_err_sep=True)
        betas = np.array([[  3.66158216],
       [  4.55700255],
       [ -0.08045502],
       [  0.44800318],
       [  0.17774677],
       [ 33.3086368 ],
       [  2.44709405],
       [ -0.18803509],
       [  0.68956598],
       [  0.75599089]])
        np.allclose(reg.betas,betas,4)
        vm = np.array([ 40.60994599,  -7.25413138,  -0.16605501,   0.48961884,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ])
        np.allclose(reg.vm[0],vm,4)
        u = np.array([ 31.97771505])
        np.allclose(reg.u[0],u,4)
        predy = np.array([ 15.02228495])
        np.allclose(reg.predy[0],predy,4)
        e = np.array([ 33.83065421])
        np.allclose(reg.e_filtered[0],e,4)
        chow_r = np.array([[  6.88023639,   0.0087154 ],
       [  0.90512612,   0.34141092],
       [  0.75996258,   0.38334023],
       [  0.56882946,   0.45072443],
       [ 12.18358581,   0.00048212]])
        np.allclose(reg.chow.regi,chow_r,4)
        chow_j = 26.673798071789673
        np.allclose(reg.chow.joint[0],chow_j,4)
        #Artficial:
        model = ML_Error_Regimes(self.y_a, self.x_a, self.regi_a, w=self.w_a, regime_err_sep=True)
        model1 = ML_Error(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a[0:(self.n2)], w=self.w_a1)
        model2 = ML_Error(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a[(self.n2):], w=self.w_a1)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.allclose(model.betas,tbetas)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.allclose(model.vm.diagonal(), vm, 4)

if __name__ == '__main__':
    unittest.main()
