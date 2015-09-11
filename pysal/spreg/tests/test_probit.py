import unittest
import pysal
import numpy as np
from pysal.spreg import probit as PB
from pysal.common import RTOL

class TestBaseProbit(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("CRIME"))
        y = np.reshape(y, (49,1))
        self.y = (y>40).astype(float)
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        self.X = np.array(X).T
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = PB.BaseProbit(self.y, self.X, w=self.w)
        betas = np.array([[ 3.35381078], [-0.1996531 ], [-0.02951371]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([ 0.00174739])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 0.])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.      ,  19.531   ,  80.467003])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        vm = np.array([[  8.52813879e-01,  -4.36272459e-02,  -8.05171472e-03], [ -4.36272459e-02,   4.11381444e-03,  -1.92834842e-04], [ -8.05171472e-03,  -1.92834842e-04,   3.09660240e-04]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        xmean = np.array([[  1.        ], [ 14.37493876], [ 38.43622447 ]])
        np.testing.assert_allclose(reg.xmean,xmean,RTOL)        
        predpc = 85.714285714285708
        np.testing.assert_allclose(reg.predpc,predpc,RTOL)
        logl = -20.06009093055782
        np.testing.assert_allclose(reg.logl,logl,RTOL)
        scale = 0.23309310130643665
        np.testing.assert_allclose(reg.scale,scale,RTOL)
        slopes = np.array([[-0.04653776], [-0.00687944]])
        np.testing.assert_allclose(reg.slopes,slopes,RTOL)
        slopes_vm = np.array([[  1.77101993e-04,  -1.65021168e-05], [ -1.65021168e-05,   1.60575016e-05]])
        np.testing.assert_allclose(reg.slopes_vm,slopes_vm,RTOL)
        LR = 25.317683245671716
        np.testing.assert_allclose(reg.LR[0],LR,RTOL)
        Pinkse_error = 2.9632385352516728
        np.testing.assert_allclose(reg.Pinkse_error[0],Pinkse_error,RTOL)
        KP_error = 1.6509224700582124
        np.testing.assert_allclose(reg.KP_error[0],KP_error,RTOL)
        PS_error = 2.3732463777623511
        np.testing.assert_allclose(reg.PS_error[0],PS_error,RTOL)

class TestProbit(unittest.TestCase):
    def setUp(self):
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("CRIME"))
        y = np.reshape(y, (49,1))
        self.y = (y>40).astype(float)
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        self.X = np.array(X).T
        self.w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = PB.Probit(self.y, self.X, w=self.w)
        betas = np.array([[ 3.35381078], [-0.1996531 ], [-0.02951371]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([ 0.00174739])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 0.])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.      ,  19.531   ,  80.467003])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        vm = np.array([[  8.52813879e-01,  -4.36272459e-02,  -8.05171472e-03], [ -4.36272459e-02,   4.11381444e-03,  -1.92834842e-04], [ -8.05171472e-03,  -1.92834842e-04,   3.09660240e-04]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        xmean = np.array([[  1.        ], [ 14.37493876], [ 38.43622447 ]])
        np.testing.assert_allclose(reg.xmean,xmean,RTOL)        
        predpc = 85.714285714285708
        np.testing.assert_allclose(reg.predpc,predpc,RTOL)
        logl = -20.06009093055782
        np.testing.assert_allclose(reg.logl,logl,RTOL)
        scale = 0.23309310130643665
        np.testing.assert_allclose(reg.scale,scale,RTOL)
        slopes = np.array([[-0.04653776], [-0.00687944]])
        np.testing.assert_allclose(reg.slopes,slopes,RTOL)
        slopes_vm = np.array([[  1.77101993e-04,  -1.65021168e-05], [ -1.65021168e-05,   1.60575016e-05]])
        np.testing.assert_allclose(reg.slopes_vm,slopes_vm,RTOL)
        LR = 25.317683245671716
        np.testing.assert_allclose(reg.LR[0],LR,RTOL)
        Pinkse_error = 2.9632385352516728
        np.testing.assert_allclose(reg.Pinkse_error[0],Pinkse_error,RTOL)
        KP_error = 1.6509224700582124
        np.testing.assert_allclose(reg.KP_error[0],KP_error,RTOL)
        PS_error = 2.3732463777623511
        np.testing.assert_allclose(reg.PS_error[0],PS_error,RTOL)
        
if __name__ == '__main__':
    unittest.main()
