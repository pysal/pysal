import unittest
import pysal
import numpy as np
#from pysal.spreg import probit as PB

from pysal.contrib.handler import Model
from functools import partial

Probit = partial(Model, mtype='Probit')

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
        reg = Probit(self.y, self.X, w=self.w)
        betas = np.array([[ 3.35381078], [-0.1996531 ], [-0.02951371]])
        np.testing.assert_array_almost_equal(reg.betas,betas,6)
        predy = np.array([ 0.00174739])
        np.testing.assert_array_almost_equal(reg.predy[0],predy,6)
        n = 49
        self.assertAlmostEqual(reg.n,n,6)
        k = 3
        self.assertAlmostEqual(reg.k,k,6)
        y = np.array([ 0.])
        np.testing.assert_array_almost_equal(reg.y[0],y,6)
        x = np.array([  1.      ,  19.531   ,  80.467003])
        np.testing.assert_array_almost_equal(reg.x[0],x,6)
        vm = np.array([[  8.52813879e-01,  -4.36272459e-02,  -8.05171472e-03], [ -4.36272459e-02,   4.11381444e-03,  -1.92834842e-04], [ -8.05171472e-03,  -1.92834842e-04,   3.09660240e-04]])
        np.testing.assert_array_almost_equal(reg.vm,vm,6)
        xmean = np.array([[  1.        ], [ 14.37493876], [ 38.43622447 ]])
        np.testing.assert_array_almost_equal(reg.xmean,xmean,6)        
        predpc = 85.714285714285708
        self.assertAlmostEqual(reg.predpc,predpc,5)
        logl = -20.06009093055782
        self.assertAlmostEqual(reg.logl,logl,5)
        scale = 0.23309310130643665
        self.assertAlmostEqual(reg.scale,scale,5)
        slopes = np.array([[-0.04653776], [-0.00687944]])
        np.testing.assert_array_almost_equal(reg.slopes,slopes,6)
        slopes_vm = np.array([[  1.77101993e-04,  -1.65021168e-05], [ -1.65021168e-05,   1.60575016e-05]])
        np.testing.assert_array_almost_equal(reg.slopes_vm,slopes_vm,6)
        LR = 25.317683245671716
        self.assertAlmostEqual(reg.LR[0],LR,5)
        Pinkse_error = 2.9632385352516728
        self.assertAlmostEqual(reg.Pinkse_error[0],Pinkse_error,5)
        KP_error = 1.6509224700582124
        self.assertAlmostEqual(reg.KP_error[0],KP_error,5)
        PS_error = 2.3732463777623511
        self.assertAlmostEqual(reg.PS_error[0],PS_error,5)
        
if __name__ == '__main__':
    unittest.main()
