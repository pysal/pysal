
"""
GWR is tested against results from GWR4
"""

import unittest
import pickle as pk
from pysal.contrib.glm.family import Gaussian, Poisson, Binomial
from pysal.contrib.gwr.sel_bw import Sel_BW
import numpy as np
import pysal

class TestSelBW(unittest.TestCase):
    def setUp(self):
        data = pysal.open(pysal.examples.get_path('GData_utm.csv'))
        self.coords = zip(data.by_col('X'), data.by_col('Y'))
        self.y = np.array(data.by_col('PctBach')).reshape((-1,1))
        rural  = np.array(data.by_col('PctRural')).reshape((-1,1))
        pov = np.array(data.by_col('PctPov')).reshape((-1,1)) 
        black = np.array(data.by_col('PctBlack')).reshape((-1,1))
        self.X = np.hstack([rural, pov, black])
        self.XB = pk.load(open(pysal.examples.get_path('XB.p'), 'r'))
        self.err = pk.load(open(pysal.examples.get_path('err.p'), 'r'))
  
    def test_golden_fixed_AICc(self):
        bw1 = 211027.34
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=True).search(criterion='AICc')
        self.assertAlmostEqual(bw1, bw2)

    def test_golden_adapt_AICc(self):
        bw1 = 93.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=False).search(criterion='AICc')
        self.assertAlmostEqual(bw1, bw2)

    def test_golden_fixed_AIC(self):
        bw1 = 76169.15
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='AIC')
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_AIC(self):
        bw1 = 50.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='AIC')
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_fixed_BIC(self):
        bw1 = 279451.43
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='BIC')
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_BIC(self):
        bw1 = 62.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='BIC')
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_fixed_CV(self):
        bw1 = 130406.67
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='CV')
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_CV(self):
        bw1 = 68.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='CV')
        self.assertAlmostEqual(bw1, bw2)
   
    def test_interval_fixed_AICc(self):
        bw1 = 211025.0#211027.00
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=True).search(criterion='AICc', search='interval',
                        bw_min=211001.0, bw_max=211035.0, interval=2)
        self.assertAlmostEqual(bw1, bw2)

    def test_interval_adapt_AICc(self):
        bw1 = 93.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=False).search(criterion='AICc', search='interval',
                        bw_min=90.0, bw_max=95.0, interval=1)
        self.assertAlmostEqual(bw1, bw2)

    def test_interval_fixed_AIC(self):
        bw1 = 76175.0#76169.00
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='AIC', search='interval',
                        bw_min=76161.0, bw_max=76175.0, interval=1)
        self.assertAlmostEqual(bw1, bw2)
    
    def test_interval_adapt_AIC(self):
        bw1 = 40.0#50.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='AIC', search='interval', bw_min=40.0,
                        bw_max=60.0, interval=2)
        self.assertAlmostEqual(bw1, bw2)
    
    def test_interval_fixed_BIC(self):
        bw1 = 279461.0#279451.00
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='BIC', search='interval', bw_min=279441.0,
                        bw_max=279461.0, interval=2)
        self.assertAlmostEqual(bw1, bw2)
    
    def test_interval_adapt_BIC(self):
        bw1 = 62.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='BIC', search='interval',
                        bw_min=52.0, bw_max=72.0, interval=2)
        self.assertAlmostEqual(bw1, bw2)
    
    def test_interval_fixed_CV(self):
        bw1 = 130400.0#130406.00
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='CV', search='interval', bw_min=130400.0,
                        bw_max=130410.0, interval=1)
        self.assertAlmostEqual(bw1, bw2)
    
    def test_interval_adapt_CV(self):
        bw1 = 62.0#68.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='CV', search='interval', bw_min=60.0,
                        bw_max=76.0 , interval=2)
        self.assertAlmostEqual(bw1, bw2)

    def test_MGWR_AIC(self):
        bw1 = [157.0, 65.0, 52.0]
        sel = Sel_BW(self.coords, self.y, self.X, multi=True, kernel='bisquare',
                constant=False)
        bw2 = sel.search(tol_multi=1e-03)
        np.testing.assert_allclose(bw1, bw2)
        np.testing.assert_allclose(sel.XB, self.XB, atol=1e-05)
        np.testing.assert_allclose(sel.err, self.err, atol=1e-05)

if __name__ == '__main__':
	unittest.main()
