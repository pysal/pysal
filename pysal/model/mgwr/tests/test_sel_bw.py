"""
GWR is tested against results from GWR4
"""

import os
import numpy as np
from pysal.lib import io
import pysal.lib as ps
import unittest
from pysal.model.spglm.family import Gaussian, Poisson, Binomial
from ..sel_bw import Sel_BW
from numpy.testing import assert_allclose

class TestSelBW(unittest.TestCase):
    def setUp(self):
        data_path = ps.examples.get_path("GData_utm.csv")
        data = io.open(data_path)
        self.coords = list(zip(data.by_col('X'), data.by_col('Y')))
        self.y = np.array(data.by_col('PctBach')).reshape((-1,1))
        rural  = np.array(data.by_col('PctRural')).reshape((-1,1))
        pov = np.array(data.by_col('PctPov')).reshape((-1,1)) 
        black = np.array(data.by_col('PctBlack')).reshape((-1,1))
        fb = np.array(data.by_col('PctFB')).reshape((-1,1))
        self.X = np.hstack([rural, pov, black])
        self.mgwr_X = np.hstack([fb, black, rural])
  
    def test_golden_fixed_AICc(self):
        bw1 = 211020.83
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=True).search(criterion='AICc')
        assert_allclose(bw1, bw2)
        scipy_known = 211025.26298
        scipy = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=True).search(criterion='AICc', search_method='scipy')
        assert_allclose(scipy_known, scipy, atol=1)

    def test_golden_adapt_AICc(self):
        bw1 = 93.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=False).search(criterion='AICc')
        assert_allclose(bw1, bw2)

    def test_golden_fixed_AIC(self):
        bw1 = 76201.66
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='AIC')
        assert_allclose(bw1, bw2)
        scipy_known = 76199.81
        scipy = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='AIC', search_method='scipy')
        assert_allclose(scipy_known, scipy, atol=1)
    
    def test_golden_adapt_AIC(self):
        bw1 = 50.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='AIC')
        assert_allclose(bw1, bw2)

    
    def test_golden_fixed_BIC(self):
        bw1 = 1117795.47
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='BIC')
        assert_allclose(bw1, bw2)
        scipy_known = 1117806.16
        scipy = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='BIC', search_method='scipy')
        assert_allclose(scipy_known, scipy, atol=1)
    
    def test_golden_adapt_BIC(self):
        bw1 = 62.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='BIC')
        assert_allclose(bw1, bw2)
    
    def test_golden_fixed_CV(self):
        bw1 = 130289.26
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='CV')
        assert_allclose(bw1, bw2)
        scipy_known = 130363.55
        scipy = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='CV', search_method='scipy')
        assert_allclose(scipy_known, scipy, atol=1)
    
    def test_golden_adapt_CV(self):
        bw1 = 68.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='CV')
        assert_allclose(bw1, bw2)
   
    def test_interval_fixed_AICc(self):
        bw1 = 211025.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=True).search(criterion='AICc', search_method='interval',
                        bw_min=211001.0, bw_max=211035.0, interval=2)
        assert_allclose(bw1, bw2)

    def test_interval_adapt_AICc(self):
        bw1 = 93.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=False).search(criterion='AICc', search_method='interval',
                        bw_min=90.0, bw_max=95.0, interval=1)
        assert_allclose(bw1, bw2)

    def test_interval_fixed_AIC(self):
        bw1 = 76175.0#76169.00
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='AIC', search_method='interval',
                        bw_min=76161.0, bw_max=76175.0, interval=1)
        assert_allclose(bw1, bw2)
    
    def test_interval_adapt_AIC(self):
        bw1 = 40.0#50.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='AIC', search_method='interval', bw_min=40.0,
                        bw_max=60.0, interval=2)
        assert_allclose(bw1, bw2)
    
    def test_interval_fixed_BIC(self):
        bw1 = 279461.0#279451.00
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='BIC', search_method='interval', bw_min=279441.0,
                        bw_max=279461.0, interval=2)
        assert_allclose(bw1, bw2)
    
    def test_interval_adapt_BIC(self):
        bw1 = 62.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='BIC', search_method='interval',
                        bw_min=52.0, bw_max=72.0, interval=2)
        assert_allclose(bw1, bw2)
    
    def test_interval_fixed_CV(self):
        bw1 = 130400.0#130406.00
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='CV', search_method='interval', bw_min=130400.0,
                        bw_max=130410.0, interval=1)
        assert_allclose(bw1, bw2)
    
    def test_interval_adapt_CV(self):
        bw1 = 62.0#68.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='CV', search_method='interval', bw_min=60.0,
                        bw_max=76.0 , interval=2)
        assert_allclose(bw1, bw2)

    def test_MGWR_AICc(self):
        bw1 = [101.0, 101.0, 117.0, 157.0]
        std_y = (self.y - self.y.mean()) / self.y.std()
        std_X = (self.mgwr_X - self.mgwr_X.mean(axis=0)) / self.mgwr_X.std(axis=0)
        selector = Sel_BW(self.coords, std_y, std_X, multi=True,
                constant=True)
        bw2 = selector.search()
        np.testing.assert_allclose(bw1, bw2)

if __name__ == '__main__':
	unittest.main()
