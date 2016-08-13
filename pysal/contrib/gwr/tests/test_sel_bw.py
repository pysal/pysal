
"""
GWR is tested against results from GWR4
"""

import unittest
import sys
sys.path.append('/Users/toshan/dev/pysal/pysal/contrib/gwr')
sys.path.append('/Users/toshan/dev/pysal/pysal/contrib/glm')
from family import Gaussian, Poisson, Binomial
from sel_bw import Sel_BW
import numpy as np
import pysal
import os

class TestBWGaussian(unittest.TestCase):
    def setUp(self):
        os.chdir('/Users/toshan/dev/pysal/pysal/contrib/gwr/examples/georgia')
        data = pysal.open('georgia/GData_utm.csv')
        self.coords = zip(data.by_col('X'), data.by_col('Y'))
        self.y = np.array(data.by_col('PctBach')).reshape((-1,1))
        rural  = np.array(data.by_col('PctRural')).reshape((-1,1))
        pov = np.array(data.by_col('PctPov')).reshape((-1,1)) 
        black = np.array(data.by_col('PctBlack')).reshape((-1,1))
        self.X = np.hstack([rural, pov, black])

    def test_golden_fixed_AICc(self):
        bw1 = 211027.34
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=True).search(criterion='AICc')
        print bw1, bw2, 'a'
        self.assertAlmostEqual(bw1, bw2)

    def test_golden_adapt_AICc(self):
        bw1 = 93.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=False).search(criterion='AICc')
        print bw1, bw2, 'b'
        self.assertAlmostEqual(bw1, bw2)

    def test_golden_fixed_AIC(self):
        bw1 = 76169.15
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='AIC')
        print bw1, bw2, 'c'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_AIC(self):
        bw1 = 50.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='AIC')
        print bw1, bw2, 'd'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_fixed_BIC(self):
        bw1 = 279451.43
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='BIC')
        print bw1, bw2, 'e'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_BIC(self):
        bw1 = 62.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='BIC')
        print bw1, bw2, 'f'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_fixed_CV(self):
        bw1 = 130406.67
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='CV')
        print bw1, bw2, 'g'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_CV(self):
        bw1 = 68.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='CV')
        print bw1, bw2, 'h'
        self.assertAlmostEqual(bw1, bw2)
       
class TestGWRPoisson(unittest.TestCase):
    def setUp(self):
        os.chdir('/Users/toshan/dev/pysal/pysal/contrib/gwr/examples/tokyo')
        data = pysal.open('tokyo/tokyomortality.csv', mode='rU')
        self.coords = zip(data.by_col('X_CENTROID'), data.by_col('Y_CENTROID'))
        self.y = np.array(data.by_col('db2564')).reshape((-1,1))
        self.off = np.array(data.by_col('eb2564')).reshape((-1,1))
        OCC  = np.array(data.by_col('OCC_TEC')).reshape((-1,1))
        OWN = np.array(data.by_col('OWNH')).reshape((-1,1)) 
        POP = np.array(data.by_col('POP65')).reshape((-1,1))
        UNEMP = np.array(data.by_col('UNEMP')).reshape((-1,1))
        self.X = np.hstack([OCC,OWN,POP,UNEMP])

    def test_golden_fixed_AICc(self):
        bw1 = 67330.37
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=True).search(criterion='AICc')
        print bw1, bw2, 'i'
        self.assertAlmostEqual(bw1, bw2)

    def test_golden_adapt_AICc(self):
        bw1 = 143.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=False).search(criterion='AICc')
        print bw1, bw2, 'j'
        self.assertAlmostEqual(bw1, bw2)

    def test_golden_fixed_AIC(self):
        bw1 = 18149.19
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='AIC')
        print bw1, bw2, 'k'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_AIC(self):
        bw1 = 51.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='AIC')
        print bw1, bw2, 'l'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_fixed_BIC(self):
        bw1 = 67330.41
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='BIC')
        print bw1, bw2, 'm'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_BIC(self):
        bw1 = 261.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='BIC')
        print bw1, bw2, 'n'
        self.assertAlmostEqual(bw1, bw2)
    

class TestGWRBinomial(unittest.TestCase):
    def setUp(self):
        os.chdir('/Users/toshan/dev/pysal/pysal/contrib/gwr/examples/clearwater')
        data = pysal.open('clearwater/landslides.csv')
        self.coords = zip(data.by_col('X'), data.by_col('Y'))
        self.y = np.array(data.by_col('Landslid')).reshape((-1,1))
        ELEV  = np.array(data.by_col('Elev')).reshape((-1,1))
        SLOPE = np.array(data.by_col('Slope')).reshape((-1,1)) 
        SIN = np.array(data.by_col('SinAspct')).reshape((-1,1))
        COS = np.array(data.by_col('CosAspct')).reshape((-1,1))
        SOUTH = np.array(data.by_col('AbsSouth')).reshape((-1,1))
        DIST = np.array(data.by_col('DistStrm')).reshape((-1,1))
        self.X = np.hstack([ELEV, SLOPE, SIN, COS, SOUTH, DIST])

    def test_golden_fixed_AICc(self):
        bw1 = 13466.82
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=True).search(criterion='AICc')
        print bw1, bw2, 'o'
        self.assertAlmostEqual(bw1, bw2)

    def test_golden_adapt_AICc(self):
        bw1 = 168.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='bisquare',
                fixed=False).search(criterion='AICc')
        print bw1, bw2, 'p'
        self.assertAlmostEqual(bw1, bw2)

    def test_golden_fixed_AIC(self):
        bw1 = 13466.82
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='AIC')
        print bw1, bw2, 'q'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_AIC(self):
        bw1 = 168.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='AIC')
        print bw1, bw2, 'r'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_fixed_BIC(self):
        bw1 = 13466.82
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=True).search(criterion='BIC')
        print bw1, bw2, 's'
        self.assertAlmostEqual(bw1, bw2)
    
    def test_golden_adapt_BIC(self):
        bw1 = 168.0
        bw2 = Sel_BW(self.coords, self.y, self.X, kernel='gaussian',
                fixed=False).search(criterion='BIC')
        print bw1, bw2, 't'
        self.assertAlmostEqual(bw1, bw2)

if __name__ == '__main__':
	unittest.main()
