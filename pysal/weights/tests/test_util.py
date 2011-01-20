"""Unit test for util.py"""
import pysal
from pysal.common import *
import pysal.weights
import numpy as np
from scipy import sparse,float32
from scipy.spatial import KDTree
import os, gc

class _Testutil(unittest.TestCase):
    def setUp(self):
        self.neighbors={'c': ['b'], 'b': ['c', 'a'], 'a': ['b']}
        self.weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
        self.id_order=['a','b','c']
        self.weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
        self.w=pysal.W(self.neighbors,self.weights,self.id_order)
        self.y = np.array([0,1,2])

    def test_lat2W(self):
        w9=pysal.lat2W(3,3)
        self.assertEquals(w9.pct_nonzero, 0.29629629629629628)
        self.assertEquals(w9[0], {1: 1.0, 3: 1.0})
        self.assertEquals(w9[3], {0: 1.0, 4: 1.0, 6: 1.0})

    def test_regime_weights(self):
        regimes=np.ones(25)
        regimes[range(10,20)]=2
        regimes[range(21,25)]=3
        regimes = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            2.,  2.,  2., 2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  3.,  3.,
            3.,  3.])
        w = pysal.regime_weights(regimes)
        ww0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertEquals(w.weights[0], ww0)
        wn0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
        self.assertEquals(w.neighbors[0], wn0)
        regimes=['n','n','s','s','e','e','w','w','e']
        n = len(regimes)
        w = pysal.regime_weights(regimes)
        wn = {0: [1], 1: [0], 2: [3], 3: [2], 4: [5, 8], 5: [4, 8], 6: [7], 7: [6], 8: [4, 5]}
        self.assertEquals(w.neighbors, wn)
       
    def test_comb(self):
        x = range(4)
        c = comb(x, 2)

        for c in comb(x,2):
            print c
        ...     
        [0, 1]
        [0, 2]
        [0, 3]
        [1, 2]
        [1, 3]
        [2, 3]
        
suite = unittest.TestLoader().loadTestsFromTestCase(_Testutil)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
