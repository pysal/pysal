
import os
import unittest
import pysal
import numpy as np

class Testlag_spatial(unittest.TestCase):
    def setUp(self):
        self.neighbors={'c': ['b'], 'b': ['c', 'a'], 'a': ['b']}
        self.weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
        self.id_order=['a','b','c']
        self.weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
        self.w=pysal.W(self.neighbors,self.weights,self.id_order)
        self.y = np.array([0,1,2])

    def test_lag_spatial(self):
        yl = pysal.lag_spatial(self.w,self.y)
        np.testing.assert_array_almost_equal(yl,[ 1.,  2.,  1.])
        self.w.id_order = ['b', 'c', 'a']
        y = np.array([1, 2, 0])
        yl = pysal.lag_spatial(self.w, y)
        np.testing.assert_array_almost_equal(yl,[ 2.,  1.,  1.])
        w = pysal.lat2W(3, 3)
        y = np.arange(9)
        yl = pysal.lag_spatial(w, y)
        ylc = np.array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])
        np.testing.assert_array_almost_equal(yl,ylc)
        w.transform = 'r'
        yl = pysal.lag_spatial(w, y)
        ylc = np.array([ 2.        ,  2.        ,  3.        ,  3.33333333,  4.        ,
            4.66666667,  5.        ,  6.        ,  6.        ])
        np.testing.assert_array_almost_equal(yl,ylc)

        
suite = unittest.TestLoader().loadTestsFromTestCase(Testlag_spatial)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
