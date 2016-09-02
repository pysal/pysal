"""
Tests for analysis of spatial autocorrelation within vectors

"""

__author__ = 'Taylor Oshan tayoshan@gmail.com'

import unittest
import numpy as np
np.random.seed(1)
import pysal
from pysal.contrib.spint.vec_SA import VecMoran
from pysal.weights import DistanceBand

class TestVecMoran(unittest.TestCase):
    """Tests VecMoran class"""

    def setUp(self):
        self.vecs = np.array([[1, 55, 60, 100, 500], 
                             [2, 60, 55, 105, 501], 
                             [3, 500, 55, 155, 500], 
                             [4, 505, 60, 160, 500], 
                             [5, 105, 950, 105, 500], 
                             [6, 155, 950, 155, 499]])
        self.origins = self.vecs[:, 1:3]
        self.dests = self.vecs[:, 3:5]

    def test_origin_focused_A(self):
        wo = DistanceBand(self.origins, threshold=9999, alpha=-1.5, binary=False)
        vmo = VecMoran(self.vecs, wo, focus='origin', rand='A')
        self.assertAlmostEquals(vmo.I, 0.645944594367)
        self.assertAlmostEquals(vmo.p_z_sim, 0.099549579548)

    def test_dest_focused_A(self):
        wd = DistanceBand(self.dests, threshold=9999, alpha=-1.5, binary=False)
        vmd = VecMoran(self.vecs, wd, focus='destination', rand='A')
        self.assertAlmostEquals(vmd.I, -0.764603695022)
        self.assertAlmostEquals(vmd.p_z_sim, 0.149472673677)
    
    def test_origin_focused_B(self):
        wo = DistanceBand(self.origins, threshold=9999, alpha=-1.5, binary=False)
        vmo = VecMoran(self.vecs, wo, focus='origin', rand='B')
        self.assertAlmostEquals(vmo.I, 0.645944594367)
        self.assertAlmostEquals(vmo.p_z_sim, 0.071427063787951814)

    def test_dest_focused_B(self):
        wd = DistanceBand(self.dests, threshold=9999, alpha=-1.5, binary=False)
        vmd = VecMoran(self.vecs, wd, focus='destination', rand='B')
        self.assertAlmostEquals(vmd.I, -0.764603695022)
        self.assertAlmostEquals(vmd.p_z_sim, 0.086894261015806051)

if __name__ == '__main__':
	    unittest.main()

