import unittest
import numpy as np
import pysal
from pysal.esda import mixture_smoothing as m_s

class MS_Tester(unittest.TestCase):
    """Mixture_Smoothing Unit Tests"""
    def setUp(self):
        self.e = np.array([10, 5, 12, 20])
        self.b = np.array([100, 150, 80, 200])

    def test_NP_Mixture_Smoother(self):
        """Test the main class"""
        mix = m_s.NP_Mixture_Smoother(self.e, self.b)
        self.failUnless(mix.r, array([ 0.10982267,  0.03445525,  0.11018393, 0.11018593]))
        self.failUnless(mix.category, array([1, 0, 1, 1]))
        self.failUnless(mix.getSeed(), (array([ 0.5,  0.5]), array([ 0.03333333,
            0.15      ])))
        self.failUnless(mix.mixalg(), {'mix_den': array([ 0.,  0.,  0.,  0.]),
            'gradient': array([ 0.]), 'k': 1, 'p': array([ 1.]), 'grid': array([
                11.27659574]), 'accuracy': 1.0})
        self.failUnless(mix.getRateEstimates(), (array([ 0.0911574,  0.0911574,
            0.0911574,  0.0911574]), array([1, 1, 1, 1])))

        
