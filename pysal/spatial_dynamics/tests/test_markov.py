import unittest
import pysal
from pysal.spatial_dynamics import markov
import numpy as np


class Markov_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_Markov(self):
        pass


class LisaMarkov_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_LISA_Markov(self):
        pass


class SpatialMarkov_Tester(unittest.TestCase):
    def setUp(self):
        pass
    def test_Spatial_Markov(self):
        pass


suite = unittest.TestSuite()
test_classes = [Markov_Tester, LisaMarkov_Tester, SpatialMarkov_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)


