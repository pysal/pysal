"""Geary Unittest."""
import unittest
import pysal
from pysal.esda import geary 
import numpy as np

class Geary_Tester(unittest.TestCase):
    """Geary class for unit tests."""
    def setUp(self):
        self.w = pysal.open("../../examples/book.gal").read()
        f = pysal.open("../../examples/book.txt")
        self.y = np.array(f.by_col['y'])
    def test_Geary(self):
        c = Geary(self.y, self.w, permutation=0)
        self.assertAlmostEquals(c.C, 0.33281733746130032 )

