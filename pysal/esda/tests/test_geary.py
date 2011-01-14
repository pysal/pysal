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
        c = geary.Geary(self.y, self.w, permutations=0)
        self.assertAlmostEquals(c.C, 0.33281733746130032 )

suite = unittest.TestSuite()
test_classes = [ Geary_Tester ]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
