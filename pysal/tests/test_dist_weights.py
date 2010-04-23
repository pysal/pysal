import os
import unittest
import pysal

class _TestDistanceWeights(unittest.TestCase):
    def setUp(self):
        """ Setup the rtree and binning contiguity weights"""
        self.polyShp = '../examples/virginia.shp'
        self.pointShp = '../examples/juvenile.shp'
    def test_A(self):
        self.assertEqual(1,1)

suite = unittest.TestLoader().loadTestsFromTestCase(_TestDistanceWeights)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
