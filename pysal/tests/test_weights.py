import os
import unittest
import pysal

class _TestWeights(unittest.TestCase):
    def setUp(self):
        """ Setup the rtree and binning contiguity weights"""
        self.polyShp = '../examples/virginia.shp'
        self.pointShp = '../examples/juvenile.shp'
    def test_iter(self):
        """ All methods names that begin with 'test' will be executed as a test case """
        self.assert_(os.path.exists(self.pointShp))
        w = pysal.rook_from_shapefile(self.polyShp)
        for i,j in zip(w,w):
            self.assertEquals(i,j)
        
    def test_B(self):
        """ All methods names that begin with 'test' will be executed as a test case """
        self.assert_(os.path.exists(self.polyShp))

suite = unittest.TestLoader().loadTestsFromTestCase(_TestWeights)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
