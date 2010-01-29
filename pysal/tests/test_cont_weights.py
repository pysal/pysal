import unittest
import pysal
from pysal.weights._contW_rtree import ContiguityWeights_rtree,QUEEN
from pysal.weights._contW_binning import ContiguityWeights_binning

class _TestContiguityWeights(unittest.TestCase):
    def setUp(self):
        """ Setup the rtree and binning contiguity weights"""

        shpObj = pysal.open('../examples/10740.shp','r')
        self.rtreeW = ContiguityWeights_rtree(shpObj,QUEEN)
        shpObj.seek(0)
        self.binningW = ContiguityWeights_binning(shpObj,QUEEN)
        shpObj.close()

    def test_w_type(self):
        self.assert_(isinstance(self.rtreeW,ContiguityWeights_rtree))
        self.assert_(isinstance(self.binningW,ContiguityWeights_binning))
    def test_w_content(self):
        self.assertEqual(self.rtreeW.w, self.binningW.w)

suite = unittest.TestLoader().loadTestsFromTestCase(_TestContiguityWeights)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
