import unittest
import pysal
from pysal.weights._contW_rtree import ContiguityWeights_rtree,QUEEN
from pysal.weights._contW_binning import ContiguityWeights_binning

class _TestContiguityWeights(unittest.TestCase):
    def setUp(self):
        """ Setup the rtree and binning contiguity weights"""

        shpObj = pysal.open('../examples/virginia.shp','r')
        self.rtreeW = ContiguityWeights_rtree(shpObj,QUEEN)
        shpObj.seek(0)
        self.binningW = ContiguityWeights_binning(shpObj,QUEEN)
        shpObj.close()

    def test_w_type(self):
        self.assert_(isinstance(self.rtreeW,ContiguityWeights_rtree))
        self.assert_(isinstance(self.binningW,ContiguityWeights_binning))
    def test_w_content(self):
        self.assertEqual(self.rtreeW.w, self.binningW.w)
    def test_nested_polygons(self):
        # load queen gal file created using Open Geoda.
        geodaW = pysal.open('../examples/virginia.gal','r').read()
        # build matching W with pysal
        pysalW = pysal.queen_from_shapefile('../examples/virginia.shp','POLY_ID')
        # compare output.
        for key in geodaW.neighbors:
            geoda_neighbors = map(int,geodaW.neighbors[key])
            pysal_neighbors = pysalW.neighbors[int(key)]
            geoda_neighbors.sort()
            pysal_neighbors.sort()
            self.assertEqual(geoda_neighbors,pysal_neighbors)

suite = unittest.TestLoader().loadTestsFromTestCase(_TestContiguityWeights)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
