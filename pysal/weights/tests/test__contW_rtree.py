import os
import unittest
import pysal
OK_TO_RUN = True
try:
    #import rtree
    from pysal.weights._contW_rtree import ContiguityWeights_rtree, QUEEN, ROOK
except ImportError:
    OK_TO_RUN = False
    print "Cannot test rtree contiguity weights, rtree not installed"


class TestRtreeContiguityWeights(unittest.TestCase):
    def setUp(self):
        """ Setup the rtree contiguity weights"""
        shpObj = pysal.open(pysal.examples.get_path('virginia.shp'), 'r')
        self.rtreeW = ContiguityWeights_rtree(shpObj, QUEEN)
        shpObj.close()

    def test_w_type(self):
        self.assert_(isinstance(self.rtreeW, ContiguityWeights_rtree))

    def test_QUEEN(self):
        self.assertEqual(QUEEN, 1)

    def test_ROOK(self):
        self.assertEqual(ROOK, 2)

    def test_ContiguityWeights_rtree(self):
        self.assert_(hasattr(self.rtreeW, 'w'))
        self.assert_(issubclass(dict, type(self.rtreeW.w)))
        self.assertEqual(len(self.rtreeW.w), 136)

    def test_nested_polygons(self):
        # load queen gal file created using Open Geoda.
        geodaW = pysal.open(
            pysal.examples.get_path('virginia.gal'), 'r').read()
        # build matching W with pysal
        pysalWr = self.build_W(
            pysal.examples.get_path('virginia.shp'), QUEEN, 'POLY_ID')
        # compare output.
        for key in geodaW.neighbors:
            geoda_neighbors = map(int, geodaW.neighbors[key])
            pysalr_neighbors = pysalWr.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalr_neighbors.sort()
            self.assertEqual(geoda_neighbors, pysalr_neighbors)

    def test_true_rook(self):
        # load rook gal file created using Open Geoda.
        geodaW = pysal.open(pysal.examples.get_path('rook31.gal'), 'r').read()
        # build matching W with pysal
        #pysalW = pysal.rook_from_shapefile(pysal.examples.get_path('rook31.shp'),'POLY_ID')
        pysalWr = self.build_W(
            pysal.examples.get_path('rook31.shp'), ROOK, 'POLY_ID')
        # compare output.
        for key in geodaW.neighbors:
            geoda_neighbors = map(int, geodaW.neighbors[key])
            pysalr_neighbors = pysalWr.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalr_neighbors.sort()
            self.assertEqual(geoda_neighbors, pysalr_neighbors)

    def test_true_rook2(self):
        # load rook gal file created using Open Geoda.
        geodaW = pysal.open(
            pysal.examples.get_path('stl_hom_rook.gal'), 'r').read()
        # build matching W with pysal
        pysalWr = self.build_W(pysal.examples.get_path(
            'stl_hom.shp'), ROOK, 'POLY_ID_OG')
        # compare output.
        for key in geodaW.neighbors:
            geoda_neighbors = map(int, geodaW.neighbors[key])
            pysalr_neighbors = pysalWr.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalr_neighbors.sort()
            self.assertEqual(geoda_neighbors, pysalr_neighbors)

    def test_true_rook3(self):
        # load rook gal file created using Open Geoda.
        geodaW = pysal.open(
            pysal.examples.get_path('sacramentot2.gal'), 'r').read()
        # build matching W with pysal
        pysalWr = self.build_W(pysal.examples.get_path(
            'sacramentot2.shp'), ROOK, 'POLYID')
        # compare output.
        for key in geodaW.neighbors:
            geoda_neighbors = map(int, geodaW.neighbors[key])
            pysalr_neighbors = pysalWr.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalr_neighbors.sort()
            self.assertEqual(geoda_neighbors, pysalr_neighbors)

    def test_true_rook4(self):
        # load rook gal file created using Open Geoda.
        geodaW = pysal.open(
            pysal.examples.get_path('virginia_rook.gal'), 'r').read()
        # build matching W with pysal
        pysalWr = self.build_W(
            pysal.examples.get_path('virginia.shp'), ROOK, 'POLY_ID')
        # compare output.
        for key in geodaW.neighbors:
            geoda_neighbors = map(int, geodaW.neighbors[key])
            pysalr_neighbors = pysalWr.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalr_neighbors.sort()
            self.assertEqual(geoda_neighbors, pysalr_neighbors)

    def build_W(self, shapefile, type, idVariable=None):
        """ Building 2 W's the hard way.  We need to do this so we can test both rtree and binning """
        dbname = os.path.splitext(shapefile)[0] + '.dbf'
        db = pysal.open(dbname)
        shpObj = pysal.open(shapefile)
        neighbor_data = ContiguityWeights_rtree(shpObj, type).w
        neighbors = {}
        weights = {}
        if idVariable:
            ids = db.by_col[idVariable]
            self.assertEqual(len(ids), len(set(ids)))
            for key in neighbor_data:
                id = ids[key]
                if id not in neighbors:
                    neighbors[id] = set()
                neighbors[id].update([ids[x] for x in neighbor_data[key]])
            for key in neighbors:
                neighbors[key] = list(neighbors[key])
            rtreeW = pysal.W(neighbors, id_order=ids)
        else:
            neighbors[key] = list(neighbors[key])
            rtreeW = pysal.W(neighbors)
        shpObj.seek(0)
        return rtreeW

suite = unittest.TestLoader().loadTestsFromTestCase(TestRtreeContiguityWeights)

if __name__ == '__main__' and OK_TO_RUN:
    runner = unittest.TextTestRunner()
    runner.run(suite)
