import os
import unittest
import pysal
from pysal.weights._contW_rtree import ContiguityWeights_rtree,QUEEN,ROOK
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
        pysalWr,pysalWb = self.build_W('../examples/virginia.shp',QUEEN,'POLY_ID')
        # compare output.
        for key in geodaW.neighbors:
            geoda_neighbors = map(int,geodaW.neighbors[key])
            pysalr_neighbors = pysalWr.neighbors[int(key)]
            pysalb_neighbors = pysalWb.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalr_neighbors.sort()
            pysalb_neighbors.sort()
            self.assertEqual(geoda_neighbors,pysalr_neighbors)
            self.assertEqual(geoda_neighbors,pysalb_neighbors)
    def test_true_rook2(self):
        # load queen gal file created using Open Geoda.
        geodaW = pysal.open('../examples/stl_hom_rook.gal','r').read()
        # build matching W with pysal
        pysalWr,pysalWb = self.build_W('../examples/stl_hom.shp',ROOK,'POLY_ID_OG')
        # compare output.
        for key in geodaW.neighbors:
            geoda_neighbors = map(int,geodaW.neighbors[key])
            pysalr_neighbors = pysalWr.neighbors[int(key)]
            pysalb_neighbors = pysalWb.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalr_neighbors.sort()
            pysalb_neighbors.sort()
            self.assertEqual(geoda_neighbors,pysalr_neighbors)
            self.assertEqual(geoda_neighbors,pysalb_neighbors)
    def test_true_rook(self):
        # load queen gal file created using Open Geoda.
        geodaW = pysal.open('../examples/rook31.gal','r').read()
        # build matching W with pysal
        #pysalW = pysal.rook_from_shapefile('../examples/rook31.shp','POLY_ID')
        pysalWr,pysalWb = self.build_W('../examples/rook31.shp',ROOK,'POLY_ID')
        # compare output.
        for key in geodaW.neighbors:
            geoda_neighbors = map(int,geodaW.neighbors[key])
            pysalr_neighbors = pysalWr.neighbors[int(key)]
            pysalb_neighbors = pysalWb.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalr_neighbors.sort()
            pysalb_neighbors.sort()
            self.assertEqual(geoda_neighbors,pysalr_neighbors)
            self.assertEqual(geoda_neighbors,pysalb_neighbors)
    def build_W(self,shapefile,type,idVariable=None):
        """ Building 2 W's the hard way.  We need to do this so we can test both rtree and binning """
        dbname = os.path.splitext(shapefile)[0]+'.dbf'
        db = pysal.open(dbname)
        shpObj = pysal.open(shapefile)
        neighbor_data = ContiguityWeights_rtree(shpObj,type).w
        neighbors={}
        weights={}
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
            rtreeW = pysal.W(neighbors,id_order=ids)
        else:
            neighbors[key] = list(neighbors[key])
            rtreeW = pysal.W(neighbors)
        shpObj.seek(0)
        neighbor_data = ContiguityWeights_binning(shpObj,type).w
        neighbors={}
        weights={}
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
            binningW = pysal.W(neighbors,id_order=ids)
        else:
            neighbors[key] = list(neighbors[key])
            binningW = pysal.W(neighbors)
        return rtreeW,binningW

        
        

suite = unittest.TestLoader().loadTestsFromTestCase(_TestContiguityWeights)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
