"""Unit test for util.py"""
from ..util import lat2W, nonplanar_neighbors, fuzzy_contiguity
from .. import util
from ..weights import W, WSP
from ..distance import DistanceBand, KNN
from ..contiguity import Queen, Rook
from ...io.fileio import FileIO as psopen
from ... import examples
import numpy as np
import unittest


try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except:
    HAS_GEOPANDAS = False

class Testutil(unittest.TestCase):
    def setUp(self):
        self.w = Rook.from_shapefile(
            examples.get_path('10740.shp'))

    def test_lat2W(self):
        w9 = lat2W(3, 3)
        self.assertEqual(w9.pct_nonzero, 29.62962962962963)
        self.assertEqual(w9[0], {1: 1.0, 3: 1.0})
        self.assertEqual(w9[3], {0: 1.0, 4: 1.0, 6: 1.0})

    def test_lat2SW(self):
        w9 = util.lat2SW(3, 3)
        rows, cols = w9.shape
        n = rows * cols
        pct_nonzero = w9.nnz / float(n)
        self.assertEqual(pct_nonzero, 0.29629629629629628)
        data = w9.todense().tolist()
        self.assertEqual(data[0], [0, 1, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(data[1], [1, 0, 1, 0, 1, 0, 0, 0, 0])
        self.assertEqual(data[2], [0, 1, 0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(data[3], [1, 0, 0, 0, 1, 0, 1, 0, 0])
        self.assertEqual(data[4], [0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(data[5], [0, 0, 1, 0, 1, 0, 0, 0, 1])
        self.assertEqual(data[6], [0, 0, 0, 1, 0, 0, 0, 1, 0])
        self.assertEqual(data[7], [0, 0, 0, 0, 1, 0, 1, 0, 1])
        self.assertEqual(data[8], [0, 0, 0, 0, 0, 1, 0, 1, 0])

    def test_block_weights(self):
        regimes = np.ones(25)
        regimes[list(range(10, 20))] = 2
        regimes[list(range(21, 25))] = 3
        regimes = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 3., 3.,
                            3., 3.])
        w = util.block_weights(regimes)
        ww0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertEqual(w.weights[0], ww0)
        wn0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
        self.assertEqual(w.neighbors[0], wn0)
        regimes = ['n', 'n', 's', 's', 'e', 'e', 'w', 'w', 'e']
        w = util.block_weights(regimes)
        wn = {0: [1], 1: [0], 2: [3], 3: [2], 4: [5, 8], 5: [4, 8],
              6: [7], 7: [6], 8: [4, 5]}
        self.assertEqual(w.neighbors, wn)
        ids = ['id-%i'%i for i in range(len(regimes))]
        w = util.block_weights(regimes, ids=np.array(ids))
        w0 = {'id-1': 1.0}
        self.assertEqual(w['id-0'], w0)
        w = util.block_weights(regimes, ids=ids)
        w0 = {'id-1': 1.0}
        self.assertEqual(w['id-0'], w0)

    def test_comb(self):
        x = list(range(4))
        l = []
        for i in util.comb(x, 2):
            l.append(i)
        lo = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        self.assertEqual(l, lo)

    def test_order(self):
        w3 = util.order(self.w, kmax=3)
        w3105 = [1, -1, 1, 2, 1]
        self.assertEqual(w3105, w3[1][0:5])

    def test_higher_order(self):
        w10 = lat2W(10, 10)
        w10_2 = util.higher_order(w10, 2)
        w10_20 = {2: 1.0, 11: 1.0, 20: 1.0}
        self.assertEqual(w10_20, w10_2[0])
        w5 = lat2W()
        w50 = {1: 1.0, 5: 1.0}
        self.assertEqual(w50, w5[0])
        w51 = {0: 1.0, 2: 1.0, 6: 1.0}
        self.assertEqual(w51, w5[1])
        w5_2 = util.higher_order(w5, 2)
        w5_20 = {2: 1.0, 10: 1.0, 6: 1.0}
        self.assertEqual(w5_20, w5_2[0])
    
    def test_higher_order_classes(self):
        wdb = DistanceBand.from_shapefile(examples.get_path('baltim.shp'), 34)
        wknn = KNN.from_shapefile(examples.get_path('baltim.shp'), 10)
        wrook = Rook.from_shapefile(examples.get_path('columbus.shp'))
        wqueen = Queen.from_shapefile(examples.get_path('columbus.shp'))
        wsparse = wqueen.sparse
        ww = W(wknn.neighbors, wknn.weights)
        util.higher_order(wdb, 2)
        util.higher_order(wknn, 3)
        util.higher_order(wrook, 4)
        util.higher_order(wqueen, 5)
        util.higher_order(wsparse, 2)
        util.higher_order(ww, 2)
        ww.transform = 'r'
        wsparse_notbinary = wrook.sparse
        with self.assertRaises(ValueError):
            util.higher_order(wsparse, 2)
            util.higher_order(ww, 3)



    def test_shimbel(self):
        w5 = lat2W()
        w5_shimbel = util.shimbel(w5)
        w5_shimbel024 = 8
        self.assertEqual(w5_shimbel024, w5_shimbel[0][24])
        w5_shimbel004 = [-1, 1, 2, 3]
        self.assertEqual(w5_shimbel004, w5_shimbel[0][0:4])

    def test_full(self):
        neighbors = {'first': ['second'], 'second': ['first',
                                                     'third'], 'third': ['second']}
        weights = {'first': [1], 'second': [1, 1], 'third': [1]}
        w = W(neighbors, weights)
        wf, ids = util.full(w)
        wfo = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        np.testing.assert_array_almost_equal(wfo, wf, decimal=8)
        idso = ['first', 'second', 'third']
        self.assertEqual(idso, ids)

    def test_full2W(self):
        a = np.zeros((4, 4))
        for i in range(len(a)):
            for j in range(len(a[i])):
                if i != j:
                    a[i, j] = np.random.random(1)
        w = util.full2W(a)
        np.testing.assert_array_equal(w.full()[0], a)
        ids = ['myID0', 'myID1', 'myID2', 'myID3']
        w = util.full2W(a, ids=ids)
        np.testing.assert_array_equal(w.full()[0], a)
        w.full()[0] == a

    def test_WSP2W(self):
        sp = util.lat2SW(2, 5)
        wsp = WSP(sp)
        w = util.WSP2W(wsp)
        self.assertEqual(w.n, 10)
        self.assertEqual(w[0], {1: 1, 5: 1})
        w = psopen(examples.get_path('sids2.gal'), 'r').read()
        wsp = WSP(w.sparse, w.id_order)
        w = util.WSP2W(wsp)
        self.assertEqual(w.n, 100)
        self.assertEqual(w['37135'], {'37001': 1.0, '37033': 1.0,
                                       '37037': 1.0, '37063': 1.0, '37145': 1.0})

    def test_insert_diagonal(self):
        w1 = util.insert_diagonal(self.w)
        r1 = {0: 1.0, 1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        self.assertEqual(w1[0], r1)
        w1 = util.insert_diagonal(self.w, 20)
        r1 = {0: 20, 1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        self.assertEqual(w1[0], r1)
        diag = np.arange(100, 100 + self.w.n)
        w1 = util.insert_diagonal(self.w, diag)
        r1 = {0: 100, 1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        self.assertEqual(w1[0], r1)

    def test_remap_ids(self):
        w = lat2W(3, 2)
        wid_order = [0, 1, 2, 3, 4, 5]
        self.assertEqual(wid_order, w.id_order)
        wneighbors0 = [2, 1]
        self.assertEqual(wneighbors0, w.neighbors[0])
        old_to_new = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}
        w_new = util.remap_ids(w, old_to_new)
        w_newid_order = ['a', 'b', 'c', 'd', 'e', 'f']
        self.assertEqual(w_newid_order, w_new.id_order)
        w_newdneighborsa = ['c', 'b']
        self.assertEqual(w_newdneighborsa, w_new.neighbors['a'])

    def test_get_ids_shp(self):
        polyids = util.get_ids(
            examples.get_path('columbus.shp'), "POLYID")
        polyids5 = [1, 2, 3, 4, 5]
        self.assertEqual(polyids5, polyids[:5])

    @unittest.skipIf(not HAS_GEOPANDAS, "Missing geopandas, cannot test get_ids with gdf")
    def test_get_ids_gdf(self):
        gdf = gpd.read_file(examples.get_path('columbus.shp'))
        polyids = util.get_ids(gdf, "POLYID")
        polyids5 = [1, 2, 3, 4, 5]
        self.assertEqual(polyids5, polyids[:5])
    
    def test_get_points_array_from_shapefile(self):
        xy = util.get_points_array_from_shapefile(
            examples.get_path('juvenile.shp'))
        xy3 = np.array([[94., 93.], [80., 95.], [79., 90.]])
        np.testing.assert_array_almost_equal(xy3, xy[:3], decimal=8)
        xy = util.get_points_array_from_shapefile(
            examples.get_path('columbus.shp'))
        xy3 = np.array([[8.82721847, 14.36907602], [8.33265837,
                                                    14.03162401], [9.01226541, 13.81971908]])
        np.testing.assert_array_almost_equal(xy3, xy[:3], decimal=8)

    def test_min_threshold_distance(self):
        x, y = np.indices((5, 5))
        x.shape = (25, 1)
        y.shape = (25, 1)
        data = np.hstack([x, y])
        mint = 1.0
        self.assertEqual(
            mint, util.min_threshold_distance(data))

    def test_attach_islands(self):
        w = Rook.from_shapefile(examples.get_path('10740.shp'))
        w_knn1 = KNN.from_shapefile(examples.get_path('10740.shp'), k=1)
        w_attach = util.attach_islands(w, w_knn1)
        self.assertEqual(w_attach.islands, [])
        self.assertEqual(w_attach[w.islands[0]], {166: 1.0})

    @unittest.skipIf(not HAS_GEOPANDAS, "Missing geopandas, cannot test nonplanar neighbors")
    def test_nonplanar_neighbors(self):
        df = gpd.read_file(examples.get_path('map_RS_BR.shp'))
        w = Queen.from_dataframe(df)
        self.assertEqual(w.islands, [0, 4, 23, 27, 80, 94, 101, 107, 109, 119, 122, 139, 169, 175, 223, 239, 247, 253, 254, 255, 256, 261, 276, 291, 294, 303, 321, 357, 374])
        wnp = nonplanar_neighbors(w, df)
        self.assertEqual(wnp.islands, [])
        self.assertEqual(w.neighbors[0], [])
        self.assertEqual(wnp.neighbors[0], [23, 59, 152, 239])
        self.assertEqual(wnp.neighbors[23], [0, 45, 59, 107, 152, 185, 246])

    @unittest.skipIf(not HAS_GEOPANDAS, "Missing geopandas, cannot test fuzzy_contiguity")
    def test_fuzzy_contiguity(self):
        rs = examples.get_path('map_RS_BR.shp')
        rs_df = gpd.read_file(rs)
        wf = fuzzy_contiguity(rs_df)
        self.assertEqual(wf.islands, [])
        self.assertEqual(set(wf.neighbors[0]), set([239, 59, 152, 23, 107]))


suite = unittest.TestLoader().loadTestsFromTestCase(Testutil)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
