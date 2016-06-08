"""Unit test for util.py"""
import pysal
from pysal.common import *
import pysal.weights
import numpy as np
from scipy import sparse, float32
from scipy.spatial import KDTree
import os
import gc


class Testutil(unittest.TestCase):
    def setUp(self):
        self.w = pysal.rook_from_shapefile(
            pysal.examples.get_path('10740.shp'))

    def test_lat2W(self):
        w9 = pysal.lat2W(3, 3)
        self.assertEquals(w9.pct_nonzero, 29.62962962962963)
        self.assertEquals(w9[0], {1: 1.0, 3: 1.0})
        self.assertEquals(w9[3], {0: 1.0, 4: 1.0, 6: 1.0})

    def test_lat2SW(self):
        w9 = pysal.weights.lat2SW(3, 3)
        rows, cols = w9.shape
        n = rows * cols
        pct_nonzero = w9.nnz / float(n)
        self.assertEquals(pct_nonzero, 0.29629629629629628)
        data = w9.todense().tolist()
        self.assertEquals(data[0], [0, 1, 0, 1, 0, 0, 0, 0, 0])
        self.assertEquals(data[1], [1, 0, 1, 0, 1, 0, 0, 0, 0])
        self.assertEquals(data[2], [0, 1, 0, 0, 0, 1, 0, 0, 0])
        self.assertEquals(data[3], [1, 0, 0, 0, 1, 0, 1, 0, 0])
        self.assertEquals(data[4], [0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEquals(data[5], [0, 0, 1, 0, 1, 0, 0, 0, 1])
        self.assertEquals(data[6], [0, 0, 0, 1, 0, 0, 0, 1, 0])
        self.assertEquals(data[7], [0, 0, 0, 0, 1, 0, 1, 0, 1])
        self.assertEquals(data[8], [0, 0, 0, 0, 0, 1, 0, 1, 0])

    def test_block_weights(self):
        regimes = np.ones(25)
        regimes[range(10, 20)] = 2
        regimes[range(21, 25)] = 3
        regimes = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 3., 3.,
                            3., 3.])
        w = pysal.block_weights(regimes)
        ww0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertEquals(w.weights[0], ww0)
        wn0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
        self.assertEquals(w.neighbors[0], wn0)
        regimes = ['n', 'n', 's', 's', 'e', 'e', 'w', 'w', 'e']
        n = len(regimes)
        w = pysal.block_weights(regimes)
        wn = {0: [1], 1: [0], 2: [3], 3: [2], 4: [5, 8], 5: [4, 8],
              6: [7], 7: [6], 8: [4, 5]}
        self.assertEquals(w.neighbors, wn)
        ids = ['id-%i'%i for i in range(len(regimes))]
        w = pysal.block_weights(regimes, ids=np.array(ids))
        w0 = {'id-1': 1.0}
        self.assertEquals(w['id-0'], w0)
        w = pysal.block_weights(regimes, ids=ids)
        w0 = {'id-1': 1.0}
        self.assertEquals(w['id-0'], w0)

    def test_comb(self):
        x = range(4)
        l = []
        for i in pysal.comb(x, 2):
            l.append(i)
        lo = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        self.assertEquals(l, lo)

    def test_order(self):
        w3 = pysal.order(self.w, kmax=3)
        w3105 = [1, -1, 1, 2, 1]
        self.assertEquals(w3105, w3[1][0:5])

    def test_higher_order(self):
        w10 = pysal.lat2W(10, 10)
        w10_2 = pysal.higher_order(w10, 2)
        w10_20 = {2: 1.0, 11: 1.0, 20: 1.0}
        self.assertEquals(w10_20, w10_2[0])
        w5 = pysal.lat2W()
        w50 = {1: 1.0, 5: 1.0}
        self.assertEquals(w50, w5[0])
        w51 = {0: 1.0, 2: 1.0, 6: 1.0}
        self.assertEquals(w51, w5[1])
        w5_2 = pysal.higher_order(w5, 2)
        w5_20 = {2: 1.0, 10: 1.0, 6: 1.0}
        self.assertEquals(w5_20, w5_2[0])

    def test_shimbel(self):
        w5 = pysal.lat2W()
        w5_shimbel = pysal.shimbel(w5)
        w5_shimbel024 = 8
        self.assertEquals(w5_shimbel024, w5_shimbel[0][24])
        w5_shimbel004 = [-1, 1, 2, 3]
        self.assertEquals(w5_shimbel004, w5_shimbel[0][0:4])

    def test_full(self):
        neighbors = {'first': ['second'], 'second': ['first',
                                                     'third'], 'third': ['second']}
        weights = {'first': [1], 'second': [1, 1], 'third': [1]}
        w = pysal.W(neighbors, weights)
        wf, ids = pysal.full(w)
        wfo = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        np.testing.assert_array_almost_equal(wfo, wf, decimal=8)
        idso = ['first', 'second', 'third']
        self.assertEquals(idso, ids)

    def test_full2W(self):
        a = np.zeros((4, 4))
        for i in range(len(a)):
            for j in range(len(a[i])):
                if i != j:
                    a[i, j] = np.random.random(1)
        w = pysal.weights.util.full2W(a)
        np.testing.assert_array_equal(w.full()[0], a)
        ids = ['myID0', 'myID1', 'myID2', 'myID3']
        w = pysal.weights.util.full2W(a, ids=ids)
        np.testing.assert_array_equal(w.full()[0], a)
        w.full()[0] == a

    def test_WSP2W(self):
        sp = pysal.weights.lat2SW(2, 5)
        wsp = pysal.weights.WSP(sp)
        w = pysal.weights.WSP2W(wsp)
        self.assertEquals(w.n, 10)
        self.assertEquals(w[0], {1: 1, 5: 1})
        w = pysal.open(pysal.examples.get_path('sids2.gal'), 'r').read()
        wsp = pysal.weights.WSP(w.sparse, w.id_order)
        w = pysal.weights.WSP2W(wsp)
        self.assertEquals(w.n, 100)
        self.assertEquals(w['37135'], {'37001': 1.0, '37033': 1.0,
                                       '37037': 1.0, '37063': 1.0, '37145': 1.0})

    def test_insert_diagonal(self):
        w1 = pysal.weights.insert_diagonal(self.w)
        r1 = {0: 1.0, 1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        self.assertEquals(w1[0], r1)
        w1 = pysal.weights.insert_diagonal(self.w, 20)
        r1 = {0: 20, 1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        self.assertEquals(w1[0], r1)
        diag = np.arange(100, 100 + self.w.n)
        w1 = pysal.weights.insert_diagonal(self.w, diag)
        r1 = {0: 100, 1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        self.assertEquals(w1[0], r1)

    def test_remap_ids(self):
        w = pysal.lat2W(3, 2)
        wid_order = [0, 1, 2, 3, 4, 5]
        self.assertEquals(wid_order, w.id_order)
        wneighbors0 = [2, 1]
        self.assertEquals(wneighbors0, w.neighbors[0])
        old_to_new = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}
        w_new = pysal.remap_ids(w, old_to_new)
        w_newid_order = ['a', 'b', 'c', 'd', 'e', 'f']
        self.assertEquals(w_newid_order, w_new.id_order)
        w_newdneighborsa = ['c', 'b']
        self.assertEquals(w_newdneighborsa, w_new.neighbors['a'])

    def test_get_ids(self):
        polyids = pysal.weights.util.get_ids(
            pysal.examples.get_path('columbus.shp'), "POLYID")
        polyids5 = [1, 2, 3, 4, 5]
        self.assertEquals(polyids5, polyids[:5])

    def test_get_points_array_from_shapefile(self):
        xy = pysal.weights.util.get_points_array_from_shapefile(
            pysal.examples.get_path('juvenile.shp'))
        xy3 = np.array([[94., 93.], [80., 95.], [79., 90.]])
        np.testing.assert_array_almost_equal(xy3, xy[:3], decimal=8)
        xy = pysal.weights.util.get_points_array_from_shapefile(
            pysal.examples.get_path('columbus.shp'))
        xy3 = np.array([[8.82721847, 14.36907602], [8.33265837,
                                                    14.03162401], [9.01226541, 13.81971908]])
        np.testing.assert_array_almost_equal(xy3, xy[:3], decimal=8)

    def test_min_threshold_distance(self):
        x, y = np.indices((5, 5))
        x.shape = (25, 1)
        y.shape = (25, 1)
        data = np.hstack([x, y])
        mint = 1.0
        self.assertEquals(
            mint, pysal.weights.util.min_threshold_distance(data))

suite = unittest.TestLoader().loadTestsFromTestCase(Testutil)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
