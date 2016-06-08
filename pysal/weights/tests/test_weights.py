import unittest
import pysal
import numpy as np

NPTA3E = np.testing.assert_array_almost_equal


class TestW(unittest.TestCase):
    def setUp(self):
        from pysal import rook_from_shapefile
        self.w = rook_from_shapefile(pysal.examples.get_path('10740.shp'))

        self.neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4],
                          4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7],
                          7: [4, 6, 8], 8: [5, 7]}
        self.weights = {0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1],
                        4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1],
                        7: [1, 1, 1], 8: [1, 1]}

        self.w3x3 = pysal.lat2W(3, 3)

    def test_W(self):
        w = pysal.W(self.neighbors, self.weights)
        self.assertEqual(w.pct_nonzero, 29.62962962962963)

    def test___getitem__(self):
        self.assertEqual(
            self.w[0], {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0})

    def test___init__(self):
        w = pysal.W(self.neighbors, self.weights)
        self.assertEqual(w.pct_nonzero, 29.62962962962963)

    def test___iter__(self):
        w = pysal.lat2W(3, 3)
        res = {}
        for i, wi in enumerate(w):
            res[i] = wi
        self.assertEqual(res[0], (0, {1: 1.0, 3: 1.0}))
        self.assertEqual(res[8], (8, {5: 1.0, 7: 1.0}))

    def test_asymmetries(self):
        w = pysal.lat2W(3, 3)
        w.transform = 'r'
        result = w.asymmetry()
        self.assertEqual(result, [(0, 1), (0, 3), (1, 0), (1, 2),
                                  (1, 4), (2, 1), (2, 5), (3, 0),
                                  (3, 4), (3, 6), (4, 1), (4, 3),
                                  (4, 5), (4, 7), (5, 2), (5, 4),
                                  (5, 8), (6, 3), (6, 7), (7, 4),
                                  (7, 6), (7, 8), (8, 5), (8, 7)])

    def test_asymmetry(self):
        w = pysal.lat2W(3, 3)
        self.assertEqual(w.asymmetry(), [])
        w.transform = 'r'
        self.assertFalse(w.asymmetry() == [])

    def test_cardinalities(self):
        w = pysal.lat2W(3, 3)
        self.assertEqual(w.cardinalities, {0: 2, 1: 3, 2: 2, 3: 3, 4: 4, 5: 3,
                                           6: 2, 7: 3, 8: 2})

    def test_diagW2(self):
        NPTA3E(self.w3x3.diagW2, np.array([2., 3., 2., 3., 4., 3., 2.,
                                           3., 2.]))

    def test_diagWtW(self):
        NPTA3E(self.w3x3.diagW2, np.array([2., 3., 2., 3., 4., 3., 2.,
                                           3., 2.]))

    def test_diagWtW_WW(self):
        NPTA3E(self.w3x3.diagWtW_WW, np.array([4., 6., 4., 6., 8.,
                                               6., 4., 6., 4.]))

    def test_full(self):
        wf = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0.],
                       [1., 0., 1., 0., 1., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                       [1., 0., 0., 0., 1., 0., 1., 0., 0.],
                       [0., 1., 0., 1., 0., 1., 0., 1., 0.],
                       [0., 0., 1., 0., 1., 0., 0., 0., 1.],
                       [0., 0., 0., 1., 0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 1., 0., 1., 0., 1.],
                       [0., 0., 0., 0., 0., 1., 0., 1., 0.]])
        ids = range(9)

        wf1, ids1 = self.w3x3.full()
        NPTA3E(wf1, wf)
        self.assertEqual(ids1, ids)

    def test_get_transform(self):
        self.assertEqual(self.w3x3.transform, 'O')
        self.w3x3.transform = 'r'
        self.assertEqual(self.w3x3.transform, 'R')
        self.w3x3.transform = 'b'

    def test_higher_order(self):
        weights = {0: [1.0, 1.0, 1.0], 1: [1.0, 1.0, 1.0], 2: [1.0, 1.0, 1.0],
                   3: [1.0, 1.0, 1.0], 4: [1.0, 1.0, 1.0, 1.0],
                   5: [1.0, 1.0, 1.0], 6: [1.0, 1.0, 1.0], 7:
                   [1.0, 1.0, 1.0], 8: [1.0, 1.0, 1.0]}
        neighbors = {0: [4, 6, 2], 1: [3, 5, 7], 2: [8, 0, 4], 3: [7, 1, 5],
                     4: [8, 0, 2, 6], 5: [1, 3, 7], 6: [4, 0, 8], 7: [3, 1, 5],
                     8: [6, 2, 4]}
        wneighbs = {k: {neighb: weights[k][i] for i, neighb in enumerate(v)}
                    for k, v in neighbors.iteritems()}
        w2 = pysal.higher_order(self.w3x3, 2)
        test_wneighbs = {k: {ne: weights[k][i] for i, ne in enumerate(v)}
                         for k, v in w2.neighbors.iteritems()}
        self.assertEqual(test_wneighbs, wneighbs)

    def test_histogram(self):
        hist = [(0, 1), (1, 1), (2, 4), (3, 20), (4, 57), (5, 44), (6, 36),
                (7, 15), (8, 7), (9, 1), (10, 6), (11, 0), (12, 2), (13, 0),
                (14, 0), (15, 1)]
        self.assertEqual(self.w.histogram, hist)

    def test_id2i(self):
        id2i = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
        self.assertEqual(self.w3x3.id2i, id2i)

    def test_id_order_set(self):
        w = pysal.W(neighbors={'a': ['b'], 'b': ['a', 'c'], 'c': ['b']})
        self.assertFalse(w.id_order_set)

    def test_islands(self):
        w = pysal.W(neighbors={'a': ['b'], 'b': ['a', 'c'], 'c':
                               ['b'], 'd': []})
        self.assertEqual(w.islands, ['d'])
        self.assertEqual(self.w3x3.islands, [])

    def test_max_neighbors(self):
        w = pysal.W(neighbors={'a': ['b'], 'b': ['a', 'c'], 'c':
                               ['b'], 'd': []})
        self.assertEqual(w.max_neighbors, 2)
        self.assertEqual(self.w3x3.max_neighbors, 4)

    def test_mean_neighbors(self):
        w = pysal.lat2W()
        self.assertEqual(w.mean_neighbors, 3.2)

    def test_min_neighbors(self):
        w = pysal.lat2W()
        self.assertEqual(w.min_neighbors, 2)

    def test_n(self):
        w = pysal.lat2W()
        self.assertEqual(w.n, 25)

    def test_neighbor_offsets(self):
        d = {0: [3, 1],
             1: [0, 4, 2],
             2: [1, 5],
             3: [0, 6, 4],
             4: [1, 3, 7, 5],
             5: [2, 4, 8],
             6: [3, 7],
             7: [4, 6, 8],
             8: [5, 7]}

        self.assertEqual(self.w3x3.neighbor_offsets, d)

    def test_nonzero(self):
        self.assertEquals(self.w3x3.nonzero, 24)

    def test_order(self):
        w = pysal.lat2W(3, 3)
        o = {0: [-1, 1, 2, 1, 2, 3, 2, 3, 0],
             1: [1, -1, 1, 2, 1, 2, 3, 2, 3],
             2: [2, 1, -1, 3, 2, 1, 0, 3, 2],
             3: [1, 2, 3, -1, 1, 2, 1, 2, 3],
             4: [2, 1, 2, 1, -1, 1, 2, 1, 2],
             5: [3, 2, 1, 2, 1, -1, 3, 2, 1],
             6: [2, 3, 0, 1, 2, 3, -1, 1, 2],
             7: [3, 2, 3, 2, 1, 2, 1, -1, 1],
             8: [0, 3, 2, 3, 2, 1, 2, 1, -1]}
        self.assertEquals(pysal.order(w), o)

    def test_pct_nonzero(self):
        self.assertEqual(self.w3x3.pct_nonzero, 29.62962962962963)

    def test_s0(self):
        self.assertEqual(self.w3x3.s0, 24.0)

    def test_s1(self):
        self.assertEqual(self.w3x3.s1, 48.0)

    def test_s2(self):
        self.assertEqual(self.w3x3.s2, 272.0)

    def test_s2array(self):
        s2a = np.array([[16.], [36.], [16.], [36.],
                        [64.], [36.], [16.], [36.], [16.]])
        NPTA3E(self.w3x3.s2array, s2a)

    def test_sd(self):
        self.assertEquals(self.w3x3.sd, 0.66666666666666663)

    def test_set_transform(self):
        w = pysal.lat2W(2, 2)
        self.assertEqual(w.transform, 'O')
        self.assertEquals(w.weights[0], [1.0, 1.0])
        w.transform = 'r'
        self.assertEquals(w.weights[0], [0.5, 0.5])

    def test_shimbel(self):
        d = {0: [-1, 1, 2, 1, 2, 3, 2, 3, 4],
             1: [1, -1, 1, 2, 1, 2, 3, 2, 3],
             2: [2, 1, -1, 3, 2, 1, 4, 3, 2],
             3: [1, 2, 3, -1, 1, 2, 1, 2, 3],
             4: [2, 1, 2, 1, -1, 1, 2, 1, 2],
             5: [3, 2, 1, 2, 1, -1, 3, 2, 1],
             6: [2, 3, 4, 1, 2, 3, -1, 1, 2],
             7: [3, 2, 3, 2, 1, 2, 1, -1, 1],
             8: [4, 3, 2, 3, 2, 1, 2, 1, -1]}
        self.assertEquals(pysal.shimbel(self.w3x3), d)

    def test_sparse(self):
        self.assertEqual(self.w3x3.sparse.nnz, 24)

    def test_trcW2(self):
        self.assertEqual(self.w3x3.trcW2, 24.)

    def test_trcWtW(self):
        self.assertEqual(self.w3x3.trcWtW, 24.)

    def test_trcWtW_WW(self):
        self.assertEqual(self.w3x3.trcWtW_WW, 48.)


class Test_WSP_Back_To_W(unittest.TestCase):
    # Test to make sure we get back to the same W functionality
    def setUp(self):
        from pysal import rook_from_shapefile
        self.w = rook_from_shapefile(pysal.examples.get_path('10740.shp'))
        wsp = pysal.weights.WSP(self.w.sparse, self.w.id_order)
        self.w = pysal.weights.WSP2W(wsp)

        self.neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4],
                          4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7],
                          7: [4, 6, 8], 8: [5, 7]}
        self.weights = {0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1],
                        4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1],
                        8: [1, 1]}

        self.w3x3 = pysal.lat2W(3, 3)
        w3x3 = pysal.weights.WSP(self.w3x3.sparse, self.w3x3.id_order)
        self.w3x3 = pysal.weights.WSP2W(w3x3)

    def test_W(self):
        w = pysal.W(self.neighbors, self.weights)
        self.assertEqual(w.pct_nonzero, 29.62962962962963)

    def test___getitem__(self):
        self.assertEqual(
            self.w[0], {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0})

    def test___init__(self):
        w = pysal.W(self.neighbors, self.weights)
        self.assertEqual(w.pct_nonzero, 29.62962962962963)

    def test___iter__(self):
        w = pysal.lat2W(3, 3)
        res = {}
        for i, wi in enumerate(w):
            res[i] = wi
        self.assertEqual(res[0], (0, {1: 1.0, 3: 1.0}))
        self.assertEqual(res[8], (8, {5: 1.0, 7: 1.0}))

    def test_asymmetries(self):
        w = pysal.lat2W(3, 3)
        w.transform = 'r'
        result = w.asymmetry()
        self.assertEqual(result, [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4),
                                  (2, 1), (2, 5), (3, 0), (3, 4), (3, 6),
                                  (4, 1), (4, 3), (4, 5), (4, 7), (5, 2),
                                  (5, 4), (5, 8), (6, 3), (6, 7), (7, 4),
                                  (7, 6), (7, 8), (8, 5), (8, 7)])

    def test_asymmetry(self):
        w = pysal.lat2W(3, 3)
        self.assertEqual(w.asymmetry(), [])
        w.transform = 'r'
        self.assertFalse(w.asymmetry() == [])

    def test_cardinalities(self):
        w = pysal.lat2W(3, 3)
        self.assertEqual(w.cardinalities, {0: 2, 1: 3, 2: 2, 3: 3, 4: 4, 5: 3,
                                           6: 2, 7: 3, 8: 2})

    def test_diagW2(self):
        NPTA3E(self.w3x3.diagW2, np.array([2., 3., 2., 3., 4., 3., 2.,
                                           3., 2.]))

    def test_diagWtW(self):
        NPTA3E(self.w3x3.diagW2, np.array([2., 3., 2., 3., 4., 3., 2.,
                                           3., 2.]))

    def test_diagWtW_WW(self):
        NPTA3E(self.w3x3.diagWtW_WW, np.array([4., 6., 4., 6., 8.,
                                               6., 4., 6., 4.]))

    def test_full(self):
        wf = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0.],
                       [1., 0., 1., 0., 1., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                       [1., 0., 0., 0., 1., 0., 1., 0., 0.],
                       [0., 1., 0., 1., 0., 1., 0., 1., 0.],
                       [0., 0., 1., 0., 1., 0., 0., 0., 1.],
                       [0., 0., 0., 1., 0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 1., 0., 1., 0., 1.],
                       [0., 0., 0., 0., 0., 1., 0., 1., 0.]])
        ids = range(9)

        wf1, ids1 = self.w3x3.full()
        NPTA3E(wf1, wf)
        self.assertEqual(ids1, ids)

    def test_get_transform(self):
        self.assertEqual(self.w3x3.transform, 'O')
        self.w3x3.transform = 'r'
        self.assertEqual(self.w3x3.transform, 'R')
        self.w3x3.transform = 'b'

    def test_higher_order(self):
        weights = {0: [1.0, 1.0, 1.0], 1: [1.0, 1.0, 1.0], 2: [1.0, 1.0, 1.0],
                   3: [1.0, 1.0, 1.0], 4: [1.0, 1.0, 1.0, 1.0],
                   5: [1.0, 1.0, 1.0], 6: [1.0, 1.0, 1.0],
                   7: [1.0, 1.0, 1.0], 8: [1.0, 1.0, 1.0]}
        neighbors = {0: [4, 6, 2], 1: [3, 5, 7], 2: [8, 0, 4], 3: [7, 1, 5],
                     4: [8, 0, 2, 6], 5: [1, 3, 7], 6: [4, 0, 8], 7: [3, 1, 5],
                     8: [6, 2, 4]}
        wneighbs = {k: {neighb: weights[k][i] for i, neighb in enumerate(v)}
                    for k, v in neighbors.iteritems()}
        w2 = pysal.higher_order(self.w3x3, 2)
        test_wneighbs = {k: {ne: w2.weights[k][i] for i, ne in enumerate(v)}
                         for k, v in w2.neighbors.iteritems()}
        self.assertEqual(test_wneighbs, wneighbs)

    def test_histogram(self):
        hist = [(0, 1), (1, 1), (2, 4), (3, 20), (4, 57), (5, 44), (6, 36),
                (7, 15), (8, 7), (9, 1), (10, 6), (11, 0), (12, 2), (13, 0),
                (14, 0), (15, 1)]
        self.assertEqual(self.w.histogram, hist)

    def test_id2i(self):
        id2i = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
        self.assertEqual(self.w3x3.id2i, id2i)

    def test_id_order_set(self):
        w = pysal.W(neighbors={'a': ['b'], 'b': ['a', 'c'], 'c': ['b']})
        self.assertFalse(w.id_order_set)

    def test_islands(self):
        w = pysal.W(neighbors={'a': ['b'], 'b': ['a', 'c'], 'c':
                               ['b'], 'd': []})
        self.assertEqual(w.islands, ['d'])
        self.assertEqual(self.w3x3.islands, [])

    def test_max_neighbors(self):
        w = pysal.W(neighbors={'a': ['b'], 'b': ['a', 'c'], 'c':
                               ['b'], 'd': []})
        self.assertEqual(w.max_neighbors, 2)
        self.assertEqual(self.w3x3.max_neighbors, 4)

    def test_mean_neighbors(self):
        w = pysal.lat2W()
        self.assertEqual(w.mean_neighbors, 3.2)

    def test_min_neighbors(self):
        w = pysal.lat2W()
        self.assertEqual(w.min_neighbors, 2)

    def test_n(self):
        w = pysal.lat2W()
        self.assertEqual(w.n, 25)

    def test_nonzero(self):
        self.assertEquals(self.w3x3.nonzero, 24)

    def test_order(self):
        w = pysal.lat2W(3, 3)
        o = {0: [-1, 1, 2, 1, 2, 3, 2, 3, 0],
             1: [1, -1, 1, 2, 1, 2, 3, 2, 3],
             2: [2, 1, -1, 3, 2, 1, 0, 3, 2],
             3: [1, 2, 3, -1, 1, 2, 1, 2, 3],
             4: [2, 1, 2, 1, -1, 1, 2, 1, 2],
             5: [3, 2, 1, 2, 1, -1, 3, 2, 1],
             6: [2, 3, 0, 1, 2, 3, -1, 1, 2],
             7: [3, 2, 3, 2, 1, 2, 1, -1, 1],
             8: [0, 3, 2, 3, 2, 1, 2, 1, -1]}
        self.assertEquals(pysal.order(w), o)

    def test_pct_nonzero(self):
        self.assertEqual(self.w3x3.pct_nonzero, 29.62962962962963)

    def test_s0(self):
        self.assertEqual(self.w3x3.s0, 24.0)

    def test_s1(self):
        self.assertEqual(self.w3x3.s1, 48.0)

    def test_s2(self):
        self.assertEqual(self.w3x3.s2, 272.0)

    def test_s2array(self):
        s2a = np.array([[16.], [36.], [16.], [36.],
                        [64.], [36.], [16.], [36.], [16.]])
        NPTA3E(self.w3x3.s2array, s2a)

    def test_sd(self):
        self.assertEquals(self.w3x3.sd, 0.66666666666666663)

    def test_set_transform(self):
        w = pysal.lat2W(2, 2)
        self.assertEqual(w.transform, 'O')
        self.assertEquals(w.weights[0], [1.0, 1.0])
        w.transform = 'r'
        self.assertEquals(w.weights[0], [0.5, 0.5])

    def test_shimbel(self):
        d = {0: [-1, 1, 2, 1, 2, 3, 2, 3, 4],
             1: [1, -1, 1, 2, 1, 2, 3, 2, 3],
             2: [2, 1, -1, 3, 2, 1, 4, 3, 2],
             3: [1, 2, 3, -1, 1, 2, 1, 2, 3],
             4: [2, 1, 2, 1, -1, 1, 2, 1, 2],
             5: [3, 2, 1, 2, 1, -1, 3, 2, 1],
             6: [2, 3, 4, 1, 2, 3, -1, 1, 2],
             7: [3, 2, 3, 2, 1, 2, 1, -1, 1],
             8: [4, 3, 2, 3, 2, 1, 2, 1, -1]}
        self.assertEquals(pysal.shimbel(self.w3x3), d)

    def test_sparse(self):
        self.assertEqual(self.w3x3.sparse.nnz, 24)

    def test_trcW2(self):
        self.assertEqual(self.w3x3.trcW2, 24.)

    def test_trcWtW(self):
        self.assertEqual(self.w3x3.trcWtW, 24.)

    def test_trcWtW_WW(self):
        self.assertEqual(self.w3x3.trcWtW_WW, 48.)


class TestWSP(unittest.TestCase):
    def setUp(self):
        self.w = pysal.open(pysal.examples.get_path("sids2.gal")).read()
        self.wsp = pysal.weights.WSP(self.w.sparse, self.w.id_order)
        w3x3 = pysal.lat2W(3, 3)
        self.w3x3 = pysal.weights.WSP(w3x3.sparse)

    def test_WSP(self):
        self.assertEquals(self.w.id_order, self.wsp.id_order)
        self.assertEquals(self.w.n, self.wsp.n)
        np.testing.assert_array_equal(
            self.w.sparse.todense(), self.wsp.sparse.todense())

    def test_diagWtW_WW(self):
        NPTA3E(self.w3x3.diagWtW_WW, np.array([4., 6., 4., 6., 8.,
                                               6., 4., 6., 4.]))

    def test_trcWtW_WW(self):
        self.assertEqual(self.w3x3.trcWtW_WW, 48.)

    def test_s0(self):
        self.assertEqual(self.w3x3.s0, 24.0)


if __name__ == '__main__':
    unittest.main()
