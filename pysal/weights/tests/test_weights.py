import unittest
import pysal
import numpy as np

NPTA3E = np.testing.assert_array_almost_equal

class TestW(unittest.TestCase):
    def setUp(self):
        from pysal import rook_from_shapefile
        self.w = rook_from_shapefile("../../examples/10740.shp")

        self.neighbors={0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3,
            7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
        self.weights={0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1], 4: [1, 1,
            1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1], 8: [1, 1]}

        self.w3x3 = pysal.lat2W(3,3)

    def test___getitem__(self):
        self.assertEqual(self.w[0], {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0})


    def test___init__(self):
        w = pysal.W(self.neighbors, self.weights)
        self.assertEqual(w.pct_nonzero, 0.29629629629629628)


    def test___iter__(self):
        w = pysal.lat2W(3,3)
        res = {}
        for i,wi in enumerate(w):
            res[i] = wi
        self.assertEqual(res[0], {1: 1.0, 3: 1.0})
        self.assertEqual(res[8], {5: 1.0, 7: 1.0})

    def test_asymmetries(self):
        w = pysal.lat2W(3,3)
        w.transform = 'r'
        result = w.asymmetry()[0:2]
        NPTA3E(result[0], np.array( [1, 3, 0, 2, 4, 1,
            5, 0, 4, 6, 1, 3, 5, 7, 2, 4, 8, 3, 7, 4, 6, 8, 5, 7]))

    def test_asymmetry(self):
        w = pysal.lat2W(3,3)
        self.assertEqual(w.asymmetry(), [])
        w.transform = 'r'
        self.assertFalse(w.asymmetry() == [])

    def test_cardinalities(self):
        w = pysal.lat2W(3,3)
        self.assertEqual(w.cardinalities, {0: 2, 1: 3, 2: 2, 3: 3, 4: 4, 5: 3,
            6: 2, 7: 3, 8: 2})

    def test_diagW2(self):
        NPTA3E(self.w3x3.diagW2, np.array([ 2.,  3.,  2.,  3.,  4.,  3.,  2.,
            3.,  2.]))
    """
    def test_diagWtW(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.diagWtW())
        assert False # TODO: implement your test here

    def test_diagWtW_WW(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.diagWtW_WW())
        assert False # TODO: implement your test here

    def test_full(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.full())
        assert False # TODO: implement your test here

    def test_get_transform(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.get_transform())
        assert False # TODO: implement your test here

    def test_higher_order(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.higher_order(k))
        assert False # TODO: implement your test here

    def test_histogram(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.histogram())
        assert False # TODO: implement your test here

    def test_id2i(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.id2i())
        assert False # TODO: implement your test here

    def test_id_order_set(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.id_order_set())
        assert False # TODO: implement your test here

    def test_islands(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.islands())
        assert False # TODO: implement your test here

    def test_max_neighbors(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.max_neighbors())
        assert False # TODO: implement your test here

    def test_mean_neighbors(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.mean_neighbors())
        assert False # TODO: implement your test here

    def test_min_neighbors(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.min_neighbors())
        assert False # TODO: implement your test here

    def test_n(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.n())
        assert False # TODO: implement your test here

    def test_neighbor_offsets(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.neighbor_offsets())
        assert False # TODO: implement your test here

    def test_nonzero(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.nonzero())
        assert False # TODO: implement your test here

    def test_order(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.order(kmax))
        assert False # TODO: implement your test here

    def test_pct_nonzero(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.pct_nonzero())
        assert False # TODO: implement your test here

    def test_s0(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.s0())
        assert False # TODO: implement your test here

    def test_s1(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.s1())
        assert False # TODO: implement your test here

    def test_s2(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.s2())
        assert False # TODO: implement your test here

    def test_s2array(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.s2array())
        assert False # TODO: implement your test here

    def test_sd(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.sd())
        assert False # TODO: implement your test here

    def test_set_transform(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.set_transform(value))
        assert False # TODO: implement your test here

    def test_shimbel(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.shimbel())
        assert False # TODO: implement your test here

    def test_sparse(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.sparse())
        assert False # TODO: implement your test here

    def test_trcW2(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.trcW2())
        assert False # TODO: implement your test here

    def test_trcWtW(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.trcWtW())
        assert False # TODO: implement your test here

    def test_trcWtW_WW(self):
        # w = W(neighbors, weights, id_order)
        # self.assertEqual(expected, w.trcWtW_WW())
        assert False # TODO: implement your test here
    """

if __name__ == '__main__':
    unittest.main()
