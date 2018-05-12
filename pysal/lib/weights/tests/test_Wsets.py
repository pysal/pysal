"""Unit test for Wsets module."""
import unittest
from ..util import lat2W, block_weights
from .. import Wsets
import numpy as np

class TestWsets(unittest.TestCase):
    """Unit test for Wsets module."""

    def test_w_union(self):
        """Unit test"""
        w1 = lat2W(4, 4)
        w2 = lat2W(6, 4)
        w3 = Wsets.w_union(w1, w2)
        self.assertEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w3.neighbors[15]), set([19, 11, 14]))

    def test_w_intersection(self):
        """Unit test"""
        w1 = lat2W(4, 4)
        w2 = lat2W(6, 4)
        w3 = Wsets.w_union(w1, w2)
        self.assertEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w3.neighbors[15]), set([19, 11, 14]))

    def test_w_difference(self):
        """Unit test"""
        w1 = lat2W(4, 4, rook=False)
        w2 = lat2W(4, 4, rook=True)
        w3 = Wsets.w_difference(w1, w2, constrained=False)
        self.assertNotEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([10, 11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14]))
        self.assertEqual(set(w3.neighbors[15]), set([10]))

    def test_w_symmetric_difference(self):
        """Unit test"""
        w1 = lat2W(4, 4, rook=False)
        w2 = lat2W(6, 4, rook=True)
        w3 = Wsets.w_symmetric_difference(
            w1, w2, constrained=False)
        self.assertNotEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([10, 11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w3.neighbors[15]), set([10, 19]))

    def test_w_subset(self):
        """Unit test"""
        w1 = lat2W(6, 4)
        ids = list(range(16))
        w2 = Wsets.w_subset(w1, ids)
        self.assertEqual(w1[0], w2[0])
        self.assertEqual(set(w1.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14]))

    def test_w_clip(self):
        """Unit test for w_clip"""
        w1 = lat2W(3, 2, rook=False)
        w1.transform = 'R'
        w2 = block_weights(['r1', 'r2', 'r1', 'r1', 'r1', 'r2'])
        w2.transform = 'R'
        wcs = Wsets.w_clip(w1, w2, outSP=True)
        expected_wcs = np.array([[ 0.,  0.,0.33333333,0.33333333, 0.,0.],
                                 [ 0.,  0.,  0.,  0.,  0.,0. ],
                                 [ 0.2,  0.,  0.,  0.2,  0.2,0.],
                                 [ 0.2,  0.,  0.2,  0.,  0.2,0.],
                                 [ 0.,  0., 0.33333333, 0.33333333, 0., 0.],
                                 [ 0.,  0.,  0.,  0.,  0.,0.]])
        np.testing.assert_array_equal(np.around(wcs.sparse.toarray(),decimals=8), expected_wcs)

        wc = Wsets.w_clip(w1, w2, outSP=False)
        np.testing.assert_array_equal(wcs.sparse.toarray(), wc.full()[0])


suite = unittest.TestLoader().loadTestsFromTestCase(TestWsets)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
