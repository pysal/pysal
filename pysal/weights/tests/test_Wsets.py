"""Unit test for Wsets module."""
import unittest
import pysal


class TestWsets(unittest.TestCase):
    """Unit test for Wsets module."""

    def test_w_union(self):
        """Unit test"""
        w1 = pysal.lat2W(4, 4)
        w2 = pysal.lat2W(6, 4)
        w3 = pysal.weights.Wsets.w_union(w1, w2)
        self.assertEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w3.neighbors[15]), set([19, 11, 14]))

    def test_w_intersection(self):
        """Unit test"""
        w1 = pysal.lat2W(4, 4)
        w2 = pysal.lat2W(6, 4)
        w3 = pysal.weights.Wsets.w_union(w1, w2)
        self.assertEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w3.neighbors[15]), set([19, 11, 14]))

    def test_w_difference(self):
        """Unit test"""
        w1 = pysal.lat2W(4, 4, rook=False)
        w2 = pysal.lat2W(4, 4, rook=True)
        w3 = pysal.weights.Wsets.w_difference(w1, w2, constrained=False)
        self.assertNotEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([10, 11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14]))
        self.assertEqual(set(w3.neighbors[15]), set([10]))

    def test_w_symmetric_difference(self):
        """Unit test"""
        w1 = pysal.lat2W(4, 4, rook=False)
        w2 = pysal.lat2W(6, 4, rook=True)
        w3 = pysal.weights.Wsets.w_symmetric_difference(
            w1, w2, constrained=False)
        self.assertNotEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([10, 11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w3.neighbors[15]), set([10, 19]))

    def test_w_subset(self):
        """Unit test"""
        w1 = pysal.lat2W(6, 4)
        ids = range(16)
        w2 = pysal.weights.Wsets.w_subset(w1, ids)
        self.assertEqual(w1[0], w2[0])
        self.assertEqual(set(w1.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14]))


suite = unittest.TestLoader().loadTestsFromTestCase(TestWsets)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
