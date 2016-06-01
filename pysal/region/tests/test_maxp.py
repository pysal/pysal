
import unittest
import pysal
import numpy as np


class Test_Maxp(unittest.TestCase):
    def setUp(self):
        np.random.seed(100)

    def test_Maxp(self):
        w = pysal.lat2W(10, 10)
        z = np.random.random_sample((w.n, 2))
        p = np.ones((w.n, 1), float)
        floor = 3
        np.random.seed(111)
        solution = pysal.region.Maxp(
            w, z, floor, floor_variable=p, initial=100)
        self.assertEquals(solution.p, 28)
        self.assertEquals(solution.regions[0], [51, 61, 71])

    def test_inference(self):
        w = pysal.weights.lat2W(5, 5)
        z = np.random.random_sample((w.n, 2))
        p = np.ones((w.n, 1), float)
        floor = 3
        np.random.seed(111)
        solution = pysal.region.Maxp(
            w, z, floor, floor_variable=p, initial=100)
        solution.inference(nperm=9)
        self.assertAlmostEquals(solution.pvalue, 0.30000000000000001, 10)

    def test_cinference(self):
        w = pysal.weights.lat2W(5, 5)
        z = np.random.random_sample((w.n, 2))
        p = np.ones((w.n, 1), float)
        floor = 3
        np.random.seed(111)
        solution = pysal.region.Maxp(
            w, z, floor, floor_variable=p, initial=100)
        solution.cinference(nperm=9, maxiter=100)
        self.assertAlmostEquals(solution.cpvalue, 0.10000000000000001, 10)

    def test_Maxp_LISA(self):
        w = pysal.lat2W(10, 10)
        z = np.random.random_sample((w.n, 2))
        p = np.ones(w.n)
        np.random.seed(111)
        mpl = pysal.region.Maxp_LISA(w, z, p, floor=3, floor_variable=p)
        self.assertEquals(mpl.p, 30)
        self.assertEquals(mpl.regions[0], [99, 89, 79, 88, 78])


suite = unittest.TestLoader().loadTestsFromTestCase(Test_Maxp)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
