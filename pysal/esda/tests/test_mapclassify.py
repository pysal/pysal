
from pysal.esda import mapclassify
import doctest
import unittest
import numpy as np

class _TestMapclassify(unittest.TestCase):
    def setUp(self):
        import pysal
        dat = pysal.open("../../examples/calempdensity.csv")
        self.V = np.array([record[-1] for record in dat])

    def test_Natural_Breaks(self):
         np.random.seed(10)
         nb = mapclassify.Natural_Breaks(self.V,5)
         self.assertEquals(nb.k,5)
         self.assertEquals(len(nb.counts),5)
         np.testing.assert_array_almost_equal(nb.counts,
                 np.array([14, 13, 14, 10, 7]))
        
    def test_quantile(self):
        import pysal
        x = np.arange(1000)
        qx = pysal.quantile(x)
        expected = np.array([ 249.75,  499.5 ,  749.25,  999.  ])
        np.testing.assert_array_almost_equal(expected, qx)

    def test_Box_Plot(self):
        bp=mapclassify.Box_Plot(self.V)
        bins = np.array([ -5.28762500e+01,   2.56750000e+00,   9.36500000e+00,
            3.95300000e+01,   9.49737500e+01,   4.11145000e+03])
        np.testing.assert_array_almost_equal(bp.bins, bins)

    def test_Equal_Interval(self):
        ei = mapclassify.Equal_Interval(self.V)
        np.testing.assert_array_almost_equal(ei.counts,
                np.array([57, 0, 0, 0, 1]))
        np.testing.assert_array_almost_equal(ei.bins,
                np.array([822.394,  1644.658,  2466.922,  3289.186,
                    4111.45]))

    def test_Percentiles(self):
        pc = mapclassify.Percentiles(self.V)
        np.testing.assert_array_almost_equal(pc.bins, np.array([
            1.35700000e-01,   5.53000000e-01, 9.36500000e+00, 2.13914000e+02,
            2.17994800e+03, 4.11145000e+03]))
        np.testing.assert_array_almost_equal(pc.counts, 
                np.array([1, 5, 23, 23, 5, 1]))


    def test_Fisher_Jenks(self):
         np.random.seed(10)
         fj = mapclassify.Fisher_Jenks(self.V)
         self.assertEquals(fj.adcm,832.8900000000001)
         self.assertEquals(fj.bins, [110.73999999999999, 192.05000000000001,
             370.5, 722.85000000000002, 4111.4499999999998])
         np.testing.assert_array_almost_equal(fj.counts, 
                 np.array([50, 2, 4, 1, 1]))
 

    def test_Jenks_Caspall(self):
         np.random.seed(10)
         jc = mapclassify.Jenks_Caspall(self.V, k=5)
         np.testing.assert_array_almost_equal(jc.counts, 
                 np.array([14, 13, 14, 10, 7]))
         np.testing.assert_array_almost_equal(jc.bins, 
                 np.array([[  1.81000000e+00], [  7.60000000e+00],
                     [ 2.98200000e+01], [  1.81270000e+02],
                     [ 4.11145000e+03]]))


    def test_Jenks_Caspall_Sampled(self):
        np.random.seed(100)
        x = np.random.random(100000)
        jc = mapclassify.Jenks_Caspall(x)
        jcs = mapclassify.Jenks_Caspall_Sampled(x)
        np.testing.assert_array_almost_equal(jc.bins,
                np.array([[ 0.19718393],
                       [ 0.39655886],
                       [ 0.59648522],
                       [ 0.79780763],
                       [ 0.99997979]]))
        np.testing.assert_array_almost_equal(jcs.bins,
                np.array([[ 0.20856569],
                       [ 0.41513931],
                       [ 0.62457691],
                       [ 0.82561423],
                       [ 0.99997979]]))


suite = unittest.TestLoader().loadTestsFromTestCase(_TestMapclassify)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
