
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
        bins = np.array([ -7.24325000e+01,   2.56750000e+00,   9.36500000e+00,
             3.95300000e+01,   1.14530000e+02,   4.11145000e+03])
        np.testing.assert_array_almost_equal(bp.bins, bins)

    def test_Equal_Interval(self):
        ei = mapclassify.Equal_Interval(self.V)
        np.testing.assert_array_almost_equal(ei.counts,
                np.array([57, 0, 0, 0, 1]))
        np.testing.assert_array_almost_equal(ei.bins,
                np.array([822.394,  1644.658,  2466.922,  3289.186,
                    4111.45]))






suite = unittest.TestLoader().loadTestsFromTestCase(_TestMapclassify)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
