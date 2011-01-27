
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

    def test_Quantiles(self):
        q = mapclassify.Quantiles(self.V, k=5)
        np.testing.assert_array_almost_equal(q.bins, 
            np.array([  1.46400000e+00,   5.79800000e+00,
                1.32780000e+01, 5.46160000e+01,   4.11145000e+03]))
        np.testing.assert_array_almost_equal(q.counts, 
                np.array([12, 11, 12, 11, 12]))

    def test_Std_Mean(self):
        s = mapclassify.Std_Mean(self.V)
        np.testing.assert_array_almost_equal(s.bins, 
            np.array([ -967.36235382,  -420.71712519,   672.57333208,
                1219.21856072, 4111.45      ]))
        np.testing.assert_array_almost_equal(s.counts,
                    np.array([0, 0, 56, 1, 1]))

    def test_Maximum_Breaks(self):
        mb = mapclassify.Maximum_Breaks(self.V, k=5)
        self.assertEquals(mb.k, 5)
        np.testing.assert_array_almost_equal(mb.bins,
                np.array([  146.005,   228.49 ,   546.675,  2417.15 ,
                    4111.45 ]))
        np.testing.assert_array_almost_equal(mb.counts,
                np.array([50,  2,  4,  1,  1]))
 



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


    def test_Jenks_Caspall_Forced(self):
        np.random.seed(100)
        jcf = mapclassify.Jenks_Caspall_Forced(self.V, k=5)
        np.testing.assert_array_almost_equal(jcf.bins,
            np.array([[  1.34000000e+00],
                   [  5.90000000e+00],
                   [  1.67000000e+01],
                   [  5.06500000e+01],
                   [  4.11145000e+03]]))
        np.testing.assert_array_almost_equal(jcf.counts,
            np.array([12, 12, 13,  9, 12]))

    def test_User_Defined(self):
        bins = [20, max(self.V)]
        ud = mapclassify.User_Defined(self.V, bins)
        np.testing.assert_array_almost_equal(ud.bins,
            np.array([   20.  ,  4111.45]))
        np.testing.assert_array_almost_equal(ud.counts,
            np.array([37, 21]))

    def test_Max_P_Classifier(self):
        np.random.seed(100)
        mp = mapclassify.Max_P_Classifier(self.V)
        np.testing.assert_array_almost_equal(mp.bins,
                np.array([8.6999999999999993, 16.699999999999999,
                    20.469999999999999, 66.260000000000005,
                    4111.4499999999998]))
        np.testing.assert_array_almost_equal(mp.counts,
                np.array([29, 8, 1, 10, 10]))

    def test_gadf(self):
        qgadf = mapclassify.gadf(self.V)
        self.assertEquals(qgadf[0], 15)
        self.assertEquals(qgadf[-1], 0.37402575909092828)

    def test_K_classifiers(self):
        np.random.seed(100)
        ks = mapclassify.K_classifiers(self.V)
        self.assertEquals(ks.best.name, 'Fisher_Jenks')
        self.assertEquals(ks.best.gadf, 0.84810327199081048)
        self.assertEquals(ks.best.k, 4)
                

     
        


suite = unittest.TestLoader().loadTestsFromTestCase(_TestMapclassify)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
