import unittest
import pysal
from pysal.esda import smoothing as sm
from pysal import knnW
import numpy as np

class Flatten_Tester(unittest.TestCase):
    def setUp(self):
        self.input = [[1,2],[3,3,4],[5,6]]

    def test_flatten(self):
        out1 = sm.flatten(self.input)
        out2 = sm.flatten(self.input, unique=False)
        self.assertEquals(out1, [1,2,3,4,5,6])
        self.assertEquals(out2, [1,2,3,3,4,5,6])

class WMean_Tester(unittest.TestCase):
    def setUp(self):
        self.d = np.array([5,4,3,1,2])
        self.w1 = np.array([10,22,9,2,5])
        self.w2 = np.array([10,14,17,2,5])
    
    def test_weighted_mean(self):
        out1 = sm.weighted_median(self.d, self.w1)
        out2 = sm.weighted_median(self.d, self.w2)
        self.assertEquals(out1, 4)
        self.assertEquals(out2, 3.5)

class SRate_Tester(unittest.TestCase):
    def setUp(self):
        sids = pysal.open('../examples/sids2.dbf', 'r')
        self.w = pysal.open('../examples/sids2.gal', 'r').read()
        self.b, self.e = np.array(sids[:,8]), np.array(sids[:,9])
        er = [0.453433, 0.000000, 0.775871, 0.973810, 3.133190]
        eb = [0.0016973, 0.0017054, 0.0017731, 0.0020129, 0.0035349]
        sr = [0.0009922, 0.0012639, 0.0009740, 0.0007605, 0.0050154]
        smr = [0.00083622, 0.00109402, 0.00081567, 0.0, 0.0048209]
        smr_w = [0.00127146, 0.00127146, 0.0008433, 0.0, 0.0049889]
        smr2 = [0.00091659, 0.00087641, 0.00091073, 0.0, 0.00467633]
        self.er = [round(i,5) for i in er]
        self.eb = [round(i,7) for i in eb]
        self.sr = [round(i,7) for i in sr]
        self.smr = [round(i,7) for i in smr]
        self.smr_w = [round(i,7) for i in smr_w]
        self.smr2 = [round(i,7) for i in smr2]

    def test_rate(self):
        out_er = sm.Excess_Risk(self.e, self.b).r
        out_eb = sm.Empirical_Bayes(self.e, self.b).r
        out_sr = sm.Spatial_Rate(self.e, self.b, self.w).r
        out_smr = sm.Spatial_Median_Rate(self.e, self.b, self.w).r
        out_smr_w = sm.Spatial_Median_Rate(self.e, self.b, self.w, aw=self.b).r
        out_smr2 = sm.Spatial_Median_Rate(self.e, self.b, self.w, iteration=2).r
        out_er = [round(i,5) for i in out_er[:5]]
        out_eb = [round(i,7) for i in out_eb[:5]]
        out_sr = [round(i,7) for i in out_sr[:5]]
        out_smr = [round(i,7) for i in out_smr[:5]]
        out_smr_w = [round(i,7) for i in out_smr_w[:5]]
        out_smr2 = [round(i,7) for i in out_smr2[:5]]
        self.assertEquals(out_er, self.er)
        self.assertEquals(out_eb, self.eb)
        self.assertEquals(out_sr, self.sr)
        self.assertEquals(out_smr, self.smr)
        self.assertEquals(out_smr_w, self.smr_w)
        self.assertEquals(out_smr2, self.smr2)

class HT_Tester(unittest.TestCase):
    def setUp(self):
        sids = pysal.open('../examples/sids2.shp', 'r')
        self.d = np.array([i.centroid for i in sids])
        self.w = knnW(self.d, k=5)
        if not self.w.id_order_set: self.w.id_order = self.w.id_order
        sids_db = pysal.open('../examples/sids2.dbf', 'r')
        self.b, self.e = np.array(sids_db[:,8]), np.array(sids_db[:,9])

    def test_ht(self):
        ht = sm.Headbanging_Triples(self.d, self.w)
        self.assertEquals(len(ht.triples), len(self.d))
        ht2 = sm.Headbanging_Triples(self.d, self.w, edgecor=True)
        self.assertTrue(hasattr(ht2, 'extra'))
        self.assertEquals(len(ht2.triples), len(self.d))
        htr = sm.Headbanging_Median_Rate(self.e,self.b,ht2,iteration=5)
        self.assertEquals(len(htr.r), len(self.e))
        for i in htr.r:
            self.assertTrue(i is not None)


suite = unittest.TestSuite()
test_classes = [Flatten_Tester, WMean_Tester, SRate_Tester, HT_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)

