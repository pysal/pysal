import unittest
import pysal
from pysal.esda import smoothing as sm
from pysal import knnW_from_array
import numpy as np


class TestFlatten(unittest.TestCase):
    def setUp(self):
        self.input = [[1, 2], [3, 3, 4], [5, 6]]

    def test_flatten(self):
        out1 = sm.flatten(self.input)
        out2 = sm.flatten(self.input, unique=False)
        self.assertEquals(out1, [1, 2, 3, 4, 5, 6])
        self.assertEquals(out2, [1, 2, 3, 3, 4, 5, 6])


class TestWMean(unittest.TestCase):
    def setUp(self):
        self.d = np.array([5, 4, 3, 1, 2])
        self.w1 = np.array([10, 22, 9, 2, 5])
        self.w2 = np.array([10, 14, 17, 2, 5])

    def test_weighted_median(self):
        out1 = sm.weighted_median(self.d, self.w1)
        out2 = sm.weighted_median(self.d, self.w2)
        self.assertEquals(out1, 4)
        self.assertEquals(out2, 3.5)


class TestAgeStd(unittest.TestCase):
    def setUp(self):
        self.e = np.array([30, 25, 25, 15, 33, 21, 30, 20])
        self.b = np.array([1000, 1000, 1100, 900, 1000, 900, 1100, 900])
        self.s_e = np.array([100, 45, 120, 100, 50, 30, 200, 80])
        self.s_b = s = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])
        self.n = 2

    def test_crude_age_standardization(self):
        crude = sm.crude_age_standardization(self.e, self.b, self.n).round(8)
        crude_exp = np.array([0.02375000, 0.02666667])
        self.assertEquals(list(crude), list(crude_exp))

    def test_direct_age_standardization(self):
        direct = np.array(sm.direct_age_standardization(
            self.e, self.b, self.s_b, self.n)).round(8)
        direct_exp = np.array([[0.02374402, 0.01920491,
                                0.02904848], [0.02665072, 0.02177143, 0.03230508]])
        self.assertEquals(list(direct.flatten()), list(direct_exp.flatten()))

    def test_indirect_age_standardization(self):
        indirect = np.array(sm.indirect_age_standardization(
            self.e, self.b, self.s_e, self.s_b, self.n)).round(8)
        indirect_exp = np.array([[0.02372382, 0.01940230,
                                  0.02900789], [0.02610803, .02154304, 0.03164035]])
        self.assertEquals(
            list(indirect.flatten()), list(indirect_exp.flatten()))


class TestSRate(unittest.TestCase):
    def setUp(self):
        sids = pysal.open(pysal.examples.get_path('sids2.dbf'), 'r')
        self.w = pysal.open(pysal.examples.get_path('sids2.gal'), 'r').read()
        self.b, self.e = np.array(sids[:, 8]), np.array(sids[:, 9])
        er = [0.453433, 0.000000, 0.775871, 0.973810, 3.133190]
        eb = [0.0016973, 0.0017054, 0.0017731, 0.0020129, 0.0035349]
        sr = [0.0009922, 0.0012639, 0.0009740, 0.0007605, 0.0050154]
        smr = [0.00083622, 0.00109402, 0.00081567, 0.0, 0.0048209]
        smr_w = [0.00127146, 0.00127146, 0.0008433, 0.0, 0.0049889]
        smr2 = [0.00091659, 0.00087641, 0.00091073, 0.0, 0.00467633]
        self.er = [round(i, 5) for i in er]
        self.eb = [round(i, 7) for i in eb]
        self.sr = [round(i, 7) for i in sr]
        self.smr = [round(i, 7) for i in smr]
        self.smr_w = [round(i, 7) for i in smr_w]
        self.smr2 = [round(i, 7) for i in smr2]

    def test_Excess_Risk(self):
        out_er = sm.Excess_Risk(self.e, self.b).r
        out_er = [round(i, 5) for i in out_er[:5]]
        self.assertEquals(out_er, self.er)

    def test_Empirical_Bayes(self):
        out_eb = sm.Empirical_Bayes(self.e, self.b).r
        out_eb = [round(i, 7) for i in out_eb[:5]]
        self.assertEquals(out_eb, self.eb)

    def test_Spatial_Empirical_Bayes(self):
        stl = pysal.open(pysal.examples.get_path('stl_hom.csv'), 'r')
        stl_e, stl_b = np.array(stl[:, 10]), np.array(stl[:, 13])
        stl_w = pysal.open(pysal.examples.get_path('stl.gal'), 'r').read()
        if not stl_w.id_order_set:
            stl_w.id_order = range(1, len(stl) + 1)
        s_eb = sm.Spatial_Empirical_Bayes(stl_e, stl_b, stl_w)
        s_ebr10 = np.array([4.01485749e-05, 3.62437513e-05,
                            4.93034844e-05, 5.09387329e-05, 3.72735210e-05,
                            3.69333797e-05, 5.40245456e-05, 2.99806055e-05,
                            3.73034109e-05, 3.47270722e-05])
        np.testing.assert_array_almost_equal(s_ebr10, s_eb.r[:10])

    def test_Spatial_Rate(self):
        out_sr = sm.Spatial_Rate(self.e, self.b, self.w).r
        out_sr = [round(i, 7) for i in out_sr[:5]]
        self.assertEquals(out_sr, self.sr)

    def test_Spatial_Median_Rate(self):
        out_smr = sm.Spatial_Median_Rate(self.e, self.b, self.w).r
        out_smr_w = sm.Spatial_Median_Rate(self.e, self.b, self.w, aw=self.b).r
        out_smr2 = sm.Spatial_Median_Rate(
            self.e, self.b, self.w, iteration=2).r
        out_smr = [round(i, 7) for i in out_smr[:5]]
        out_smr_w = [round(i, 7) for i in out_smr_w[:5]]
        out_smr2 = [round(i, 7) for i in out_smr2[:5]]
        self.assertEquals(out_smr, self.smr)
        self.assertEquals(out_smr_w, self.smr_w)
        self.assertEquals(out_smr2, self.smr2)


class TestHB(unittest.TestCase):
    def setUp(self):
        sids = pysal.open(pysal.examples.get_path('sids2.shp'), 'r')
        self.sids = sids
        self.d = np.array([i.centroid for i in sids])
        self.w = knnW_from_array(self.d, k=5)
        if not self.w.id_order_set:
            self.w.id_order = self.w.id_order
        sids_db = pysal.open(pysal.examples.get_path('sids2.dbf'), 'r')
        self.b, self.e = np.array(sids_db[:, 8]), np.array(sids_db[:, 9])

    def test_Headbanging_Triples(self):
        ht = sm.Headbanging_Triples(self.d, self.w)
        self.assertEquals(len(ht.triples), len(self.d))
        ht2 = sm.Headbanging_Triples(self.d, self.w, edgecor=True)
        self.assertTrue(hasattr(ht2, 'extra'))
        self.assertEquals(len(ht2.triples), len(self.d))
        htr = sm.Headbanging_Median_Rate(self.e, self.b, ht2, iteration=5)
        self.assertEquals(len(htr.r), len(self.e))
        for i in htr.r:
            self.assertTrue(i is not None)

    def test_Headbanging_Median_Rate(self):
        sids_d = np.array([i.centroid for i in self.sids])
        sids_w = pysal.knnW_from_array(sids_d, k=5)
        if not sids_w.id_order_set:
            sids_w.id_order = sids_w.id_order
        s_ht = sm.Headbanging_Triples(sids_d, sids_w, k=5)
        sids_db = pysal.open(pysal.examples.get_path('sids2.dbf'), 'r')
        s_e, s_b = np.array(sids_db[:, 9]), np.array(sids_db[:, 8])
        sids_hb_r = sm.Headbanging_Median_Rate(s_e, s_b, s_ht)
        sids_hb_rr5 = np.array([0.00075586, 0.,
                                0.0008285, 0.0018315, 0.00498891])
        np.testing.assert_array_almost_equal(sids_hb_rr5, sids_hb_r.r[:5])
        sids_hb_r2 = sm.Headbanging_Median_Rate(s_e, s_b, s_ht, iteration=5)
        sids_hb_r2r5 = np.array([0.0008285, 0.00084331,
                                 0.00086896, 0.0018315, 0.00498891])
        np.testing.assert_array_almost_equal(sids_hb_r2r5, sids_hb_r2.r[:5])
        sids_hb_r3 = sm.Headbanging_Median_Rate(s_e, s_b, s_ht, aw=s_b)
        sids_hb_r3r5 = np.array([0.00091659, 0.,
                                 0.00156838, 0.0018315, 0.00498891])
        np.testing.assert_array_almost_equal(sids_hb_r3r5, sids_hb_r3.r[:5])


class TestKernel_AgeAdj_SM(unittest.TestCase):
    def setUp(self):
        self.e = np.array([10, 1, 3, 4, 2, 5])
        self.b = np.array([100, 15, 20, 20, 80, 90])
        self.e1 = np.array([10, 8, 1, 4, 3, 5, 4, 3, 2, 1, 5, 3])
        self.b1 = np.array([100, 90, 15, 30, 25, 20, 30, 20, 80, 80, 90, 60])
        self.s = np.array([98, 88, 15, 29, 20, 23, 33, 25, 76, 80, 89, 66])
        self.points = [(
            10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        self.kw = pysal.weights.Kernel(self.points)
        if not self.kw.id_order_set:
            self.kw.id_order = range(0, len(self.points))

    def test_Kernel_Smoother(self):
        kr = sm.Kernel_Smoother(self.e, self.b, self.kw)
        exp = [0.10543301, 0.0858573, 0.08256196, 0.09884584,
               0.04756872, 0.04845298]
        self.assertEquals(list(kr.r.round(8)), exp)

    def test_Age_Adjusted_Smoother(self):
        ar = sm.Age_Adjusted_Smoother(self.e1, self.b1, self.kw, self.s)
        exp = [0.10519625, 0.08494318, 0.06440072, 0.06898604,
               0.06952076, 0.05020968]
        self.assertEquals(list(ar.r.round(8)), exp)

    def test_Disk_Smoother(self):
        self.kw.transform = 'b'
        exp = [0.12222222000000001, 0.10833333, 0.08055556,
               0.08944444, 0.09944444, 0.09351852]
        disk = sm.Disk_Smoother(self.e, self.b, self.kw)
        self.assertEqual(list(disk.r.round(8)), exp)

    def test_Spatial_Filtering(self):
        points = np.array(self.points)
        bbox = [[0, 0], [45, 45]]
        sf = sm.Spatial_Filtering(bbox, points, self.e, self.b, 2, 2, r=30)
        exp = [0.11111111, 0.11111111, 0.20000000000000001, 0.085106379999999995,
               0.076923080000000005, 0.05789474, 0.052173909999999997, 0.066666669999999997, 0.04117647]
        self.assertEqual(list(sf.r.round(8)), exp)


class TestUtils(unittest.TestCase):
    def test_sum_by_n(self):
        d = np.array([10, 9, 20, 30])
        w = np.array([0.5, 0.1, 0.3, 0.8])
        n = 2
        exp_sum = np.array([5.9, 30.])
        np.testing.assert_array_almost_equal(exp_sum, sm.sum_by_n(d, w, n))

    def test_standardized_mortality_ratio(self):
        e = np.array([30, 25, 25, 15, 33, 21, 30, 20])
        b = np.array([100, 100, 110, 90, 100, 90, 110, 90])
        s_e = np.array([100, 45, 120, 100, 50, 30, 200, 80])
        s_b = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])
        n = 2
        exp_smr = np.array([2.48691099, 2.73684211])
        np.testing.assert_array_almost_equal(exp_smr,
                                             sm.standardized_mortality_ratio(e, b, s_e, s_b, n))

    def test_choynowski(self):
        e = np.array([30, 25, 25, 15, 33, 21, 30, 20])
        b = np.array([100, 100, 110, 90, 100, 90, 110, 90])
        n = 2
        exp_choy = np.array([0.30437751, 0.29367033])
        np.testing.assert_array_almost_equal(exp_choy, sm.choynowski(e, b, n))

    def test_assuncao_rate(self):
        e = np.array([30, 25, 25, 15, 33, 21, 30, 20])
        b = np.array([100, 100, 110, 90, 100, 90, 110, 90])
        exp_assuncao = np.array([1.03843594, -0.04099089, -0.56250375,
            -1.73061861])
        np.testing.assert_array_almost_equal(
            exp_assuncao, sm.assuncao_rate(e, b)[:4])


suite = unittest.TestSuite()
test_classes = [TestFlatten, TestWMean, TestAgeStd, TestSRate, TestHB,
                TestKernel_AgeAdj_SM, TestUtils]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
