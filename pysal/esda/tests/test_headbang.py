# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import unittest
import pysal
from pysal.esda import smoothing as sm
from pysal.esda import headbang as smh
from pysal import knnW_from_array
import numpy as np
from pysal.common import RTOL, ATOL

from pysal.common import pandas
import warnings

PANDAS_EXTINCT = pandas is None


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
        self.sids_hb_rr5 = np.array([0.00075586, 0.,
                                     0.0008285, 0.0018315, 0.00498891])
        self.sids_hb_r2r5 = np.array([0.0008285, 0.00084331,
                                      0.00086896, 0.0018315, 0.00498891])
        self.sids_hb_r3r5 = np.array([0.00091659, 0.,
                                      0.00156838, 0.0018315, 0.00498891])
        if not PANDAS_EXTINCT:
            self.df = sids_db.to_df()
            self.ename = 'SID74'
            self.bname = 'BIR74'
            self.enames = [self.ename, 'SID79']
            self.bnames = [self.bname, 'BIR79']
            self.sids79r = np.array([.000563, .001659, .001879,
                                     .002410, .002720])

    # @unittest.skip("Depreication")
    def test_Headbanging_Triples(self):
        with warnings.catch_warnings(record=True) as w:
            ht = smh.Headbanging_Triples(self.d, self.w)
            self.assertTrue(len(w) >= 5)  # Should have at least 5 warnings
            self.assertEqual(len(ht.triples), len(self.d))

        with warnings.catch_warnings(record=True) as w:
            ht2 = smh.Headbanging_Triples(self.d, self.w, edgecor=True)
            self.assert_(len(w) > 0)  # Should have at least 1 warning
            self.assertTrue(hasattr(ht2, 'extra'))

        with warnings.catch_warnings(record=True) as w:
            ht = smh.Headbanging_Triples(self.d, self.w, edgecor=True,
                                         angle=120)
            self.assertTrue(len(w) == 0)  # Should have no warnings
        self.assertEqual(len(ht2.triples), len(self.d))
        # htr = sm.Headbanging_Median_Rate(self.e, self.b, ht2, iteration=5)
        # self.assertEqual(len(htr.r), len(self.e))
        # for i in htr.r:
        #     self.assertTrue(i is not None)

    def test_headbang_valid_triple(self):
        p0 = (0, 0)               # Center
        p1 = (0, -1)              # x_1 vertex
        p2_45_yes = (1, 1.01)     # Should be just beyond 135 degrees
        p2_45_no = (1.01, 1)      # Should be just before 135 degrees
        p2_n45_yes = (-1, 1.01)   # Should be just beyond 135 degrees
        p2_n45_no = (-1.01, 1)    # Should be just before 135 degrees

        result = smh.Headbanging_Triples.is_valid_triple(p0, p1, p2_45_yes,
                                                         135)
        np.testing.assert_(result)

        result = smh.Headbanging_Triples.is_valid_triple(p0, p1, p2_n45_yes,
                                                         135)
        np.testing.assert_(result)

        result = smh.Headbanging_Triples.is_valid_triple(p0, p1, p2_45_no, 135)
        np.testing.assert_(~result)

        result = smh.Headbanging_Triples.is_valid_triple(p0, p1, p2_n45_no,
                                                         135)
        np.testing.assert_(~result)

    def test_headbang_make_triple(self):
        p0 = (0, 0)
        neighbors = {2: [-1, 0],
                     5: [0, -1],
                     42: [1, 0],
                     99: [0, 1]}
        result = smh.Headbanging_Triples.construct_triples(p0, neighbors, 135)
        # expected = [(0, 2), (1, 3)]
        expected = [(((-1, 0), (1, 0)), (2, 42)), (((0, -1), (0, 1)), (5, 99))]
        np.testing.assert_(result == expected)
        p0 = (0, 0)
        neighbors = {2: [-1, 0.5],
                     5: [0, -1],
                     42: [1, 0],
                     99: [0.5, 1]}
        result = smh.Headbanging_Triples.construct_triples(p0, neighbors, 135)
        expected = [(((-1, .5), (1, 0)), (2, 42)),
                    (((0, -1), (.5, 1)), (5, 99))]
        np.testing.assert_(result == expected)

    def test_construct_one_extra(self):
        p0 = (0., 0.)
        p1 = (1., 100.)
        p2 = (1., 105.)
        result = smh.Headbanging_Triples.construct_one_extra(p0, p1, p2)
        expected = (1., -100.)
        np.testing.assert_allclose(result, expected)

        p0 = (0., 0.)
        p1 = (-1., 100.)
        p2 = (-1., 105.)
        result = smh.Headbanging_Triples.construct_one_extra(p0, p1, p2)
        expected = (-1., -100.)
        np.testing.assert_allclose(result, expected)

    def test_construct_extras(self):
        p0 = (0., 0.)
        neighbors = {2: (1, 100),
                     5: (1, 105)}
        result = smh.Headbanging_Triples.construct_extra_triples(p0, neighbors,
                                                                 135)
        expected = [(2, 5), 5., np.sqrt(100 ** 2 + 1 ** 2)]
        np.testing.assert_equal(result[0], expected[0])
        np.testing.assert_allclose(result[1], expected[1])
        np.testing.assert_allclose(result[2], expected[2])

        p0 = (0., 0.)
        neighbors = {2: (1, 105),
                     5: (1, 100)}
        result = smh.Headbanging_Triples.construct_extra_triples(p0, neighbors,
                                                                 135)
        expected = [(5, 2), 5., np.sqrt(100 ** 2 + 1 ** 2)]
        np.testing.assert_equal(result[0], expected[0])
        np.testing.assert_allclose(result[1], expected[1])
        np.testing.assert_allclose(result[2], expected[2])
