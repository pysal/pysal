"""Unit test for Contiguity.py"""
import unittest
import pysal
import numpy as np


class TestContiguity(unittest.TestCase):
    def setUp(self):
        self.polyShp = pysal.examples.get_path('10740.shp')

    def test_buildContiguity(self):
        w = pysal.buildContiguity(pysal.open(self.polyShp, 'r'))
        self.assertEqual(w[0], {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0})
        w = pysal.buildContiguity(
            pysal.open(self.polyShp, 'r'), criterion='queen')
        self.assertEqual(w.pct_nonzero, 3.1926364234056544)
        w = pysal.buildContiguity(
            pysal.open(self.polyShp, 'r'), criterion='rook')
        self.assertEqual(w.pct_nonzero, 2.6351084812623275)
        fips = pysal.open(pysal.examples.get_path('10740.dbf')).by_col('STFID')
        w = pysal.buildContiguity(pysal.open(self.polyShp, 'r'), ids=fips)
        self.assertEqual(w['35001000107'], {'35001003805': 1.0, '35001003721':
                                            1.0, '35001000111': 1.0, '35001000112': 1.0, '35001000108': 1.0})


if __name__ == "__main__":
    unittest.main()
