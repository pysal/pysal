"""Unit tests for gal.py"""
import unittest
import tempfile
import os
from ..gal import GalIO
from ...fileio import FileIO as psopen
from .... import examples as pysal_examples


class test_GalIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal_examples.get_path('sids2.gal')
        self.obj = GalIO(test_file, 'r')

    def test___init__(self):
        self.assertEqual(self.obj._typ, str)

    def test_close(self):
        f = self.obj
        f.close()
        self.assertRaises(ValueError, f.read)

    def test_read(self):
        # reading a GAL returns a W
        w = self.obj.read()
        self.assertEqual(w.n, 100)
        self.assertAlmostEqual(w.sd, 1.5151237573214935)
        self.assertEqual(w.s0, 462.0)
        self.assertEqual(w.s1, 924.0)

    def test_seek(self):
        self.test_read()
        self.assertRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(suffix='.gal')
        fname = f.name
        f.close()
        o = psopen(fname, 'w')
        o.write(w)
        o.close()
        wnew = psopen(fname, 'r').read()
        self.assertEqual(wnew.pct_nonzero, w.pct_nonzero)


if __name__ == '__main__':
    unittest.main()
