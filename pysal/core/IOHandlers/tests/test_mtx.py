import unittest
import pysal
from pysal.core.IOHandlers.mtx import MtxIO
import tempfile
import os
import warnings
import scipy.sparse as SP


class test_MtxIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal.examples.get_path('wmat.mtx')
        self.obj = MtxIO(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.failUnlessRaises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        self.assertEqual(49, w.n)
        self.assertEqual(4.7346938775510203, w.mean_neighbors)
        self.assertEqual([0.33329999999999999, 0.33329999999999999,
                          0.33329999999999999], w[1].values())
        s0 = w.s0
        self.obj.seek(0)
        wsp = self.obj.read(sparse=True)
        self.assertEqual(49, wsp.n)
        self.assertEqual(s0, wsp.s0)

    def test_seek(self):
        self.test_read()
        self.failUnlessRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        for i in [False, True]:
            self.obj.seek(0)
            w = self.obj.read(sparse=i)
            f = tempfile.NamedTemporaryFile(
                suffix='.mtx', dir=pysal.examples.get_path(''))
            fname = f.name
            f.close()
            o = pysal.open(fname, 'w')
            o.write(w)
            o.close()
            wnew = pysal.open(fname, 'r').read(sparse=i)
            if i:
                self.assertEqual(wnew.s0, w.s0)
            else:
                self.assertEqual(wnew.pct_nonzero, w.pct_nonzero)
            os.remove(fname)

if __name__ == '__main__':
    unittest.main()
