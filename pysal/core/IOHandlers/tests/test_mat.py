import unittest
import pysal
from pysal.core.IOHandlers.mat import MatIO
import tempfile
import os
import warnings


class test_MatIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal.examples.get_path('spat-sym-us.mat')
        self.obj = MatIO(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.failUnlessRaises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        self.assertEqual(46, w.n)
        self.assertEqual(4.0869565217391308, w.mean_neighbors)
        self.assertEqual([1.0, 1.0, 1.0, 1.0], w[1].values())

    def test_seek(self):
        self.test_read()
        self.failUnlessRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(
            suffix='.mat', dir=pysal.examples.get_path(''))
        fname = f.name
        f.close()
        o = pysal.open(fname, 'w')
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            o.write(w)
            if len(warn) > 0:
                assert issubclass(warn[0].category, FutureWarning)
        o.close()
        wnew = pysal.open(fname, 'r').read()
        self.assertEqual(wnew.pct_nonzero, w.pct_nonzero)
        os.remove(fname)

if __name__ == '__main__':
    unittest.main()
