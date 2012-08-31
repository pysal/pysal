import unittest
import pysal
from pysal.core.IOHandlers.wk1 import Wk1IO
import tempfile
import os


class test_Wk1IO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal.examples.get_path('spat-sym-us.wk1')
        self.obj = Wk1IO(test_file, 'r')

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
            suffix='.wk1', dir=pysal.examples.get_path(''))
        fname = f.name
        f.close()
        o = pysal.open(fname, 'w')
        o.write(w)
        o.close()
        wnew = pysal.open(fname, 'r').read()
        self.assertEqual(wnew.pct_nonzero, w.pct_nonzero)
        os.remove(fname)

if __name__ == '__main__':
    unittest.main()
