import unittest
from ...fileio import FileIO as psopen
from ..dat import DatIO
from .... import examples as pysal_examples
import tempfile
import os


class test_DatIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal_examples.get_path('wmat.dat')
        self.obj = DatIO(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.assertRaises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        self.assertEqual(49, w.n)
        self.assertEqual(4.7346938775510203, w.mean_neighbors)
        self.assertEqual([0.5, 0.5], list(w[5.0].values()))

    def test_seek(self):
        self.test_read()
        self.assertRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(
            suffix='.dat', dir=pysal_examples.get_path(''))
        fname = f.name
        f.close()
        o = psopen(fname, 'w')
        o.write(w)
        o.close()
        wnew = psopen(fname, 'r').read()
        self.assertEqual(wnew.pct_nonzero, w.pct_nonzero)
        os.remove(fname)

if __name__ == '__main__':
    unittest.main()
