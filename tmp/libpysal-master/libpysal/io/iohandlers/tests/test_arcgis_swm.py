import unittest
from ..arcgis_swm import ArcGISSwmIO
from ...fileio import FileIO as psopen
from .... import examples as pysal_examples
import tempfile
import os


class test_ArcGISSwmIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal_examples.get_path('ohio.swm')
        self.obj = ArcGISSwmIO(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.assertRaises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        self.assertEqual(88, w.n)
        self.assertEqual(5.25, w.mean_neighbors)
        self.assertEqual([1.0, 1.0, 1.0, 1.0], list(w[1].values()))

    def test_seek(self):
        self.test_read()
        self.assertRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(
            suffix='.swm', dir=pysal_examples.get_path(''))
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
