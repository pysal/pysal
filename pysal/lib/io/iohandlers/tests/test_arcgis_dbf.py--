import unittest
from ...fileio import FileIO as psopen
from .... import examples as pysal_examples
from ..arcgis_dbf import ArcGISDbfIO
import tempfile
import os
import warnings


class test_ArcGISDbfIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal_examples.get_path('arcgis_ohio.dbf')
        self.obj = ArcGISDbfIO(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.assertRaises(ValueError, f.read)

    def test_read(self):
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            w = self.obj.read()
            if len(warn) > 0:
                assert issubclass(warn[0].category, RuntimeWarning)
                assert "Missing Value Found, setting value to pysal.MISSINGVALUE" in str(warn[0].message)
        self.assertEqual(88, w.n)
        self.assertEqual(5.25, w.mean_neighbors)
        self.assertEqual([1.0, 1.0, 1.0, 1.0], list(w[1].values()))

    def test_seek(self):
        self.test_read()
        self.assertRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            w = self.obj.read()
            if len(warn) > 0:
                assert issubclass(warn[0].category, RuntimeWarning)
                assert "Missing Value Found, setting value to pysal.MISSINGVALUE" in str(warn[0].message)
        f = tempfile.NamedTemporaryFile(
            suffix='.dbf', dir=pysal_examples.get_path(''))
        fname = f.name
        f.close()
        o = psopen(fname, 'w', 'arcgis_dbf')
        o.write(w)
        o.close()
        f = psopen(fname, 'r', 'arcgis_dbf')
        wnew = f.read()
        f.close()
        self.assertEqual(wnew.pct_nonzero, w.pct_nonzero)
        os.remove(fname)

if __name__ == '__main__':
    unittest.main()
