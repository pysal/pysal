import unittest
import pysal
from pysal.core.IOHandlers.geobugs_txt import GeoBUGSTextIO
import tempfile
import os


class test_GeoBUGSTextIO(unittest.TestCase):
    def setUp(self):
        self.test_file_scot = test_file_scot = pysal.examples.get_path(
            'geobugs_scot')
        self.test_file_col = test_file_col = pysal.examples.get_path(
            'spdep_listw2WB_columbus')
        self.obj_scot = GeoBUGSTextIO(test_file_scot, 'r')
        self.obj_col = GeoBUGSTextIO(test_file_col, 'r')

    def test_close(self):
        for obj in [self.obj_scot, self.obj_col]:
            f = obj
            f.close()
            self.failUnlessRaises(ValueError, f.read)

    def test_read(self):
        w_scot = self.obj_scot.read()
        self.assertEqual(56, w_scot.n)
        self.assertEqual(4.1785714285714288, w_scot.mean_neighbors)
        self.assertEqual([1.0, 1.0, 1.0], w_scot[1].values())

        w_col = self.obj_col.read()
        self.assertEqual(49, w_col.n)
        self.assertEqual(4.6938775510204085, w_col.mean_neighbors)
        self.assertEqual([0.5, 0.5], w_col[1].values())

    def test_seek(self):
        self.test_read()
        self.failUnlessRaises(StopIteration, self.obj_scot.read)
        self.failUnlessRaises(StopIteration, self.obj_col.read)
        self.obj_scot.seek(0)
        self.obj_col.seek(0)
        self.test_read()

    def test_write(self):
        for obj in [self.obj_scot, self.obj_col]:
            w = obj.read()
            f = tempfile.NamedTemporaryFile(
                suffix='', dir=pysal.examples.get_path(''))
            fname = f.name
            f.close()
            o = pysal.open(fname, 'w', 'geobugs_text')
            o.write(w)
            o.close()
            wnew = pysal.open(fname, 'r', 'geobugs_text').read()
            self.assertEqual(wnew.pct_nonzero, w.pct_nonzero)
            os.remove(fname)

if __name__ == '__main__':
    unittest.main()
