import unittest
import pysal
from pysal.core.IOHandlers.arcgis_dbf import ArcGISDbfIO
import tempfile
import os

class test_ArcGISDbfIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = '../../../examples/arcgis_ohio.dbf'
        self.obj = ArcGISDbfIO(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.failUnlessRaises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        self.assertEqual(88, w.n)
        self.assertEqual(5.25, w.mean_neighbors)
        self.assertEqual([1.0, 1.0, 1.0, 1.0], w[1].values())

    def test_seek(self):
        self.test_read()
        self.failUnlessRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(suffix='.dbf',dir="../../../examples")
        fname = f.name
        f.close()
        o = pysal.open(fname,'w','arcgis_dbf')
        o.write(w)
        o.close()
        wnew =  pysal.open(fname,'r','arcgis_dbf').read()
        self.assertEqual( wnew.pct_nonzero, w.pct_nonzero)
        os.remove(fname)

if __name__ == '__main__':
    unittest.main()
