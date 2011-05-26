import unittest
import pysal
from pysal.core.IOHandlers.arcgis_txt import ArcGISTextIO
import tempfile
import os

class test_ArcGISTextIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = '../../../examples/arcgis_txt'
        self.obj = ArcGISTextIO(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.failUnlessRaises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        self.assertEqual(3, w.n)
        self.assertEqual(3, w.mean_neighbors)
        self.assertEqual([0.1, 0.05, 0.0], w['2'].values())

    def test_seek(self):
        self.test_read()
        self.failUnlessRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(suffix='',dir="../../../examples")
        fname = f.name
        f.close()
        o = pysal.open(fname,'w','arcgis_text')
        o.write(w)
        o.close()
        wnew =  pysal.open(fname,'r','arcgis_text').read()
        self.assertEqual( wnew.pct_nonzero, w.pct_nonzero)
        os.remove(fname)

if __name__ == '__main__':
    unittest.main()
