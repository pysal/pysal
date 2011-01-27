import unittest
import pysal
from pysal.core.IOHandlers.gwt import GwtIO
import tempfile
import os

class test_GwtIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = '../../../examples/juvenile.gwt'
        self.obj = GwtIO(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.failUnlessRaises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        self.assertEqual(168, w.n)
        self.assertEqual(16.678571428571427, w.mean_neighbors)
        self.assertEqual([14.1421356], w[1].values())

    def test_seek(self):
        self.test_read()
        self.failUnlessRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(suffix='.gwt')
        fname = f.name
        f.close()
        o = pysal.open(fname,'w')
        o.write(w)
        o.close()
        wnew =  pysal.open(fname,'r').read()
        self.assertEqual( wnew.pct_nonzero, w.pct_nonzero)
        os.remove(fname)



if __name__ == '__main__':
    unittest.main()
