import unittest
import pysal
from pysal.core.IOHandlers.gwt import GwtIO
import tempfile
import os
import warnings

class test_GwtIO(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = '../../../examples/juvenile.gwt'
        self.obj = GwtIO(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.failUnlessRaises(ValueError, f.read)

    def test_read(self):
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            w = self.obj.read()
            assert issubclass(warn[0].category, RuntimeWarning)
            assert "Weights have been converted to binary. To retrieve original values use w.transform='o'" in str(warn[0].message)
        self.assertEqual(168, w.n)
        self.assertEqual(16.678571428571427, w.mean_neighbors)
        self.assertEqual([1.0], w[1].values())

    def test_seek(self):
        self.test_read()
        self.failUnlessRaises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    # Commented out by CRS, GWT 'w' mode removed until we can find a good solution for retaining distances.
    # see issue #153.
    #def test_write(self):
        #w = self.obj.read()
        #f = tempfile.NamedTemporaryFile(suffix='.gwt',dir="../../../examples")
        #fname = f.name
        #f.close()
        #o = pysal.open(fname,'w')
        #copy the shapefile and ID variable names from the old gwt.
        # this is only available after the read() method has been called.
        #o.shpName = self.obj.shpName
        #o.varName = self.obj.varName
        #o.write(w)
        #o.close()
        #wnew =  pysal.open(fname,'r').read()
        #self.assertEqual( wnew.pct_nonzero, w.pct_nonzero)
        #os.remove(fname)



if __name__ == '__main__':
    unittest.main()
