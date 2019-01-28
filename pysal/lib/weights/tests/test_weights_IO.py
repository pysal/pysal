import unittest
import pysal.lib
import tempfile, os





class TestWIO(unittest.TestCase):
    def setUp(self):
        self.swmFile1 = pysal.lib.examples.get_path('ohio.swm')
        self.swmFile2 = pysal.lib.examples.get_path('us48_CONTIGUITY_EDGES_ONLY.swm')
        self.swmFile3 = pysal.lib.examples.get_path('us48_INVERSE_DISTANCE.swm')
        self.files = [self.swmFile1, self.swmFile2, self.swmFile3]


    def test_SWMIO(self):
        for file in self.files:
            f1 = pysal.lib.io.open(file)
            w1 = f1.read()

            f = tempfile.NamedTemporaryFile(suffix='.swm')
            fname = f.name
            f.close()

            f2 = pysal.lib.io.open(fname, 'w')
            f2.varName = f1.varName
            f2.srs = f1.srs

            f2.write(w1)
            f2.close()
            w2 = pysal.lib.io.open(fname, 'r').read()
            assert w1.pct_nonzero == w2.pct_nonzero
            os.remove(fname)


if __name__ == '__main__':
    unittest.main()
