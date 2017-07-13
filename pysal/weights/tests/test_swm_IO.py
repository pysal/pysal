import unittest
import pysal
import tempfile, os

class TestWIO(unittest.TestCase):
    def setUp(self):
        self.swmFile1 = pysal.examples.get_path('ohio.swm')
        self.swmFile2 = pysal.examples.get_path('us48_CONTIGUITY_EDGES_ONLY.swm')
        self.swmFile3 = pysal.examples.get_path('us48_INVERSE_DISTANCE.swm')

    def SWMIO(self, file_path):
        f1 = pysal.open(file_path)
        w1 = f1.read()
        # print f1.varName
        # print f1.srs

        f = tempfile.NamedTemporaryFile(suffix='.swm')
        fname = f.name
        f.close()

        f2 = pysal.open(fname, 'w')
        f2.varName = f1.varName
        f2.srs = f1.srs

        f2.write(w1)
        f2.close()
        w2 = pysal.open(fname, 'r').read()
        assert w1.pct_nonzero == w2.pct_nonzero
        os.remove(fname)
        # print w2.neighbors
        # print w2.weights

    def test_RWFunctions(self):
        self.SWMIO(self.swmFile1)
        self.SWMIO(self.swmFile2)
        self.SWMIO(self.swmFile3)

suite = unittest.TestLoader().loadTestsFromTestCase(TestWIO)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
