'''GeoDa Text File Reader Unit Tests'''
import unittest
import pysal
from pysal.core.IOHandlers.geoda_txt import GeoDaTxtReader as GTR


class test_GeoDaTxtReader(unittest.TestCase):
    def setUp(self):
        test_file = pysal.examples.get_path('stl_hom.txt')
        self.obj = GTR(test_file, 'r')

    def test___init__(self):
        self.assertEqual(
            self.obj.header, ['FIPSNO', 'HR8488', 'HR8893', 'HC8488'])

    def test___len__(self):
        expected = 78
        self.assertEqual(expected, len(self.obj))

    def test_close(self):
        f = self.obj
        f.close()
        self.failUnlessRaises(ValueError, f.read)

if __name__ == '__main__':
    unittest.main()
