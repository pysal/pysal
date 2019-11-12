'''GeoDa Text File Reader Unit Tests'''
import unittest
from ..geoda_txt import GeoDaTxtReader as GTR
from .... import examples as pysal_examples


class test_GeoDaTxtReader(unittest.TestCase):
    def setUp(self):
        test_file = pysal_examples.get_path('stl_hom.txt')
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
        self.assertRaises(ValueError, f.read)

if __name__ == '__main__':
    unittest.main()
