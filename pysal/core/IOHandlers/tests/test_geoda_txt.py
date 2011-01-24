'''GeoDa Text File Reader Unit Tests'''
import unittest
import pysal


class test_GeoDaTxtReader(unittest.TestCase):
    def setUp(self):
        test_file = '../../../examples/stl_hom.txt'
        self.file = pysal.open(test_file,'r')
    def test___init__(self):
        # geo_da_txt_reader = GeoDaTxtReader(*args, **kwargs)
        self.failUnless(self.file, 'DataTable: ../../../examples/stl_hom.txt')
        self.assertEqual(self.file.header, ['FIPSNO', 'HR8488', 'HR8893', 'HC8488'])

    def test___len__(self):
        # geo_da_txt_reader = GeoDaTxtReader(*args, **kwargs)
        expected = 78
        self.assertEqual(expected, len(self.file))

    def test_close(self):
        # geo_da_txt_reader = GeoDaTxtReader(*args, **kwargs)
        # self.assertEqual(expected, geo_da_txt_reader.close())
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
