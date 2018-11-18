import unittest
from ..wkt import WKTReader
from .... import examples as pysal_examples



class test_WKTReader(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal_examples.get_path('stl_hom.wkt')
        self.obj = WKTReader(test_file, 'r')

    def test_close(self):
        f = self.obj
        f.close()
        self.assertRaises(ValueError, f.read)
        # w_kt_reader = WKTReader(*args, **kwargs)
        # self.assertEqual(expected, w_kt_reader.close())

    def test_open(self):
        f = self.obj
        expected = ['wkt']
        self.assertEqual(expected, f.FORMATS)

    def test__read(self):
        polys = self.obj.read()
        self.assertEqual(78, len(polys))
        self.assertEqual((-91.195784694307383, 39.990883050220845),
                         polys[1].centroid)


if __name__ == '__main__':
    unittest.main()
