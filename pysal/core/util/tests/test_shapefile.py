import unittest
from cStringIO import StringIO
from pysal.core.util.shapefile import noneMax, noneMin, shp_file, shx_file, NullShape, Point, PolyLine
import os

class TestNoneMax(unittest.TestCase):
    def test_none_max(self):
        self.assertEqual(5, noneMax(5, None))
        self.assertEqual(1, noneMax(None, 1))
        self.assertEqual(None, noneMax(None, None))

class TestNoneMin(unittest.TestCase):
    def test_none_min(self):
        self.assertEqual(5, noneMin(5, None))
        self.assertEqual(1, noneMin(None, 1))
        self.assertEqual(None, noneMin(None, None))

class TestShpFile(unittest.TestCase):
    def test___init__(self):
        shp = shp_file('../../../examples/10740.shp')
        assert shp.header == {'BBOX Xmax': -105.29012, 'BBOX Ymax': 36.219799000000002, 'BBOX Mmax': 0.0, 'BBOX Zmin': 0.0, 'BBOX Mmin': 0.0, 'File Code': 9994, 'BBOX Ymin': 34.259672000000002, 'BBOX Xmin': -107.62651, 'Unused': (0, 0, 0, 0, 0), 'Version': 1000, 'BBOX Zmax': 0.0, 'Shape Type': 5, 'File Length': 260534}

    def test___iter__(self):
        shp = shp_file('../../../examples/shp_test/Point.shp')
        points = [pt for pt in shp]
        expected = [{'Y': -0.25904661905760773, 'X': -0.00068176617532103578, 'Shape Type': 1},
                    {'Y': -0.25630328607387354, 'X': 0.11697145363360706, 'Shape Type': 1},
                    {'Y': -0.33930131004366804, 'X': 0.05043668122270728, 'Shape Type': 1},
                    {'Y': -0.41266375545851519, 'X': -0.041266375545851552, 'Shape Type': 1},
                    {'Y': -0.44017467248908293, 'X': -0.011462882096069604, 'Shape Type': 1},
                    {'Y': -0.46080786026200882, 'X': 0.027510917030567628, 'Shape Type': 1}, 
                    {'Y': -0.45851528384279472, 'X': 0.075655021834060809, 'Shape Type': 1},
                    {'Y': -0.43558951965065495, 'X': 0.11233624454148461, 'Shape Type': 1},
                    {'Y': -0.40578602620087334, 'X': 0.13984716157205224, 'Shape Type': 1}]
        assert points == expected

    def test___len__(self):
        shp = shp_file('../../../examples/10740.shp')
        assert len(shp) == 195

    def test_add_shape(self):
        shp = shp_file('test_point','w','POINT')
        points = [ {'Shape Type': 1, 'X': 0, 'Y': 0},
                   {'Shape Type': 1, 'X': 1, 'Y': 1},
                   {'Shape Type': 1, 'X': 2, 'Y': 2},
                   {'Shape Type': 1, 'X': 3, 'Y': 3},
                   {'Shape Type': 1, 'X': 4, 'Y': 4} ]
        for pt in points:
            shp.add_shape(pt)
        shp.close()

        for a,b in zip(points, shp_file('test_point')):
            self.assertEquals(a,b)
        os.remove('test_point.shp')
        os.remove('test_point.shx')

    def test_close(self):
        shp = shp_file('../../../examples/10740.shp')
        shp.close()
        self.assertEqual(shp.fileObj.closed,True)

    def test_get_shape(self):
        shp = shp_file('../../../examples/shp_test/Line.shp')
        rec = shp.get_shape(0)
        expected = {'BBOX Ymax': -0.25832280562918325,
                    'NumPoints': 3, 
                    'BBOX Ymin': -0.25895877033237352, 
                    'NumParts': 1, 
                    'Vertices': [(-0.0090539248870159517, -0.25832280562918325), 
                                 (0.0074811573959305822, -0.25895877033237352),
                                 (0.0074811573959305822, -0.25895877033237352)],
                    'BBOX Xmax': 0.0074811573959305822,
                    'BBOX Xmin': -0.0090539248870159517,
                    'Shape Type': 3,
                    'Parts Index': [0]}
        self.assertEqual(expected, shp.get_shape(0))

    def test_next(self):
        shp = shp_file('../../../examples/shp_test/Point.shp')
        points = [pt for pt in shp]
        expected = {'Y': -0.25904661905760773, 'X': -0.00068176617532103578, 'Shape Type': 1}
        self.assertEqual(expected, shp.next())
        expected = {'Y': -0.25630328607387354, 'X': 0.11697145363360706, 'Shape Type': 1}
        self.assertEqual(expected, shp.next())

    def test_type(self):
        shp = shp_file('../../../examples/shp_test/Point.shp')
        self.assertEqual("POINT", shp.type())
        shp = shp_file('../../../examples/shp_test/Polygon.shp')
        self.assertEqual("POLYGON", shp.type())
        shp = shp_file('../../../examples/shp_test/Line.shp')
        self.assertEqual("ARC", shp.type())

class TestShxFile(unittest.TestCase):
    def test___init__(self):
        shx = shx_file('../../../examples/shp_test/Point')
        assert isinstance(shx,shx_file)

    def test_add_record(self):
        shx = shx_file('../../../examples/shp_test/Point')
        expectedIndex = [(100, 20), (128, 20), (156, 20),
                         (184, 20), (212, 20), (240, 20),
                         (268, 20), (296, 20), (324, 20)]
        assert shx.index == expectedIndex
        shx2 = shx_file('test','w')
        for i,rec in enumerate(shx.index):
            id,location = shx2.add_record(rec[1])
            assert id == (i+1)
            assert location == rec[0]
        assert shx2.index == shx.index
        shx2.close(shx._header)
        new_shx = open('test.shx','rb').read()
        expected_shx = open('../../../examples/shp_test/Point.shx','rb').read()
        assert new_shx == expected_shx

    def test_close(self):
        shx = shx_file('../../../examples/shp_test/Point')
        shx.close(None)
        self.assertEqual(shx.fileObj.closed,True)

class TestNullShape(unittest.TestCase):
    def test_pack(self):
        null_shape = NullShape()
        self.assertEqual('\x00'*4, null_shape.pack())

    def test_unpack(self):
        null_shape = NullShape()
        self.assertEqual(None, null_shape.unpack())

class TestPoint(unittest.TestCase):
    def test_pack(self):
        record = {"X":5,"Y":5,"Shape Type":1}
        expected = "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14\x40\x00\x00\x00\x00\x00\x00\x14\x40"
        self.assertEqual(expected, Point.pack(record))

    def test_unpack(self):
        dat = StringIO("\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14\x40\x00\x00\x00\x00\x00\x00\x14\x40")
        expected = {"X":5,"Y":5,"Shape Type":1}
        self.assertEqual(expected, Point.unpack(dat))

class TestPolyLine(unittest.TestCase):
    def test_pack(self):
        record = {'BBOX Ymax': -0.25832280562918325, 'NumPoints': 3, 'BBOX Ymin': -0.25895877033237352, 'NumParts': 1, 'Vertices': [(-0.0090539248870159517, -0.25832280562918325), (0.0074811573959305822, -0.25895877033237352), (0.0074811573959305822, -0.25895877033237352)], 'BBOX Xmax': 0.0074811573959305822, 'BBOX Xmin': -0.0090539248870159517, 'Shape Type': 3, 'Parts Index': [0]}
        expected = """\x03\x00\x00\x00\xc0\x46\x52\x3a\xdd\x8a\x82\
\xbf\x3d\xc1\x65\xce\xc7\x92\xd0\xbf\x00\xc5\
\xa0\xe5\x8f\xa4\x7e\x3f\x6b\x40\x7f\x60\x5c\
\x88\xd0\xbf\x01\x00\x00\x00\x03\x00\x00\x00\
\x00\x00\x00\x00\xc0\x46\x52\x3a\xdd\x8a\x82\
\xbf\x6b\x40\x7f\x60\x5c\x88\xd0\xbf\x00\xc5\
\xa0\xe5\x8f\xa4\x7e\x3f\x3d\xc1\x65\xce\xc7\
\x92\xd0\xbf\x00\xc5\xa0\xe5\x8f\xa4\x7e\x3f\
\x3d\xc1\x65\xce\xc7\x92\xd0\xbf"""
        self.assertEqual(expected, PolyLine.pack(record))

    def test_unpack(self):
        dat = StringIO("""\x03\x00\x00\x00\xc0\x46\x52\x3a\xdd\x8a\x82\
\xbf\x3d\xc1\x65\xce\xc7\x92\xd0\xbf\x00\xc5\
\xa0\xe5\x8f\xa4\x7e\x3f\x6b\x40\x7f\x60\x5c\
\x88\xd0\xbf\x01\x00\x00\x00\x03\x00\x00\x00\
\x00\x00\x00\x00\xc0\x46\x52\x3a\xdd\x8a\x82\
\xbf\x6b\x40\x7f\x60\x5c\x88\xd0\xbf\x00\xc5\
\xa0\xe5\x8f\xa4\x7e\x3f\x3d\xc1\x65\xce\xc7\
\x92\xd0\xbf\x00\xc5\xa0\xe5\x8f\xa4\x7e\x3f\
\x3d\xc1\x65\xce\xc7\x92\xd0\xbf""")
        expected = {'BBOX Ymax': -0.25832280562918325, 'NumPoints': 3, 'BBOX Ymin': -0.25895877033237352, 'NumParts': 1, 'Vertices': [(-0.0090539248870159517, -0.25832280562918325), (0.0074811573959305822, -0.25895877033237352), (0.0074811573959305822, -0.25895877033237352)], 'BBOX Xmax': 0.0074811573959305822, 'BBOX Xmin': -0.0090539248870159517, 'Shape Type': 3, 'Parts Index': [0]}
        self.assertEqual(expected, PolyLine.unpack(dat))

class TestMultiPoint(unittest.TestCase):
    def test___init__(self):
        # multi_point = MultiPoint()
        assert False # TODO: implement your test here

class TestPointZ(unittest.TestCase):
    def test___init__(self):
        # point_z = PointZ()
        assert False # TODO: implement your test here

class TestPolyLineZ(unittest.TestCase):
    def test___init__(self):
        # poly_line_z = PolyLineZ()
        assert False # TODO: implement your test here

class TestPolygonZ(unittest.TestCase):
    def test___init__(self):
        # polygon_z = PolygonZ()
        assert False # TODO: implement your test here

class TestMultiPointZ(unittest.TestCase):
    def test___init__(self):
        # multi_point_z = MultiPointZ()
        assert False # TODO: implement your test here

class TestPointM(unittest.TestCase):
    def test___init__(self):
        # point_m = PointM()
        assert False # TODO: implement your test here

class TestPolyLineM(unittest.TestCase):
    def test___init__(self):
        # poly_line_m = PolyLineM()
        assert False # TODO: implement your test here

class TestPolygonM(unittest.TestCase):
    def test___init__(self):
        # polygon_m = PolygonM()
        assert False # TODO: implement your test here

class TestMultiPointM(unittest.TestCase):
    def test___init__(self):
        # multi_point_m = MultiPointM()
        assert False # TODO: implement your test here

class TestMultiPatch(unittest.TestCase):
    def test___init__(self):
        # multi_patch = MultiPatch()
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
