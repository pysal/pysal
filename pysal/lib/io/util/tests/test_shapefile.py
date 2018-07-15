import unittest

import sys
#import pysal_examples
from .... import examples as  pysal_examples
PY3 = int(sys.version[0]) > 2
if PY3:
    import io
else:
    from cStringIO import StringIO

from ..shapefile import noneMax, noneMin, shp_file, shx_file, NullShape, Point, PolyLine, MultiPoint, PointZ, PolyLineZ, PolygonZ, MultiPointZ, PointM, PolyLineM, PolygonM, MultiPointM, MultiPatch
import os

def bufferIO(buf):
    """
    Temp stringIO function to force compat
    """
    if PY3:
        return io.BytesIO(buf)
    else:
        return StringIO(buf)

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


class test_shp_file(unittest.TestCase):
    def test___init__(self):
        shp = shp_file(pysal_examples.get_path('10740.shp'))
        assert shp.header == {'BBOX Xmax': -105.29012, 'BBOX Ymax': 36.219799000000002, 'BBOX Mmax': 0.0, 'BBOX Zmin': 0.0, 'BBOX Mmin': 0.0, 'File Code': 9994, 'BBOX Ymin': 34.259672000000002, 'BBOX Xmin': -107.62651, 'Unused0': 0, 'Unused1': 0, 'Unused2': 0, 'Unused3': 0, 'Unused4': 0, 'Version': 1000, 'BBOX Zmax': 0.0, 'Shape Type': 5, 'File Length': 260534}

    def test___iter__(self):
        shp = shp_file(pysal_examples.get_path('Point.shp'))
        points = [pt for pt in shp]
        expected = [{'Y': -0.25904661905760773, 'X': -0.00068176617532103578, 'Shape Type': 1},
                    {'Y': -0.25630328607387354, 'X': 0.11697145363360706,
                        'Shape Type': 1},
                    {'Y': -0.33930131004366804, 'X': 0.05043668122270728,
                        'Shape Type': 1},
                    {'Y': -0.41266375545851519, 'X': -0.041266375545851552,
                        'Shape Type': 1},
                    {'Y': -0.44017467248908293, 'X': -0.011462882096069604,
                        'Shape Type': 1},
                    {'Y': -0.46080786026200882, 'X': 0.027510917030567628,
                        'Shape Type': 1},
                    {'Y': -0.45851528384279472, 'X': 0.075655021834060809,
                        'Shape Type': 1},
                    {'Y': -0.43558951965065495, 'X': 0.11233624454148461,
                        'Shape Type': 1},
                    {'Y': -0.40578602620087334, 'X': 0.13984716157205224, 'Shape Type': 1}]
        assert points == expected

    def test___len__(self):
        shp = shp_file(pysal_examples.get_path('10740.shp'))
        assert len(shp) == 195

    def test_add_shape(self):
        shp = shp_file('test_point', 'w', 'POINT')
        points = [{'Shape Type': 1, 'X': 0, 'Y': 0},
                  {'Shape Type': 1, 'X': 1, 'Y': 1},
                  {'Shape Type': 1, 'X': 2, 'Y': 2},
                  {'Shape Type': 1, 'X': 3, 'Y': 3},
                  {'Shape Type': 1, 'X': 4, 'Y': 4}]
        for pt in points:
            shp.add_shape(pt)
        shp.close()

        for a, b in zip(points, shp_file('test_point')):
            self.assertEqual(a, b)
        os.remove('test_point.shp')
        os.remove('test_point.shx')

    def test_close(self):
        shp = shp_file(pysal_examples.get_path('10740.shp'))
        shp.close()
        self.assertEqual(shp.fileObj.closed, True)

    def test_get_shape(self):
        shp = shp_file(pysal_examples.get_path('Line.shp'))
        rec = shp.get_shape(0)
        expected = {'BBOX Ymax': -0.25832280562918325,
                    'NumPoints': 3,
                    'BBOX Ymin': -0.25895877033237352,
                    'NumParts': 1,
                    'Vertices': [(-0.0090539248870159517, -0.25832280562918325),
                                 (0.0074811573959305822, -0.25895877033237352),
                                 (
                                     0.0074811573959305822, -0.25895877033237352)],
                    'BBOX Xmax': 0.0074811573959305822,
                    'BBOX Xmin': -0.0090539248870159517,
                    'Shape Type': 3,
                    'Parts Index': [0]}
        self.assertEqual(expected, shp.get_shape(0))

    def test_next(self):
        shp = shp_file(pysal_examples.get_path('Point.shp'))
        points = [pt for pt in shp]
        expected = {'Y': -0.25904661905760773, 'X': -
                    0.00068176617532103578, 'Shape Type': 1}
        self.assertEqual(expected, next(shp))
        expected = {'Y': -0.25630328607387354, 'X':
                    0.11697145363360706, 'Shape Type': 1}
        self.assertEqual(expected, next(shp))

    def test_type(self):
        shp = shp_file(pysal_examples.get_path('Point.shp'))
        self.assertEqual("POINT", shp.type())
        shp = shp_file(pysal_examples.get_path('Polygon.shp'))
        self.assertEqual("POLYGON", shp.type())
        shp = shp_file(pysal_examples.get_path('Line.shp'))
        self.assertEqual("ARC", shp.type())


class test_shx_file(unittest.TestCase):
    def test___init__(self):
        shx = shx_file(pysal_examples.get_path('Point.shx'))
        assert isinstance(shx, shx_file)

    def test_add_record(self):
        shx = shx_file(pysal_examples.get_path('Point.shx'))
        expectedIndex = [(100, 20), (128, 20), (156, 20),
                         (184, 20), (212, 20), (240, 20),
                         (268, 20), (296, 20), (324, 20)]
        assert shx.index == expectedIndex
        shx2 = shx_file('test', 'w')
        for i, rec in enumerate(shx.index):
            id, location = shx2.add_record(rec[1])
            assert id == (i + 1)
            assert location == rec[0]
        assert shx2.index == shx.index
        shx2.close(shx._header)
        new_shx = open('test.shx', 'rb').read()
        expected_shx = open(pysal_examples.get_path('Point.shx'), 'rb').read()
        assert new_shx == expected_shx
        os.remove('test.shx')

    def test_close(self):
        shx = shx_file(pysal_examples.get_path('Point.shx'))
        shx.close(None)
        self.assertEqual(shx.fileObj.closed, True)


class TestNullShape(unittest.TestCase):
    def test_pack(self):
        null_shape = NullShape()
        self.assertEqual(b'\x00' * 4, null_shape.pack())

    def test_unpack(self):
        null_shape = NullShape()
        self.assertEqual(None, null_shape.unpack())


class TestPoint(unittest.TestCase):
    def test_pack(self):
        record = {"X": 5, "Y": 5, "Shape Type": 1}
        expected = b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14\x40\x00\x00\x00\x00\x00\x00\x14\x40"
        self.assertEqual(expected, Point.pack(record))

    def test_unpack(self):
        dat = bufferIO(b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14\x40\x00\x00\x00\x00\x00\x00\x14\x40")
        expected = {"X": 5, "Y": 5, "Shape Type": 1}
        self.assertEqual(expected, Point.unpack(dat))


class TestPolyLine(unittest.TestCase):
    def test_pack(self):
        record = {'BBOX Ymax': -0.25832280562918325, 'NumPoints': 3, 'BBOX Ymin': -0.25895877033237352, 'NumParts': 1, 'Vertices': [(-0.0090539248870159517, -0.25832280562918325), (0.0074811573959305822, -0.25895877033237352), (0.0074811573959305822, -0.25895877033237352)], 'BBOX Xmax': 0.0074811573959305822, 'BBOX Xmin': -0.0090539248870159517, 'Shape Type': 3, 'Parts Index': [0]}
        expected = b"""\x03\x00\x00\x00\xc0\x46\x52\x3a\xdd\x8a\x82\
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
        dat = bufferIO(b"""\x03\x00\x00\x00\xc0\x46\x52\x3a\xdd\x8a\x82\
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
        self.assertRaises(NotImplementedError, MultiPoint)


class TestPointZ(unittest.TestCase):
    def test_pack(self):
        record = {"X": 5, "Y": 5, "Z": 5, "M": 5, "Shape Type": 11}
        expected = b"\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x14@"
        self.assertEqual(expected, PointZ.pack(record))

    def test_unpack(self):
        dat = bufferIO(b"\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x14@")
        expected = {"X": 5, "Y": 5, "Z": 5, "M": 5, "Shape Type": 11}
        self.assertEqual(expected, PointZ.unpack(dat))


class TestPolyLineZ(unittest.TestCase):
    def test___init__(self):
        self.assertRaises(NotImplementedError, PolyLineZ)


class TestPolyLineZ(unittest.TestCase):
    def test_pack(self):
        record = {'BBOX Ymax': -0.25832280562918325, 'NumPoints': 3, 'BBOX Ymin': -0.25895877033237352, 'NumParts': 1, 'Vertices': [(-0.0090539248870159517, -0.25832280562918325), (0.0074811573959305822, -0.25895877033237352), (0.0074811573959305822, -0.25895877033237352)], 'BBOX Xmax': 0.0074811573959305822, 'BBOX Xmin': -0.0090539248870159517, 'Shape Type': 13, 'Parts Index': [0], 'Zmin': 0, 'Zmax': 10, 'Zarray': [0, 5, 10], 'Mmin': 2, 'Mmax': 4, 'Marray': [2, 3, 4]}
        expected = b"""\r\x00\x00\x00\xc0FR:\xdd\x8a\x82\xbf=\xc1e\xce\xc7\x92\xd0\xbf\x00\xc5\xa0\xe5\x8f\xa4~?k@\x7f`\\\x88\xd0\xbf\x01\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\xc0FR:\xdd\x8a\x82\xbfk@\x7f`\\\x88\xd0\xbf\x00\xc5\xa0\xe5\x8f\xa4~?=\xc1e\xce\xc7\x92\xd0\xbf\x00\xc5\xa0\xe5\x8f\xa4~?=\xc1e\xce\xc7\x92\xd0\xbf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@"""
        self.assertEqual(expected, PolyLineZ.pack(record))

    def test_unpack(self):
        dat = bufferIO(b"""\r\x00\x00\x00\xc0FR:\xdd\x8a\x82\xbf=\xc1e\xce\xc7\x92\xd0\xbf\x00\xc5\xa0\xe5\x8f\xa4~?k@\x7f`\\\x88\xd0\xbf\x01\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\xc0FR:\xdd\x8a\x82\xbfk@\x7f`\\\x88\xd0\xbf\x00\xc5\xa0\xe5\x8f\xa4~?=\xc1e\xce\xc7\x92\xd0\xbf\x00\xc5\xa0\xe5\x8f\xa4~?=\xc1e\xce\xc7\x92\xd0\xbf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@""")
        expected = {'BBOX Ymax': -0.25832280562918325, 'NumPoints': 3, 'BBOX Ymin': -0.25895877033237352, 'NumParts': 1, 'Vertices': [(-0.0090539248870159517, -0.25832280562918325), (0.0074811573959305822, -0.25895877033237352), (0.0074811573959305822, -0.25895877033237352)], 'BBOX Xmax': 0.0074811573959305822, 'BBOX Xmin': -0.0090539248870159517, 'Shape Type': 13, 'Parts Index': [0], 'Zmin': 0, 'Zmax': 10, 'Zarray': [0, 5, 10], 'Mmin': 2, 'Mmax': 4, 'Marray': [2, 3, 4]}
        self.assertEqual(expected, PolyLineZ.unpack(dat))


class TestPolygonZ(unittest.TestCase):
    def test_pack(self):
        record = {
            'BBOX Xmin': 0.0,
            'BBOX Xmax': 10.0,
            'BBOX Ymin': 0.0,
            'BBOX Ymax': 10.0,
            'NumPoints': 4,
            'NumParts': 1,
            'Vertices': [(0.0, 0.0),
                         (10.0, 10.0),
                         (10.0, 0.0),
                         (0.0, 0.0)],
            'Shape Type': 15,
            'Parts Index': [0],
            'Zmin': 0,
            'Zmax': 10,
            'Zarray': [0, 10, 0, 0],
            'Mmin': 2,
            'Mmax': 4,
            'Marray': [2, 4, 2, 2]
        }
        dat = bufferIO(PolygonZ.pack(record))
        self.assertEqual(record, PolygonZ.unpack(dat))


class TestMultiPointZ(unittest.TestCase):
    def test___init__(self):
        self.assertRaises(NotImplementedError, MultiPointZ)
        # multi_point_z = MultiPointZ()


class TestPointM(unittest.TestCase):
    def test___init__(self):
        self.assertRaises(NotImplementedError, PointM)
        # point_m = PointM()


class TestPolyLineM(unittest.TestCase):
    def test___init__(self):
        self.assertRaises(NotImplementedError, PolyLineM)
        # poly_line_m = PolyLineM()


class TestPolygonM(unittest.TestCase):
    def test___init__(self):
        self.assertRaises(NotImplementedError, PolygonM)
        # polygon_m = PolygonM()


class TestMultiPointM(unittest.TestCase):
    def test___init__(self):
        self.assertRaises(NotImplementedError, MultiPointM)
        # multi_point_m = MultiPointM()


class TestMultiPatch(unittest.TestCase):
    def test___init__(self):
        self.assertRaises(NotImplementedError, MultiPatch)
        # multi_patch = MultiPatch()


class _TestPoints(unittest.TestCase):
    def test1(self):
        """ Test creating and reading Point Shape Files """
        shp = shp_file('test_point', 'w', 'POINT')
        points = [{'Shape Type': 1, 'X': 0, 'Y': 0}, {'Shape Type': 1, 'X': 1, 'Y': 1}, {'Shape Type': 1, 'X': 2, 'Y': 2}, {'Shape Type': 1, 'X': 3, 'Y': 3}, {'Shape Type': 1, 'X': 4, 'Y': 4}]
        for pt in points:
            shp.add_shape(pt)
        shp.close()

        shp = list(shp_file('test_point'))
        for a, b in zip(points, shp):
            self.assertEqual(a, b)
        os.remove('test_point.shp')
        os.remove('test_point.shx')


class _TestPolyLines(unittest.TestCase):
    def test1(self):
        """ Test creating and reading PolyLine Shape Files """
        lines = [[(0, 0), (4, 4)], [(1, 0), (5, 4)], [(2, 0), (6, 4)]]
        shapes = []
        for line in lines:
            x = [v[0] for v in line]
            y = [v[1] for v in line]
            rec = {}
            rec['BBOX Xmin'] = min(x)
            rec['BBOX Ymin'] = min(y)
            rec['BBOX Xmax'] = max(x)
            rec['BBOX Ymax'] = max(y)
            rec['NumPoints'] = len(line)
            rec['NumParts'] = 1
            rec['Vertices'] = line
            rec['Shape Type'] = 3
            rec['Parts Index'] = [0]
            shapes.append(rec)
        shp = shp_file('test_line', 'w', 'ARC')
        for line in shapes:
            shp.add_shape(line)
        shp.close()
        shp = list(shp_file('test_line'))
        for a, b in zip(shapes, shp):
            self.assertEqual(a, b)
        os.remove('test_line.shp')
        os.remove('test_line.shx')


class _TestPolygons(unittest.TestCase):
    def test1(self):
        """ Test creating and reading PolyLine Shape Files """
        lines = [[(0, 0), (4, 4), (5, 4), (
            1, 0), (0, 0)], [(1, 0), (5, 4), (6, 4), (2, 0), (1, 0)]]
        shapes = []
        for line in lines:
            x = [v[0] for v in line]
            y = [v[1] for v in line]
            rec = {}
            rec['BBOX Xmin'] = min(x)
            rec['BBOX Ymin'] = min(y)
            rec['BBOX Xmax'] = max(x)
            rec['BBOX Ymax'] = max(y)
            rec['NumPoints'] = len(line)
            rec['NumParts'] = 1
            rec['Vertices'] = line
            rec['Shape Type'] = 5
            rec['Parts Index'] = [0]
            shapes.append(rec)
        shp = shp_file('test_poly', 'w', 'POLYGON')
        for line in shapes:
            shp.add_shape(line)
        shp.close()
        shp = list(shp_file('test_poly'))
        for a, b in zip(shapes, shp):
            self.assertEqual(a, b)
        os.remove('test_poly.shp')
        os.remove('test_poly.shx')


if __name__ == '__main__':
    unittest.main()
