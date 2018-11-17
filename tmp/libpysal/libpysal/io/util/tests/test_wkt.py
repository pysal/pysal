import unittest
from ..wkt import WKTParser
from ....cg.shapes import Point, Chain, Polygon


class test_WKTParser(unittest.TestCase):
    def setUp(self):
        #Create some Well-Known Text objects
        self.wktPOINT = 'POINT(6 10)'
        self.wktLINESTRING = 'LINESTRING(3 4,10 50,20 25)'
        self.wktPOLYGON = 'POLYGON((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2))'
        self.unsupported = ['MULTIPOINT(3.5 5.6,4.8 10.5)',
                            'MULTILINESTRING((3 4,10 50,20 25),(-5 -8,-10 -8,-15 -4))',
                            'MULTIPOLYGON(((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2)),((3 3,6 2,6 4,3 3)))',
                            'GEOMETRYCOLLECTION(POINT(4 6),LINESTRING(4 6,7 10))',
                            'POINT ZM (1 1 5 60)',
                            'POINT M (1 1 80)']
        self.empty = ['POINT EMPTY', 'MULTIPOLYGON EMPTY']
        self.parser = WKTParser()

    def test_Point(self):
        pt = self.parser(self.wktPOINT)
        self.assertTrue(issubclass(type(pt), Point))
        self.assertEqual(pt[:], (6.0, 10.0))

    def test_LineString(self):
        line = self.parser(self.wktLINESTRING)
        self.assertTrue(issubclass(type(line), Chain))
        parts = [[pt[:] for pt in part] for part in line.parts]
        self.assertEqual(parts, [[(3.0, 4.0), (10.0, 50.0), (20.0, 25.0)]])
        self.assertEqual(line.len, 73.455384532199886)

    def test_Polygon(self):
        poly = self.parser(self.wktPOLYGON)
        self.assertTrue(issubclass(type(poly), Polygon))
        parts = [[pt[:] for pt in part] for part in poly.parts]
        self.assertEqual(parts, [[(1.0, 1.0), (1.0, 5.0), (5.0, 5.0), (5.0,
                                                                        1.0), (1.0, 1.0)], [(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0),
                                                                                            (2.0, 2.0)]])
        self.assertEqual(
            poly.centroid, (2.9705882352941178, 2.9705882352941178))
        self.assertEqual(poly.area, 17.0)

    def test_fromWKT(self):
        for wkt in self.unsupported:
            self.assertRaises(
                NotImplementedError, self.parser.fromWKT, wkt)
        for wkt in self.empty:
            self.assertEqual(self.parser.fromWKT(wkt), None)
        self.assertEqual(self.parser.__call__, self.parser.fromWKT)

if __name__ == '__main__':
    unittest.main()
