
"""locators Unittest."""
from pysal.cg import *
import unittest


class PolygonLocator_Tester(unittest.TestCase):
    """setup class for unit tests."""
    def setUp(self):
        p1 = Polygon([Point((0, 1)), Point((4, 5)), Point((5, 1))])
        p2 = Polygon([Point((3, 9)), Point((6, 7)), Point((1, 1))])
        p3 = Polygon([Point((7, 1)), Point((8, 7)), Point((9, 1))])
        self.polygons = [p1, p2, p3]
        self.pl = PolygonLocator(self.polygons)

        pt = Point
        pg = Polygon
        polys = []
        for i in range(5):
            l = i * 10
            r = l + 10
            b = 10
            t = 20
            sw = pt((l, b))
            se = pt((r, b))
            ne = pt((r, t))
            nw = pt((l, t))
            polys.append(pg([sw, se, ne, nw]))
        self.pl2 = PolygonLocator(polys)

    def test_PolygonLocator(self):
        qr = Rectangle(3, 7, 5, 8)
        res = self.pl.inside(qr)
        self.assertEqual(len(res), 0)

    def test_inside(self):
        qr = Rectangle(3, 3, 5, 5)
        res = self.pl.inside(qr)
        self.assertEqual(len(res), 0)
        qr = Rectangle(0, 0, 5, 5)
        res = self.pl.inside(qr)
        self.assertEqual(len(res), 1)

    def test_overlapping(self):

        qr = Rectangle(3, 3, 5, 5)
        res = self.pl.overlapping(qr)
        self.assertEqual(len(res), 2)
        qr = Rectangle(8, 3, 10, 10)
        res = self.pl.overlapping(qr)
        self.assertEqual(len(res), 1)

        qr = Rectangle(2, 12, 35, 15)
        res = self.pl2.overlapping(qr)
        self.assertEqual(len(res), 4)

suite = unittest.TestSuite()
test_classes = [PolygonLocator_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
