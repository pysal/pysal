from pysal.cg import Point, LineSegment, Line, Ray, Chain, Rectangle, Polygon
import doctest
import unittest


class test_Point(unittest.TestCase):

    def test___init__1(self):
        """
        Tests whether points are created without issue.

        Test tag: <tc>#tests#Point.__init__</tc>
        """
        for l in [(-5.0, 10.0), (0.0, -6.0), (float(1e300), float(-1e300))]:
            p = Point(l)

    def test___str__1(self):
        """
        Tests whether the string produced is valid for corner cases.

        Test tag: <tc>#tests#Point__str__</tc>
        """
        for l in [(-5, 10), (0, -6.0), (float(1e300), -1e300)]:
            p = Point(l)
            self.assertEquals(str(p), str((float(l[0]), float(
                l[1]))))  # Recast to floats like point does


class test_LineSegment(unittest.TestCase):

    def test_is_ccw1(self):
        """
        Test corner cases for horizontal segment starting at origin.

        Test tag: <tc>#tests#LineSegment.is_ccw</tc>
        """
        ls = LineSegment(Point((0, 0)), Point((5, 0)))
        self.assertFalse(ls.is_ccw(
            Point((10, 0))))  # At positive boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((3, 0))))  # On segment
        self.assertFalse(ls.is_ccw(
            Point((-10, 0))))  # At negative boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((0, 0))))  # Endpoint of segment
        self.assertFalse(ls.is_ccw(Point((5, 0))))  # Endpoint of segment

    def test_is_ccw2(self):
        """
        Test corner cases for vertical segment ending at origin.

        Test tag: <tc>#tests#LineSegment.is_ccw</tc>
        """
        ls = LineSegment(Point((0, -5)), Point((0, 0)))
        self.assertFalse(ls.is_ccw(
            Point((0, 10))))  # At positive boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((0, -3))))  # On segment
        self.assertFalse(ls.is_ccw(
            Point((0, -10))))  # At negative boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((0, -5))))  # Endpoint of segment
        self.assertFalse(ls.is_ccw(Point((0, 0))))  # Endpoint of segment

    def test_is_ccw3(self):
        """
        Test corner cases for non-axis-aligned segment not through origin.

        Test tag: <tc>#tests#LineSegment.is_ccw</tc>
        """
        ls = LineSegment(Point((0, 1)), Point((5, 6)))
        self.assertFalse(ls.is_ccw(
            Point((10, 11))))  # At positive boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((3, 4))))  # On segment
        self.assertFalse(ls.is_ccw(
            Point((-10, -9))))  # At negative boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((0, 1))))  # Endpoint of segment
        self.assertFalse(ls.is_ccw(Point((5, 6))))  # Endpoint of segment

    def test_is_cw1(self):
        """
        Test corner cases for horizontal segment starting at origin.

        Test tag: <tc>#tests#LineSegment.is_cw</tc>
        """
        ls = LineSegment(Point((0, 0)), Point((5, 0)))
        self.assertFalse(ls.is_cw(
            Point((10, 0))))  # At positive boundary beyond segment
        self.assertFalse(ls.is_cw(Point((3, 0))))  # On segment
        self.assertFalse(ls.is_cw(
            Point((-10, 0))))  # At negative boundary beyond segment
        self.assertFalse(ls.is_cw(Point((0, 0))))  # Endpoint of segment
        self.assertFalse(ls.is_cw(Point((5, 0))))  # Endpoint of segment

    def test_is_cw2(self):
        """
        Test corner cases for vertical segment ending at origin.

        Test tag: <tc>#tests#LineSegment.is_cw</tc>
        """
        ls = LineSegment(Point((0, -5)), Point((0, 0)))
        self.assertFalse(ls.is_cw(
            Point((0, 10))))  # At positive boundary beyond segment
        self.assertFalse(ls.is_cw(Point((0, -3))))  # On segment
        self.assertFalse(ls.is_cw(
            Point((0, -10))))  # At negative boundary beyond segment
        self.assertFalse(ls.is_cw(Point((0, -5))))  # Endpoint of segment
        self.assertFalse(ls.is_cw(Point((0, 0))))  # Endpoint of segment

    def test_is_cw3(self):
        """
        Test corner cases for non-axis-aligned segment not through origin.

        Test tag: <tc>#tests#LineSegment.is_cw</tc>
        """
        ls = LineSegment(Point((0, 1)), Point((5, 6)))
        self.assertFalse(ls.is_cw(
            Point((10, 11))))  # At positive boundary beyond segment
        self.assertFalse(ls.is_cw(Point((3, 4))))  # On segment
        self.assertFalse(ls.is_cw(
            Point((-10, -9))))  # At negative boundary beyond segment
        self.assertFalse(ls.is_cw(Point((0, 1))))  # Endpoint of segment
        self.assertFalse(ls.is_cw(Point((5, 6))))  # Endpoint of segment

    def test_get_swap1(self):
        """
        Tests corner cases.

        Test tag: <tc>#tests#LineSegment.get_swap</tc>
        """
        ls = LineSegment(Point((0, 0)), Point((10, 0)))
        swap = ls.get_swap()
        self.assertEquals(ls.p1, swap.p2)
        self.assertEquals(ls.p2, swap.p1)

        ls = LineSegment(Point((-5, 0)), Point((5, 0)))
        swap = ls.get_swap()
        self.assertEquals(ls.p1, swap.p2)
        self.assertEquals(ls.p2, swap.p1)

        ls = LineSegment(Point((0, 0)), Point((0, 0)))
        swap = ls.get_swap()
        self.assertEquals(ls.p1, swap.p2)
        self.assertEquals(ls.p2, swap.p1)

        ls = LineSegment(Point((5, 5)), Point((5, 5)))
        swap = ls.get_swap()
        self.assertEquals(ls.p1, swap.p2)
        self.assertEquals(ls.p2, swap.p1)

    def test_bounding_box(self):
        """
        Tests corner cases.

        Test tag: <tc>#tests#LineSegment.bounding_box</tc>
        """
        ls = LineSegment(Point((0, 0)), Point((0, 10)))
        self.assertEquals(ls.bounding_box.left, 0)
        self.assertEquals(ls.bounding_box.lower, 0)
        self.assertEquals(ls.bounding_box.right, 0)
        self.assertEquals(ls.bounding_box.upper, 10)

        ls = LineSegment(Point((0, 0)), Point((-3, -4)))
        self.assertEquals(ls.bounding_box.left, -3)
        self.assertEquals(ls.bounding_box.lower, -4)
        self.assertEquals(ls.bounding_box.right, 0)
        self.assertEquals(ls.bounding_box.upper, 0)

        ls = LineSegment(Point((-5, 0)), Point((3, 0)))
        self.assertEquals(ls.bounding_box.left, -5)
        self.assertEquals(ls.bounding_box.lower, 0)
        self.assertEquals(ls.bounding_box.right, 3)
        self.assertEquals(ls.bounding_box.upper, 0)

    def test_len1(self):
        """
        Tests corner cases.

        Test tag: <tc>#tests#LineSegment.len</tc>
        """
        ls = LineSegment(Point((0, 0)), Point((0, 0)))
        self.assertEquals(ls.len, 0)

        ls = LineSegment(Point((0, 0)), Point((-3, 0)))
        self.assertEquals(ls.len, 3)

    def test_line1(self):
        """
        Tests corner cases.

        Test tag: <tc>#tests#LineSegment.line</tc>
        """
        import math
        ls = LineSegment(Point((0, 0)), Point((1, 0)))
        self.assertEquals(ls.line.m, 0)
        self.assertEquals(ls.line.b, 0)

        ls = LineSegment(Point((0, 0)), Point((0, 1)))
        self.assertEquals(ls.line.m, float('inf'))
        self.assertTrue(math.isnan(ls.line.b))

        ls = LineSegment(Point((0, 0)), Point((0, -1)))
        self.assertEquals(ls.line.m, float('inf'))
        self.assertTrue(math.isnan(ls.line.b))

        ls = LineSegment(Point((0, 0)), Point((0, 0)))
        self.assertEquals(ls.line, None)

        ls = LineSegment(Point((5,0)), Point((10,0)))
        ls1 = LineSegment(Point((5,0)), Point((10,1)))
        self.assertTrue(ls.intersect(ls1))
        ls2 = LineSegment(Point((5,1)), Point((10,1)))
        self.assertFalse(ls.intersect(ls2))
        ls2 = LineSegment(Point((7,-1)), Point((7,2)))
        self.assertTrue(ls.intersect(ls2))
        




class test_Line(unittest.TestCase):

    def test___init__1(self):
        """
        Tests a variety of generic cases.

        Test tag: <tc>#tests#Line.__init__</tc>
        """
        for m, b in [(4, 0.0), (-140, 5), (0, 0)]:
            l = Line(m, b)

    def test_y1(self):
        """
        Tests a variety of generic and special cases (+-infinity).

        Test tag: <tc>#tests#Line.y</tc>
        """
        l = Line(0, 0)
        self.assertEquals(l.y(0), 0)
        self.assertEquals(l.y(-1e600), 0)
        self.assertEquals(l.y(1e600), 0)

        l = Line(1, 1)
        self.assertEquals(l.y(2), 3)
        self.assertEquals(l.y(-1e600), -1e600)
        self.assertEquals(l.y(1e600), 1e600)

        l = Line(-1, 1)
        self.assertEquals(l.y(2), -1)
        self.assertEquals(l.y(-1e600), 1e600)
        self.assertEquals(l.y(1e600), -1e600)

    def test_x1(self):
        """
        Tests a variety of generic and special cases (+-infinity).

        Test tag: <tc>#tests#Line.x</tc>
        """
        l = Line(0, 0)
        #self.assertEquals(l.x(0), 0)
        with self.assertRaises(ArithmeticError):
            l.x(0)
        with self.assertRaises(ArithmeticError):
            l.x(-1e600)
        with self.assertRaises(ArithmeticError):
            l.x(1e600)

        l = Line(1, 1)
        self.assertEquals(l.x(3), 2)
        self.assertEquals(l.x(-1e600), -1e600)
        self.assertEquals(l.x(1e600), 1e600)

        l = Line(-1, 1)
        self.assertEquals(l.x(2), -1)
        self.assertEquals(l.x(-1e600), 1e600)
        self.assertEquals(l.x(1e600), -1e600)


class test_Ray(unittest.TestCase):

    def test___init__1(self):
        """
        Tests generic cases.

        <tc>#tests#Ray.__init__</tc>
        """
        r = Ray(Point((0, 0)), Point((1, 1)))
        r = Ray(Point((8, -3)), Point((-5, 9)))


class test_Chain(unittest.TestCase):

    def test___init__1(self):
        """
        Generic testing that no exception is thrown.

        Test tag: <tc>#tests#Chain.__init__</tc>
        """
        c = Chain([Point((0, 0))])
        c = Chain([[Point((0, 0)), Point((1, 1))], [Point((2, 5))]])

    def test_vertices1(self):
        """
        Testing for repeated vertices and multiple parts.

        Test tag: <tc>#tests#Chain.vertices</tc>
        """
        vertices = [Point((0, 0)), Point((1, 1)), Point((2, 5)),
                    Point((0, 0)), Point((1, 1)), Point((2, 5))]
        self.assertEquals(Chain(vertices).vertices, vertices)

        vertices = [[Point((0, 0)), Point((1, 1)), Point((2, 5))],
                    [Point((0, 0)), Point((1, 1)), Point((2, 5))]]
        self.assertEquals(Chain(vertices).vertices, vertices[0] + vertices[1])

    def test_parts1(self):
        """
        Generic testing of parts functionality.

        Test tag: <tc>#tests#Chain.parts</tc>
        """
        vertices = [Point((0, 0)), Point((1, 1)), Point((2, 5)),
                    Point((0, 0)), Point((1, 1)), Point((2, 5))]
        self.assertEquals(Chain(vertices).parts, [vertices])

        vertices = [[Point((0, 0)), Point((1, 1)), Point((2, 5))],
                    [Point((0, 0)), Point((1, 1)), Point((2, 5))]]
        self.assertEquals(Chain(vertices).parts, vertices)

    def test_bounding_box1(self):
        """
        Test correctness with multiple parts.

        Test tag: <tc>#tests#Chain.bounding_box</tc>
        """
        vertices = [[Point((0, 0)), Point((1, 1)), Point((2, 6))],
                    [Point((-5, -5)), Point((0, 0)), Point((2, 5))]]
        bb = Chain(vertices).bounding_box
        self.assertEquals(bb.left, -5)
        self.assertEquals(bb.lower, -5)
        self.assertEquals(bb.right, 2)
        self.assertEquals(bb.upper, 6)

    def test_len1(self):
        """
        Test correctness with multiple parts and zero-length point-to-point distances.

        Test tag: <tc>#tests#Chain.len</tc>
        """
        vertices = [[Point((0, 0)), Point((1, 0)), Point((1, 5))],
                    [Point((-5, -5)), Point((-5, 0)), Point((0, 0)), Point((0, 0))]]
        self.assertEquals(Chain(vertices).len, 6 + 10)


class test_Polygon(unittest.TestCase):

    def test___init__1(self):
        """
        Test various input configurations (list vs. lists of lists, holes)

        <tc>#tests#Polygon.__init__</tc>
        """
        # Input configurations tested (in order of test):
        # one part, no holes
        # multi parts, no holes
        # one part, one hole
        # multi part, one hole
        # one part, multi holes
        # multi part, multi holes
        p = Polygon([Point(
            (0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))])
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]])
        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))])
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point(
                         (
                             30, 30)), Point(
                                 (40, 30)), Point((40, 40)), Point((30, 40))]],
                    holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))])
        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[[Point(
                        (2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                        [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]])
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point(
                         (
                             30, 30)), Point(
                                 (40, 30)), Point((40, 40)), Point((30, 40))]],
                    holes=[[Point(
                        (2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                        [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]])

    def test_area1(self):
        """
        Test multiple parts.

        Test tag: <tc>#tests#Polygon.area</tc>
        """
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]])
        self.assertEquals(p.area, 200)

    def test_area2(self):
        """
        Test holes.

        Test tag: <tc>#tests#Polygon.area</tc>
        """
        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))])
        self.assertEquals(p.area, 100 - 4)

        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[[Point(
                        (2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                        [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]])
        self.assertEquals(p.area, 100 - (4 + 4))

        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point(
                         (
                             30, 30)), Point(
                                 (40, 30)), Point((40, 40)), Point((30, 40))]],
                    holes=[[Point(
                        (2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                        [Point((36, 36)), Point((36, 38)), Point((38, 38)), Point((38, 36))]])
        self.assertEquals(p.area, 200 - (4 + 4))

    def test_area4(self):
        """
        Test polygons with vertices in both orders (cw, ccw).

        Test tag: <tc>#tests#Polygon.area</tc>
        """
        p = Polygon([Point(
            (0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))])
        self.assertEquals(p.area, 100)

        p = Polygon([Point(
            (0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))])
        self.assertEquals(p.area, 100)

    def test_bounding_box1(self):
        """
        Test polygons with multiple parts.

        Test tag: <tc>#tests#Polygon.bounding_box</tc>
        """
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]])
        bb = p.bounding_box
        self.assertEquals(bb.left, 0)
        self.assertEquals(bb.lower, 0)
        self.assertEquals(bb.right, 40)
        self.assertEquals(bb.upper, 40)

    def test_centroid1(self):
        """
        Test polygons with multiple parts of the same size.

        Test tag: <tc>#tests#Polygon.centroid</tc>
        """
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]])
        c = p.centroid
        self.assertEquals(c[0], 20)
        self.assertEquals(c[1], 20)

    def test_centroid2(self):
        """
        Test polygons with multiple parts of different size.

        Test tag: <tc>#tests#Polygon.centroid</tc>
        """
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((35, 30)), Point((35, 35)), Point((30, 35))]])
        c = p.centroid
        self.assertEquals(c[0], 10.5)
        self.assertEquals(c[1], 10.5)

    def test_holes1(self):
        """
        Test for correct vertex values/order.

        Test tag: <tc>#tests#Polygon.holes</tc>
        """
        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))])
        self.assertEquals(len(p.holes), 1)
        e_holes = [Point((2, 2)), Point((2, 4)), Point((4, 4)), Point((4, 2))]
        self.assertTrue(p.holes[0] in [e_holes, [e_holes[-1]] + e_holes[:3],
                                       e_holes[-2:] + e_holes[:2], e_holes[-3:] + [e_holes[0]]])

    def test_holes2(self):
        """
        Test for multiple holes.

        Test tag: <tc>#tests#Polygon.holes</tc>
        """
        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[[Point(
                        (2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                        [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]])
        holes = p.holes
        self.assertEquals(len(holes), 2)

    def test_parts1(self):
        """
        Test for correct vertex values/order.

        Test tag: <tc>#tests#Polygon.parts</tc>
        """
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((30, 40))]])
        self.assertEquals(len(p.parts), 2)

        part1 = [Point(
            (0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))]
        part2 = [Point((30, 30)), Point((30, 40)), Point((40, 30))]
        if len(p.parts[0]) == 4:
            self.assertTrue(p.parts[0] in [part1, part1[-1:] + part1[:3],
                                           part1[-2:] + part1[:2], part1[-3:] + part1[:1]])
            self.assertTrue(p.parts[1] in [part2, part2[-1:] +
                                           part2[:2], part2[-2:] + part2[:1]])
        elif len(p.parts[0]) == 3:
            self.assertTrue(p.parts[0] in [part2, part2[-1:] +
                                           part2[:2], part2[-2:] + part2[:1]])
            self.assertTrue(p.parts[1] in [part1, part1[-1:] + part1[:3],
                                           part1[-2:] + part1[:2], part1[-3:] + part1[:1]])
        else:
            self.fail()

    def test_perimeter1(self):
        """
        Test with multiple parts.

        Test tag: <tc>#tests#Polygon.perimeter</tc>
        """
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]])
        self.assertEquals(p.perimeter, 80)

    def test_perimeter2(self):
        """
        Test with holes.

        Test tag: <tc>#tests#Polygon.perimeter</tc>
        """
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point(
                         (
                             30, 30)), Point(
                                 (40, 30)), Point((40, 40)), Point((30, 40))]],
                    holes=[[Point(
                        (2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                        [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]])
        self.assertEquals(p.perimeter, 80 + 16)

    def test_vertices1(self):
        """
        Test for correct values/order of vertices.

        Test tag: <tc>#tests#Polygon.vertices</tc>
        """
        p = Polygon([Point(
            (0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))])
        self.assertEquals(len(p.vertices), 4)
        e_verts = [Point(
            (0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))]
        self.assertTrue(p.vertices in [e_verts, e_verts[-1:] + e_verts[:3],
                                       e_verts[-2:] + e_verts[:2], e_verts[-3:] + e_verts[:1]])

    def test_vertices2(self):
        """
        Test for multiple parts.

        Test tag: <tc>#tests#Polygon.vertices</tc>
        """
        p = Polygon(
            [[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]])
        self.assertEquals(len(p.vertices), 8)

    def test_contains_point(self):
        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))], [Point((1, 2)), Point((2, 2)), Point((2, 1)), Point((1, 1))])
        self.assertEquals(p.contains_point((0, 0)), 0)
        self.assertEquals(p.contains_point((1, 1)), 0)
        self.assertEquals(p.contains_point((5, 5)), 1)
        self.assertEquals(p.contains_point((10, 10)), 0)


class test_Rectangle(unittest.TestCase):

    def test___init__1(self):
        """
        Test exceptions are thrown correctly.

        Test tag: <tc>#tests#Rectangle.__init__</tc>
        """
        try:
            r = Rectangle(1, 1, -1, 5)  # right < left
        except ArithmeticError:
            pass
        else:
            self.fail()

        try:
            r = Rectangle(1, 1, 5, -1)  # upper < lower
        except ArithmeticError:
            pass
        else:
            self.fail()

    def test_set_centroid1(self):
        """
        Test with rectangles of zero width or height.

        Test tag: <tc>#tests#Rectangle.set_centroid</tc>
        """
        r = Rectangle(5, 5, 5, 10)  # Zero width
        r.set_centroid(Point((0, 0)))
        self.assertEquals(r.left, 0)
        self.assertEquals(r.lower, -2.5)
        self.assertEquals(r.right, 0)
        self.assertEquals(r.upper, 2.5)

        r = Rectangle(10, 5, 20, 5)  # Zero height
        r.set_centroid(Point((40, 40)))
        self.assertEquals(r.left, 35)
        self.assertEquals(r.lower, 40)
        self.assertEquals(r.right, 45)
        self.assertEquals(r.upper, 40)

        r = Rectangle(0, 0, 0, 0)  # Zero width and height
        r.set_centroid(Point((-4, -4)))
        self.assertEquals(r.left, -4)
        self.assertEquals(r.lower, -4)
        self.assertEquals(r.right, -4)
        self.assertEquals(r.upper, -4)

    def test_set_scale1(self):
        """
        Test repeated scaling.

        Test tag: <tc>#tests#Rectangle.set_scale</tc>
        """
        r = Rectangle(2, 2, 4, 4)

        r.set_scale(0.5)
        self.assertEquals(r.left, 2.5)
        self.assertEquals(r.lower, 2.5)
        self.assertEquals(r.right, 3.5)
        self.assertEquals(r.upper, 3.5)

        r.set_scale(2)
        self.assertEquals(r.left, 2)
        self.assertEquals(r.lower, 2)
        self.assertEquals(r.right, 4)
        self.assertEquals(r.upper, 4)

    def test_set_scale2(self):
        """
        Test scaling of rectangles with zero width/height..

        Test tag: <tc>#tests#Rectangle.set_scale</tc>
        """
        r = Rectangle(5, 5, 5, 10)  # Zero width
        r.set_scale(2)
        self.assertEquals(r.left, 5)
        self.assertEquals(r.lower, 2.5)
        self.assertEquals(r.right, 5)
        self.assertEquals(r.upper, 12.5)

        r = Rectangle(10, 5, 20, 5)  # Zero height
        r.set_scale(2)
        self.assertEquals(r.left, 5)
        self.assertEquals(r.lower, 5)
        self.assertEquals(r.right, 25)
        self.assertEquals(r.upper, 5)

        r = Rectangle(0, 0, 0, 0)  # Zero width and height
        r.set_scale(100)
        self.assertEquals(r.left, 0)
        self.assertEquals(r.lower, 0)
        self.assertEquals(r.right, 0)
        self.assertEquals(r.upper, 0)

        r = Rectangle(0, 0, 0, 0)  # Zero width and height
        r.set_scale(0.01)
        self.assertEquals(r.left, 0)
        self.assertEquals(r.lower, 0)
        self.assertEquals(r.right, 0)
        self.assertEquals(r.upper, 0)

    def test_area1(self):
        """
        Test rectangles with zero width/height

        Test tag: <tc>#tests#Rectangle.area</tc>
        """
        r = Rectangle(5, 5, 5, 10)  # Zero width
        self.assertEquals(r.area, 0)

        r = Rectangle(10, 5, 20, 5)  # Zero height
        self.assertEquals(r.area, 0)

        r = Rectangle(0, 0, 0, 0)  # Zero width and height
        self.assertEquals(r.area, 0)

    def test_height1(self):
        """
        Test rectangles with zero height.

        Test tag: <tc>#tests#Rectangle.height</tc>
        """
        r = Rectangle(10, 5, 20, 5)  # Zero height
        self.assertEquals(r.height, 0)

    def test_width1(self):
        """
        Test rectangles with zero width.

        Test tag: <tc>#tests#Rectangle.width</tc>
        """
        r = Rectangle(5, 5, 5, 10)  # Zero width
        self.assertEquals(r.width, 0)


#suite = unittest.TestSuite()
#suite.addTest(doctest.DocTestSuite('pysal.cg.shapes'))
#A = unittest.TestLoader().loadTestsFromTestCase(_TestPoint)
#B = unittest.TestLoader().loadTestsFromTestCase(_TestLineSegment)
#C = unittest.TestLoader().loadTestsFromTestCase(_TestLine)
#D = unittest.TestLoader().loadTestsFromTestCase(_TestRay)
#E = unittest.TestLoader().loadTestsFromTestCase(_TestChain)
#F = unittest.TestLoader().loadTestsFromTestCase(_TestPolygon)
#G = unittest.TestLoader().loadTestsFromTestCase(_TestRectangle)
#suite.addTests([A,B,C,D,E,D,G])
if __name__ == '__main__':
    unittest.main()
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
