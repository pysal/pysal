import unittest
import pysal
import numpy as np


class TestBbcommon(unittest.TestCase):
    def test_bbcommon(self):
        # self.assertEqual(expected, bbcommon(bb, bbother))
        assert False # TODO: implement your test here

class TestGetBoundingBox(unittest.TestCase):
    def test_get_bounding_box(self):
        # self.assertEqual(expected, get_bounding_box(items))
        assert False # TODO: implement your test here

class TestGetAngleBetween(unittest.TestCase):
    def test_get_angle_between(self):
        # self.assertEqual(expected, get_angle_between(ray1, ray2))
        assert False # TODO: implement your test here

class TestIsCollinear(unittest.TestCase):
    def test_is_collinear(self):
        # self.assertEqual(expected, is_collinear(p1, p2, p3))
        assert False # TODO: implement your test here

class TestGetSegmentsIntersect(unittest.TestCase):
    def test_get_segments_intersect(self):
        # self.assertEqual(expected, get_segments_intersect(seg1, seg2))
        assert False # TODO: implement your test here

class TestGetSegmentPointIntersect(unittest.TestCase):
    def test_get_segment_point_intersect(self):
        # self.assertEqual(expected, get_segment_point_intersect(seg, pt))
        assert False # TODO: implement your test here

class TestGetPolygonPointIntersect(unittest.TestCase):
    def test_get_polygon_point_intersect(self):
        # self.assertEqual(expected, get_polygon_point_intersect(poly, pt))
        assert False # TODO: implement your test here

class TestGetRectanglePointIntersect(unittest.TestCase):
    def test_get_rectangle_point_intersect(self):
        # self.assertEqual(expected, get_rectangle_point_intersect(rect, pt))
        assert False # TODO: implement your test here

class TestGetRaySegmentIntersect(unittest.TestCase):
    def test_get_ray_segment_intersect(self):
        # self.assertEqual(expected, get_ray_segment_intersect(ray, seg))
        assert False # TODO: implement your test here

class TestGetRectangleRectangleIntersection(unittest.TestCase):
    def test_get_rectangle_rectangle_intersection(self):
        # self.assertEqual(expected, get_rectangle_rectangle_intersection(r0, r1, checkOverlap))
        assert False # TODO: implement your test here

class TestGetPolygonPointDist(unittest.TestCase):
    def test_get_polygon_point_dist(self):
        # self.assertEqual(expected, get_polygon_point_dist(poly, pt))
        assert False # TODO: implement your test here

class TestGetPointsDist(unittest.TestCase):
    def test_get_points_dist(self):
        # self.assertEqual(expected, get_points_dist(pt1, pt2))
        assert False # TODO: implement your test here

class TestGetSegmentPointDist(unittest.TestCase):
    def test_get_segment_point_dist(self):
        # self.assertEqual(expected, get_segment_point_dist(seg, pt))
        assert False # TODO: implement your test here

class TestGetPointAtAngleAndDist(unittest.TestCase):
    def test_get_point_at_angle_and_dist(self):
        # self.assertEqual(expected, get_point_at_angle_and_dist(ray, angle, dist))
        assert False # TODO: implement your test here

class TestConvexHull(unittest.TestCase):
    def test_convex_hull(self):
        # self.assertEqual(expected, convex_hull(points))
        assert False # TODO: implement your test here

class TestIsClockwise(unittest.TestCase):
    def test_is_clockwise(self):
        # self.assertEqual(expected, is_clockwise(vertices))
        assert False # TODO: implement your test here

class TestPointTouchesRectangle(unittest.TestCase):
    def test_point_touches_rectangle(self):
        # self.assertEqual(expected, point_touches_rectangle(point, rect))
        assert False # TODO: implement your test here

class TestGetSharedSegments(unittest.TestCase):
    def test_get_shared_segments(self):
        # self.assertEqual(expected, get_shared_segments(poly1, poly2, bool_ret))
        assert False # TODO: implement your test here

class TestDistanceMatrix(unittest.TestCase):
    def test_distance_matrix(self):
        points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        dist = pysal.cg.distance_matrix(np.array(points), 2)
        for i in range(0,len(points)):
            for j in range(i,len(points)):
                x,y = points[i]
                X,Y = points[j]
                d = ((x-X)**2+(y-Y)**2)**(0.5)
                self.assertEqual(dist[i,j],d)

if __name__ == '__main__':
    unittest.main()
