"""Segment Locator Unittest."""
from pysal.cg import *
from pysal.cg.segmentLocator import *
import unittest


class SegmentGrid_Tester(unittest.TestCase):
    """setup class for unit tests."""
    def setUp(self):
        # 10x10 grid with four line segments, one for each edge of the grid.
        self.grid = SegmentGrid(Rectangle(0, 0, 10, 10), 1)
        self.grid.add(LineSegment(Point((0.0, 0.0)), Point((0.0, 10.0))), 0)
        self.grid.add(LineSegment(Point((0.0, 10.0)), Point((10.0, 10.0))), 1)
        self.grid.add(LineSegment(Point((10.0, 10.0)), Point((10.0, 0.0))), 2)
        self.grid.add(LineSegment(Point((10.0, 0.0)), Point((0.0, 0.0))), 3)

    def test_nearest_1(self):
        self.assertEquals([0, 1, 2, 3], self.grid.nearest(Point((
            5.0, 5.0))))  # Center
        self.assertEquals(
            [0], self.grid.nearest(Point((0.0, 5.0))))  # Left Edge
        self.assertEquals(
            [1], self.grid.nearest(Point((5.0, 10.0))))  # Top Edge
        self.assertEquals(
            [2], self.grid.nearest(Point((10.0, 5.0))))  # Right Edge
        self.assertEquals(
            [3], self.grid.nearest(Point((5.0, 0.0))))  # Bottom Edge

    def test_nearest_2(self):
        self.assertEquals([0, 1, 3], self.grid.nearest(Point((-
                                                              100000.0, 5.0))))  # Left Edge
        self.assertEquals([1, 2, 3], self.grid.nearest(Point((
            100000.0, 5.0))))  # Right Edge
        self.assertEquals([0, 2, 3], self.grid.nearest(Point((5.0,
                                                              -100000.0))))  # Bottom Edge
        self.assertEquals([0, 1, 2], self.grid.nearest(Point((5.0,
                                                              100000.0))))  # Top Edge


suite = unittest.TestSuite()
test_classes = [SegmentGrid_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
