"""
Computational geometry code for PySAL: Python Spatial Analysis Library.

Authors:
Sergio Rey <srey@asu.edu>
Xinyue Ye <xinyue.ye@gmail.com>
Charles Schmidt <Charles.Schmidt@asu.edu>
Andrew Winslow <Andrew.Winslow@asu.edu>

Not to be used without permission of the authors. 

Style Guide, Follow:
http://www.python.org/dev/peps/pep-0008/


Class comment format:

    Brief class description.

    Attributes:
    attr 1 -- type -- description of attr 1
    attr 2 -- type -- description of attr 2

    Extras (notes, references, examples, doctest, etc.)


Function comment format:

    Brief function description.

    function(arg 1 type, arg 2 type, keyword=keyword arg 3 type) -> return type

    Argument:
    arg 1 -- description of arg 1
    arg 2 -- description of arg 2

    Keyword Arguments:
    arg 3 -- description of arg 3

    Extras (notes, references, examples, doctest, etc.)
"""

__author__  = "Sergio J. Rey, Xinyue Ye, Charles Schmidt, Andrew Winslow"
__credits__ = "Copyright (c) 2005-2009 Sergio J. Rey"

import doctest
import math
import random
import unittest
import standalone

class Point(object):
    """
    Geometric class for point objects.

    Attributes:
    None
    """
    def __init__(self, loc):
        """
        Returns an instance of a Point object.

        __init__((number, number)) -> Point 

        Test tag: <tc>#is#Point.__init__</tc>    
        Test tag: <tc>#tests#Point.__init__</tc>    
 
        Arguments:
        loc -- tuple location (number x-tuple, x > 1) 

        Example:
        >>> p = Point((1, 3)) 
        """
        self.__loc = tuple(map(float, loc))

    def __eq__(self,other):
        """
        Tests if the Point is equal to another object.

        __eq__(x) -> bool

        Arguments:
        other -- an object to test equality against

        Example:
        >>> Point((0,1)) == Point((0,1))
        True
        >>> Point((0,1)) == Point((1,1))
        False
        """
        try:
            return (self.__loc) == (other.__loc)
        except AttributeError:
            return False

    def __ne__(self,other):
        """
        Tests if the Point is not equal to another object.

        __ne__(x) -> bool

        Arguments:
        other -- an object to test equality against

        Example:
        >>> Point((0,1)) != Point((0,1))
        False
        >>> Point((0,1)) != Point((1,1))
        True
        """
        try:
            return (self.__loc) != (other.__loc)
        except AttributeError:
            return True

    def __hash__(self):
        """
        Returns the hash of the Point's location.

        x.__hash__() -> hash(x)

        Arguments:
        None

        Example:
        >>> hash(Point((0,1))) == hash(Point((0,1)))
        True
        >>> hash(Point((0,1))) == hash(Point((1,1)))
        False
        """
        return hash(self.__loc)

    def __getitem__(self,*args):
        """
        Return the coordinate for the given dimension.

        x.__getitem__(i) -> x[i]

        Arguments:
        i -- index of the desired dimension.

        Example:
        >>> p = Point((5.5,4.3))
        >>> p[0] == 5.5
        True
        >>> p[1] == 4.3
        True
        """
        return self.__loc.__getitem__(*args)

    def __getslice__(self,*args):
        """
        Return the coordinate for the given dimensions.

        x.__getitem__(i,j) -> x[i:j]

        Arguments:
        i -- index to start slice
        j -- index to end slice (excluded).

        Example:
        >>> p = Point((3,6,2))
        >>> p[:2] == (3,6)
        True
        >>> p[1:2] == (6,)
        True
        """
        return self.__loc.__getslice__(*args)

    def __len__(self):
        """
        Returns the number of dimension in the point.

        __len__() -> int

        Arguments:
        None

        Example:
        >>> len(Point((1,2)))
        2
        """
        return len(self.__loc)

    def __repr__(self):
        """
        Returns the string representation of the Point

        __repr__() -> string
        
        Arguments:
        None

        Example:
        >>> Point((0,1))
        (0.0, 1.0)
        """
        return self.__loc.__repr__()

    def __str__(self):
        """
        Returns a string representation of a Point object.
                   
        __str__() -> string 

        Test tag: <tc>#is#Point.__str__</tc>    
        Test tag: <tc>#tests#Point.__str__</tc>    
 
        Example:
        >>> p = Point((1, 3))
        >>> str(p)
        '(1.0, 3.0)'
        """
        return str(self.__loc)

class _TestPoint(unittest.TestCase):

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
            self.assertEquals(str(p), str((float(l[0]), float(l[1])))) # Recast to floats like point does       

class LineSegment:
    """
    Geometric representation of line segment objects.

    Attributes:
    p1 -- starting point (Point)
    p2 -- ending point (Point)
    bounding_box -- the bounding box of the segment (number 4-tuple)
    len -- the length of the segment (number)
    line -- the line on which the segment lies (Line)
    """
 
    def __init__(self, start_pt, end_pt):
        """
        Creates a LineSegment object.
 
        __init__(Point, Point) -> LineSegment

        Test tag: <tc>#is#LineSegment.__init__</tc>
        Test tag: <tc>#tests#LineSegment.__init__</tc>

        Arguments:
        start_pt -- point where segment begins
        end_pt -- point where segment ends
 
        Example:
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        """
        self._p1 = start_pt
        self._p2 = end_pt
        self._reset_props()

    def _reset_props(self):
        """
        HELPER METHOD. DO NOT CALL.

        Resets attributes which are functions of other attributes. The getters for these attributes (implemented as
        properties) then recompute their values if they have been reset since the last call to the getter.

        _reset_props() -> None

        Example:
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> ls._reset_props()
        """ 
        self._bounding_box = None
        self._len = None
        self._line = None

    def _get_p1(self):
        """
        HELPER METHOD. DO NOT CALL.

        Returns the p1 attribute of the line segment.

        _get_p1() -> Point

        Example:
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> r = ls._get_p1() 
        >>> r == Point((1, 2))
        True
        """
        return self._p1

    def _set_p1(self, p1):
        """
        HELPER METHOD. DO NOT CALL.

        Sets the p1 attribute of the line segment.

        _set_p1(Point) -> Point

        Example:
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> r = ls._set_p1(Point((3, -1)))
        >>> r == Point((3.0, -1.0))
        True
        """
        self._p1 = p1
        self._reset_props()
        return self._p1

    p1 = property(_get_p1, _set_p1)

    def _get_p2(self):
        """
        HELPER METHOD. DO NOT CALL.

        Returns the p2 attribute of the line segment.

        _get_p2() -> Point

        Example:
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> r = ls._get_p2() 
        >>> r == Point((5, 6))
        True
        """
        return self._p2

    def _set_p2(self, p2):
        """
        HELPER METHOD. DO NOT CALL.

        Sets the p2 attribute of the line segment.

        _set_p2(Point) -> Point

        Example:
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> r = ls._set_p2(Point((3, -1)))
        >>> r == Point((3.0, -1.0))
        True
        """
        self._p2 = p2
        self._reset_props()
        return self._p2

    p2 = property(_get_p2, _set_p2)

    def is_ccw(self, pt):
        """
        Returns whether a point is counterclockwise of the segment. Exclusive.
 
        is_ccw(Point) -> bool

        Test tag: <tc>#is#LineSegment.is_ccw</tc>
        Test tag: <tc>#tests#LineSegment.is_ccw</tc>

        Arguments:
        pt -- point lying ccw or cw of a segment
 
        Example:
        >>> ls = LineSegment(Point((0, 0)), Point((5, 0)))
        >>> ls.is_ccw(Point((2, 2)))
        True
        >>> ls.is_ccw(Point((2, -2)))
        False
        """
        v1 = (self._p2[0] - self._p1[0], self._p2[1] - self._p1[1])
        v2 = (pt[0] - self._p1[0], pt[1] - self._p1[1])
        return v1[0]*v2[1] - v1[1]*v2[0] > 0

    def is_cw(self, pt):
        """
        Returns whether a point is clockwise of the segment. Exclusive. 
 
        is_cw(Point) -> bool

        Test tag: <tc>#is#LineSegment.is_cw</tc>
        Test tag: <tc>#tests#LineSegment.is_cw</tc>

        Arguments:
        pt -- point lying ccw or cw of a segment
 
        Example:
        >>> ls = LineSegment(Point((0, 0)), Point((5, 0)))
        >>> ls.is_cw(Point((2, 2)))
        False
        >>> ls.is_cw(Point((2, -2)))
        True
        """
        v1 = (self._p2[0] - self._p1[0], self._p2[1] - self._p1[1])
        v2 = (pt[0] - self._p1[0], pt[1] - self._p1[1])
        return v1[0]*v2[1] - v1[1]*v2[0] < 0

    def get_swap(self):
        """
        Returns a LineSegment object which has its endpoints swapped.
 
        get_swap() -> LineSegment

        Test tag: <tc>#is#LineSegment.get_swap</tc>
        Test tag: <tc>#tests#LineSegment.get_swap</tc>

        Example:
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> swap = ls.get_swap()
        >>> swap.p1[0]
        5.0
        >>> swap.p1[1]
        6.0
        >>> swap.p2[0]
        1.0
        >>> swap.p2[1]
        2.0
        """
        return LineSegment(self._p2, self._p1) 

    @property
    def bounding_box(self):
        """
        Returns the minimum bounding box of a LineSegment object.

        Test tag: <tc>#is#LineSegment.bounding_box</tc>
        Test tag: <tc>#tests#LineSegment.bounding_box</tc>
 
        bounding_box -> Rectangle

        Example:
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> ls.bounding_box.left
        1.0
        >>> ls.bounding_box.lower
        2.0
        >>> ls.bounding_box.right
        5.0
        >>> ls.bounding_box.upper
        6.0
        """
        if self._bounding_box == None: # If LineSegment attributes p1, p2 changed, recompute
            self._bounding_box = Rectangle(min([self._p1[0], self._p2[0]]), min([self._p1[1], self._p2[1]]),
                                           max([self._p1[0], self._p2[0]]), max([self._p1[1], self._p2[1]]))
        return Rectangle(self._bounding_box.left, self._bounding_box.lower, self._bounding_box.right, 
                         self._bounding_box.upper)

    @property
    def len(self):
        """
        Returns the length of a LineSegment object.

        Test tag: <tc>#is#LineSegment.len</tc>
        Test tag: <tc>#tests#LineSegment.len</tc>
 
        len() -> number

        Example:
        >>> ls = LineSegment(Point((2, 2)), Point((5, 2)))
        >>> ls.len
        3.0
        """
        if self._len == None: # If LineSegment attributes p1, p2 changed, recompute
            self._len = math.hypot(self._p1[0] - self._p2[0], self._p1[1] - self._p2[1])
        return self._len

    @property
    def line(self):
        """
        Returns a Line object of the line which the segment lies on.
 
        Test tag: <tc>#is#LineSegment.line</tc>
        Test tag: <tc>#tests#LineSegment.line</tc>
 
        line() -> Line

        Example:
        >>> ls = LineSegment(Point((2, 2)), Point((3, 3)))
        >>> l = ls.line
        >>> l.m
        1.0
        >>> l.b
        0.0
        """
        if self._line == None:
            dx = self._p1[0] - self._p2[0]
            dy = self._p1[1] - self._p2[1]
            if dx == 0:
                self._line = False
            else:
                m = dy/dx
                b = self._p1[1] - m*self._p1[0] # y - mx
                self._line = Line(m, b)
        if not self._line: # If the line is not defined
            return None
        return Line(self._line.m, self._line.b)

class _TestLineSegment(unittest.TestCase):

    def test_is_ccw1(self):
        """
        Test corner cases for horizontal segment starting at origin.

        Test tag: <tc>#tests#LineSegment.is_ccw</tc>
        """
        ls = LineSegment(Point((0, 0)), Point((5, 0)))
        self.assertFalse(ls.is_ccw(Point((10, 0)))) # At positive boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((3, 0)))) # On segment
        self.assertFalse(ls.is_ccw(Point((-10, 0)))) # At negative boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((0, 0)))) # Endpoint of segment
        self.assertFalse(ls.is_ccw(Point((5, 0)))) # Endpoint of segment

    def test_is_ccw2(self):
        """
        Test corner cases for vertical segment ending at origin.

        Test tag: <tc>#tests#LineSegment.is_ccw</tc>
        """ 
        ls = LineSegment(Point((0, -5)), Point((0, 0)))
        self.assertFalse(ls.is_ccw(Point((0, 10)))) # At positive boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((0, -3)))) # On segment
        self.assertFalse(ls.is_ccw(Point((0, -10)))) # At negative boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((0, -5)))) # Endpoint of segment
        self.assertFalse(ls.is_ccw(Point((0, 0)))) # Endpoint of segment

    def test_is_ccw3(self):
        """
        Test corner cases for non-axis-aligned segment not through origin.

        Test tag: <tc>#tests#LineSegment.is_ccw</tc>
        """ 
        ls = LineSegment(Point((0, 1)), Point((5, 6)))
        self.assertFalse(ls.is_ccw(Point((10, 11)))) # At positive boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((3, 4)))) # On segment
        self.assertFalse(ls.is_ccw(Point((-10, -9)))) # At negative boundary beyond segment
        self.assertFalse(ls.is_ccw(Point((0, 1)))) # Endpoint of segment
        self.assertFalse(ls.is_ccw(Point((5, 6)))) # Endpoint of segment
        
    def test_is_cw1(self):
        """
        Test corner cases for horizontal segment starting at origin.

        Test tag: <tc>#tests#LineSegment.is_cw</tc>
        """
        ls = LineSegment(Point((0, 0)), Point((5, 0)))
        self.assertFalse(ls.is_cw(Point((10, 0)))) # At positive boundary beyond segment
        self.assertFalse(ls.is_cw(Point((3, 0)))) # On segment
        self.assertFalse(ls.is_cw(Point((-10, 0)))) # At negative boundary beyond segment
        self.assertFalse(ls.is_cw(Point((0, 0)))) # Endpoint of segment
        self.assertFalse(ls.is_cw(Point((5, 0)))) # Endpoint of segment

    def test_is_cw2(self):
        """
        Test corner cases for vertical segment ending at origin.

        Test tag: <tc>#tests#LineSegment.is_cw</tc>
        """
        ls = LineSegment(Point((0, -5)), Point((0, 0)))
        self.assertFalse(ls.is_cw(Point((0, 10)))) # At positive boundary beyond segment
        self.assertFalse(ls.is_cw(Point((0, -3)))) # On segment
        self.assertFalse(ls.is_cw(Point((0, -10)))) # At negative boundary beyond segment
        self.assertFalse(ls.is_cw(Point((0, -5)))) # Endpoint of segment
        self.assertFalse(ls.is_cw(Point((0, 0)))) # Endpoint of segment

    def test_is_cw3(self):
        """
        Test corner cases for non-axis-aligned segment not through origin.

        Test tag: <tc>#tests#LineSegment.is_cw</tc>
        """
        ls = LineSegment(Point((0, 1)), Point((5, 6)))
        self.assertFalse(ls.is_cw(Point((10, 11)))) # At positive boundary beyond segment
        self.assertFalse(ls.is_cw(Point((3, 4)))) # On segment
        self.assertFalse(ls.is_cw(Point((-10, -9)))) # At negative boundary beyond segment
        self.assertFalse(ls.is_cw(Point((0, 1)))) # Endpoint of segment
        self.assertFalse(ls.is_cw(Point((5, 6)))) # Endpoint of segment

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
        ls = LineSegment(Point((0, 0)), Point((1, 0)))
        self.assertEquals(ls.line.m, 0) 
        self.assertEquals(ls.line.b, 0) 

        ls = LineSegment(Point((0, 0)), Point((0, 1)))
        self.assertEquals(ls.line, None)
        
        ls = LineSegment(Point((0, 0)), Point((0, -1)))
        self.assertEquals(ls.line, None)
   
        ls  = LineSegment(Point((0, 0)), Point((0, 0)))
        self.assertEquals(ls.line, None)
     
class Line:
    """
    Geometric representation of line objects.

    Attributes:
    m -- slope (number)
    b -- y-intercept (number)
    """

    def __init__(self, m, b):
        """
        Returns a Line object.
 
        __init__(number, number) -> Line

        Test tag: <tc>#is#Line.__init__</tc>
        Test tag: <tc>#tests#Line.__init__</tc>
 
        Arguments:
        m -- the slope of the line
        b -- the y-intercept of the line

        Example:
        >>> ls = Line(1, 0)
        >>> ls.m
        1
        >>> ls.b
        0
        """
        if m == 1e600 or m == -1e600:
            raise ArithmeticException, 'Slope cannot be infinite.'
        self.m = m
        self.b = b

    def y(self, x):
        """
        Returns the y-value of the line at a particular x-value.
 
        y(number) -> number

        Arguments:
        x -- the x-value to compute y at

        Example:
        >>> l = Line(1, 0)
        >>> l.y(1)
        1
        """
        if self.m == 0:
            return self.b
        return self.m*x + self.b  

class _TestLine(unittest.TestCase):

    def test___init__1(self):
        """
        Tests a variety of generic cases.

        Test tag: <tc>#tests#Line.__init__</tc>
        """
        for m,b in [(4, 0.0), (-140, 5), (0, 0)]:
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

class Ray:
    """
    Geometric representation of ray objects.

    Attributes:
    o -- origin (point where ray originates)
    p -- second point on the ray (not point where ray originates)
    """

    def __init__(self, origin, second_p):
        """
        Returns a ray with the values specified.
 
        __init__(Point, Point) -> Ray

        Arguments:
        origin -- the point where the ray originates
        second_p -- the second point specifying the ray (not the origin) 

        Example:
        >>> l = Ray(Point((0, 0)), Point((1, 0)))
        >>> str(l.o)
        '(0.0, 0.0)'
        >>> str(l.p)
        '(1.0, 0.0)'
        """
        self.o = origin
        self.p = second_p

class _TestRay(unittest.TestCase):

    def test___init__1(self):
        """
        Tests generic cases.

        <tc>#tests#Ray.__init__</tc>
        """
        r = Ray(Point((0, 0)), Point((1, 1)))
        r = Ray(Point((8, -3)), Point((-5, 9)))

class Chain(object):
    """
    Geometric representation of a chain, also known as a polyline.

    Attributes:
    vertices -- a Point list of the vertices of the chain in order.
    len -- the geometric length of the chain.
    """

    def __init__(self, vertices):
        """
        Returns a chain created from the points specified.
 
        __init__(Point list or list of Point lists) -> Chain

        Arguments:
        vertices -- list -- Point list or list of Point lists.

        Example:
        >>> c = Chain([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((2, 1))])
        """ 
        if isinstance(vertices[0], list):
            self._vertices = [part for part in vertices]
        else:
            self._vertices = [vertices]
        self._reset_props()

    def _reset_props(self):
        """
        HELPER METHOD. DO NOT CALL.

        Resets attributes which are functions of other attributes. The getters for these attributes (implemented as
        properties) then recompute their values if they have been reset since the last call to the getter.

        _reset_props() -> None

        Example:
        >>> ls = Chain([Point((1, 2)), Point((5, 6))])
        >>> ls._reset_props()
        """ 
        self._len = None
        self._bounding_box = None

    @property
    def vertices(self):
        """
        Returns the vertices of the chain in clockwise order.

        vertices -> Point list

        Example:
        >>> c = Chain([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((2, 1))])
        >>> verts = c.vertices
        >>> len(verts) 
        4
        """
        return sum([part for part in self._vertices], [])

    @property
    def parts(self):
        """
        Returns the parts of the chain.
        
        parts -> Point list

        Example:
        >>> c = Chain([[Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))],[Point((2,1)),Point((2,2)),Point((1,2)),Point((1,1))]])
        >>> len(c.parts)
        2
        """
        return [[v for v in part] for part in self._vertices]

    @property
    def bounding_box(self):
        """
        Returns the bounding box of the chain.
 
        bounding_box -> Rectangle 

        Example:
        >>> c = Chain([Point((0, 0)), Point((2, 0)), Point((2, 1)), Point((0, 1))])
        >>> c.bounding_box.left
        0.0
        >>> c.bounding_box.lower
        0.0
        >>> c.bounding_box.right
        2.0
        >>> c.bounding_box.upper
        1.0
        """
        if self._bounding_box == None:
            vertices = self.vertices
            self._bounding_box = Rectangle(min([v[0] for v in vertices]), min([v[1] for v in vertices]),
                                           max([v[0] for v in vertices]), max([v[1] for v in vertices]))
        return self._bounding_box 

    @property
    def len(self):
        """
        Returns the geometric length of the chain. 
 
        len -> number

        Example:
        >>> c = Chain([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((2, 1))])
        >>> c.len
        3.0
        >>> c = Chain([[Point((0, 0)), Point((1, 0)), Point((1, 1))],[Point((10,10)),Point((11,10)),Point((11,11))]])
        >>> c.len
        4.0
        """
        def dist(v1, v2):
            return math.hypot(v1[0] - v2[0], v1[1] - v2[1])

        def part_perimeter(part):
            return sum([dist(part[i], part[i+1]) for i in xrange(len(part)-1)])

        if self._len == None:
            self._len = sum([part_perimeter(part) for part in self._vertices])
        return self._len

class _TestChain(unittest.TestCase):

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

class Polygon(object):
    """
    Geometric representation of polygon objects.

    Attributes:
    vertices -- the vertices of the Polygon in clockwise order
    len -- the number of verticies (not including holes)
    perimeter -- the geometric length of the perimeter of the Polygon
    bounding_box -- the bounding box of the polygon
    area -- the area enclosed by the polygon
    centroid -- the 'center of gravity', i.e. the mean point of the polygon.
    """

    def __init__(self, vertices, holes=None):
        """
        Returns a polygon created from the objects specified.
 
        __init__(Point list or list of Point lists, holes list ) -> Polygon

        Arguments:
        vertices -- list -- a list of vertices or a list of lists of vertices.
        Keyword Arguments:
        holes -- list -- a list of sub-polygons to be considered as holes.

        Example:
        >>> p1 = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        """
        def clockwise(part):
            if standalone.is_clockwise(part):
                return part[:] 
            else:
                return part[::-1]

        if isinstance(vertices[0], list):
            self._vertices = [clockwise(part) for part in vertices]
        else:
            self._vertices = [clockwise(vertices)]
        if holes != None:
            if isinstance(holes[0], list):
                self._holes = [clockwise(hole) for hole in holes]
            else:
                self._holes = [clockwise(holes)]
        else:
            self._holes = [[]] 
        self._reset_props()

    def _reset_props(self):
        self._perimeter = None
        self._bounding_box = None
        self._area = None
        self._centroid = None
        self._len = None
    def __len__(self):
        return self.len

    @property
    def len(self):
        """
        Returns the number of vertices in the polygon. Does not include holes.

        len -> int

        Example:
        >>> p1 = Polygon([Point((0, 0)), Point((0, 1)), Point((1, 1)), Point((1, 0))])
        >>> p1.len
        4
        >>> len(p1)
        4
        """
        if self._len == None:
            self._len = len(self.vertices)
        return self._len
    
    @property
    def vertices(self):
        """
        Returns the vertices of the polygon in clockwise order.

        vertices -> Point list

        Example:
        >>> p1 = Polygon([Point((0, 0)), Point((0, 1)), Point((1, 1)), Point((1, 0))])
        >>> len(p1.vertices)
        4
        """
        return sum([part for part in self._vertices], [])

    @property
    def holes(self):
        """
        Returns the holes of the polygon in clockwise order.
        
        holes -> Point list

        Example:
        >>> p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))], [Point((1, 2)), Point((2, 2)), Point((2, 1)), Point((1, 1))])
        >>> len(p.holes)
        4
        """
        return sum([part for part in self._holes], [])

    @property
    def parts(self):
        """
        Returns the parts of the polygon in clockwise order.
        
        parts -> Point list

        Example:
        >>> p = Polygon([[Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))], [Point((2,1)),Point((2,2)),Point((1,2)),Point((1,1))]])
        >>> len(p.parts)
        2
        """
        return [[v for v in part] for part in self._vertices]

    @property
    def perimeter(self):
        """
        Returns the perimeter of the polygon.
 
        perimeter() -> number

        Example:
        >>> p = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        >>> p.perimeter
        4.0
        """
        def dist(v1, v2):
            return math.hypot(v1[0] - v2[0], v1[1] - v2[1])

        def part_perimeter(part):
            return sum([dist(part[i], part[i+1]) for i in xrange(-1, len(part)-1)])

        if self._perimeter == None:
            self._perimeter = (sum([part_perimeter(part) for part in self._vertices]) + 
                               sum([part_perimeter(hole) for hole in self._holes])) 
        return self._perimeter

    @property
    def bounding_box(self):
        """
        Returns the bounding box of the polygon.
 
        bounding_box -> Rectangle 

        Example:
        >>> p = Polygon([Point((0, 0)), Point((2, 0)), Point((2, 1)), Point((0, 1))])
        >>> p.bounding_box.left
        0.0
        >>> p.bounding_box.lower
        0.0
        >>> p.bounding_box.right
        2.0
        >>> p.bounding_box.upper
        1.0
        """
        if self._bounding_box == None:
            vertices = self.vertices
            self._bounding_box = Rectangle(min([v[0] for v in vertices]), min([v[1] for v in vertices]),
                                           max([v[0] for v in vertices]), max([v[1] for v in vertices]))
        return self._bounding_box 

    @property
    def area(self):
        """
        Returns the area of the polygon.
 
        area -> number

        Example:
        >>> p = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        >>> p.area
        1.0
        >>> p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],[Point((2,1)),Point((2,2)),Point((1,2)),Point((1,1))])
        >>> p.area
        99.0
        """
        def part_area(part_verts):
            area = 0
            for i in xrange(-1, len(part_verts)-1):
                area += (part_verts[i][0] + part_verts[i+1][0])*(part_verts[i][1] - part_verts[i+1][1])
            area = area*0.5
            if area < 0:
                area = -area
            return area

        return (sum([part_area(part) for part in self._vertices]) -
                sum([part_area(hole) for hole in self._holes]))

    @property
    def centroid(self):
        """
        Returns the centroid of the polygon.
 
        centroid -> Point

        Example:
        >>> p = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        >>> cent = p.centroid
        >>> str(cent)
        '(0.5, 0.5)'
        """
        def part_area(part_verts):
            area = 0
            for i in xrange(-1, len(part_verts)-1):
                area = area + (part_verts[i][0] + part_verts[i+1][0])*(part_verts[i][1] - part_verts[i+1][1])
            area = area*0.5
            if area < 0:
                area = -area
            return area

        def part_centroid_area(vertices):
            # Return a 2-tuple of (centroid, area) for the part
            area = part_area(vertices)
            x_center = sum([(vertices[i][0] + vertices[i-1][0])*
                           (vertices[i][0]*vertices[i-1][1] - vertices[i-1][0]*vertices[i][1])
                               for i in xrange(0, len(vertices))])/(6*area)
            y_center = sum([(vertices[i][1] + vertices[i-1][1])*
                           (vertices[i][0]*vertices[i-1][1] - vertices[i-1][0]*vertices[i][1])
                               for i in xrange(0, len(vertices))])/(6*area)
            return ((x_center, y_center), area)

        if self._holes != [[]]:
            raise NotImplementedError, 'Cannot compute centroid for polygon with holes'
        part_centroids_areas = [part_centroid_area(vertices) for vertices in self._vertices]
        tot_area = sum([a for c,a in part_centroids_areas])
        return (sum([c[0]*(a/tot_area) for c,a in part_centroids_areas]),
                sum([c[1]*(a/tot_area) for c,a in part_centroids_areas]))

class _TestPolygon(unittest.TestCase):

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
        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))]) 
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]]) 
        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))]) 
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]], 
                    holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))]) 
        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                           [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]]) 
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]], 
                    holes=[[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                           [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]]) 

    def test_area1(self):
        """
        Test multiple parts.

        Test tag: <tc>#tests#Polygon.area</tc>
        """
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]]) 
        self.assertEquals(p.area, 200) 

    def test_area2(self):
        """
        Test holes.

        Test tag: <tc>#tests#Polygon.area</tc>
        """
        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))]) 
        self.assertEquals(p.area, 100 - 4)

        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                           [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]]) 
        self.assertEquals(p.area, 100 - (4 + 4))

        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]], 
                    holes=[[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                           [Point((36, 36)), Point((36, 38)), Point((38, 38)), Point((38, 36))]]) 
        self.assertEquals(p.area, 200 - (4 + 4))

    def test_area4(self):
        """
        Test polygons with vertices in both orders (cw, ccw).

        Test tag: <tc>#tests#Polygon.area</tc>
        """
        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))]) 
        self.assertEquals(p.area, 100)

        p = Polygon([Point((0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))]) 
        self.assertEquals(p.area, 100)

    def test_bounding_box1(self):
        """
        Test polygons with multiple parts.

        Test tag: <tc>#tests#Polygon.bounding_box</tc>
        """
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
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
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]]) 
        c = p.centroid
        self.assertEquals(c[0], 20)
        self.assertEquals(c[1], 20)

    def test_centroid2(self):
        """
        Test polygons with multiple parts of different size.

        Test tag: <tc>#tests#Polygon.centroid</tc>
        """ 
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((35, 30)), Point((35, 35)), Point((30, 35))]]) 
        c = p.centroid
        self.assertEquals(c[0], 10.5)
        self.assertEquals(c[1], 10.5)

    def test_holes1(self):
        """
        Test for correct vertex values/order.

        Test tag: <tc>#tests#Polygon.holes</tc>
        """
        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))]) 
        self.assertEquals(len(p.holes), 4)
        e_holes = [Point((2, 2)), Point((2, 4)), Point((4, 4)), Point((4, 2))] 
        self.assertTrue(p.holes in [e_holes, [e_holes[-1]] + e_holes[:3], e_holes[-2:] + e_holes[:2], e_holes[-3:] + [e_holes[0]]])

    def test_holes2(self):
        """
        Test for multiple holes.

        Test tag: <tc>#tests#Polygon.holes</tc>
        """
        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                    holes=[[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                           [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]]) 
        holes = p.holes
        self.assertEquals(len(holes), 8)

    def test_parts1(self):
        """
        Test for correct vertex values/order.

        Test tag: <tc>#tests#Polygon.parts</tc>
        """
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((30, 40))]]) 
        self.assertEquals(len(p.parts), 2)

        part1 = [Point((0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))]
        part2 = [Point((30, 30)), Point((30, 40)), Point((40, 30))] 
        if len(p.parts[0]) == 4:
            self.assertTrue(p.parts[0] in [part1, part1[-1:] + part1[:3], part1[-2:] + part1[:2], part1[-3:] + part1[:1]])
            self.assertTrue(p.parts[1] in [part2, part2[-1:] + part2[:2], part2[-2:] + part2[:1]])
        elif len(p.parts[0]) == 3:
            self.assertTrue(p.parts[0] in [part2, part2[-1:] + part2[:2], part2[-2:] + part2[:1]])
            self.assertTrue(p.parts[1] in [part1, part1[-1:] + part1[:3], part1[-2:] + part1[:2], part1[-3:] + part1[:1]])
        else:
            self.fail()

    def test_perimeter1(self):
        """
        Test with multiple parts.

        Test tag: <tc>#tests#Polygon.perimeter</tc>
        """
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]]) 
        self.assertEquals(p.perimeter, 80)    

    def test_perimeter2(self):
        """
        Test with holes.

        Test tag: <tc>#tests#Polygon.perimeter</tc>
        """
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]], 
                    holes=[[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                           [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))]]) 
        self.assertEquals(p.perimeter, 80 + 16)    

    def test_vertices1(self):
        """
        Test for correct values/order of vertices.
 
        Test tag: <tc>#tests#Polygon.vertices</tc>
        """
        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))])
        self.assertEquals(len(p.vertices), 4)
        e_verts = [Point((0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))] 
        self.assertTrue(p.vertices in [e_verts, e_verts[-1:] + e_verts[:3], e_verts[-2:] + e_verts[:2], e_verts[-3:] + e_verts[:1]])
        
    def test_vertices2(self):
        """
        Test for multiple parts.
 
        Test tag: <tc>#tests#Polygon.vertices</tc>
        """
        p = Polygon([[Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                     [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))]]) 
        self.assertEquals(len(p.vertices), 8)
     

class Rectangle:
    """
    Geometric representation of rectangle objects.

    Attributes:
    left -- the minimum x-value of the rectangle
    lower -- the minimum y-value of the rectangle
    right -- the maximum x-value of the rectangle
    upper -- the maximum y-value of the rectangle
    """

    def __init__(self, left, lower, right, upper):
        """
        Returns a Rectangle object.
 
        __init__(number, number, number, number) -> Rectangle

        Arguments:
        left -- the minimum x-value of the rectangle
        lower -- the minimum y-value of the rectangle
        right -- the maximum x-value of the rectangle
        upper -- the maximum y-value of the rectangle

        Example:
        >>> r = Rectangle(-4, 3, 10, 17)
        >>> r.left #minx
        -4.0
        >>> r.lower #miny
        3.0
        >>> r.right #maxx
        10.0
        >>> r.upper #maxy
        17.0
        """
        if right < left or upper < lower:
            raise ArithmeticError, 'Rectangle must have positive area.'
        self.left = float(left)
        self.lower = float(lower)
        self.right = float(right)
        self.upper = float(upper)

    def __getitem__(self,key):
        """
        >>> r = Rectangle(-4, 3, 10, 17)
        >>> r[:]
        [-4.0, 3.0, 10.0, 17.0]
        """
        l = [self.left,self.lower,self.right,self.upper]
        return l.__getitem__(key)

    def set_centroid(self, new_center):
        """
        Moves the rectangle center to a new specified point.
 
        set_centroid(Point) -> Point

        Arguments:
        new_center -- the new location of the centroid of the polygon

        Example:
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.set_centroid(Point((4, 4)))
        >>> r.left
        2.0
        >>> r.right
        6.0
        >>> r.lower
        2.0
        >>> r.upper
        6.0
        """
        shift = (new_center[0] - (self.left + self.right)/2, new_center[1] - (self.lower + self.upper)/2)
        self.left = self.left + shift[0]
        self.right = self.right + shift[0]
        self.lower = self.lower + shift[1]
        self.upper = self.upper + shift[1]

    def set_scale(self, scale):
        """
        Rescales the rectangle around its center.  
 
        set_scale(number) -> number

        Arguments:
        scale -- the ratio of the new scale to the old scale (e.g. 1.0 is current size)

        Example:
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.set_scale(2)
        >>> r.left
        -2.0
        >>> r.right
        6.0
        >>> r.lower
        -2.0
        >>> r.upper
        6.0
        """
        center = ((self.left + self.right)/2, (self.lower + self.upper)/2)
        self.left = center[0] + scale*(self.left - center[0])
        self.right = center[0] + scale*(self.right - center[0])
        self.lower = center[1] + scale*(self.lower - center[1])
        self.upper = center[1] + scale*(self.upper - center[1])

    @property
    def area(self):
        """
        Returns the area of the Rectangle. 
 
        area -> number

        Example:
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.area
        16.0
        """
        return (self.right - self.left)*(self.upper - self.lower)

    @property
    def width(self):
        """
        Returns the width of the Rectangle. 
 
        width -> number

        Example:
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.width
        4.0
        """
        return self.right - self.left

    @property
    def height(self):
        """
        Returns the height of the Rectangle. 
 
        height -> number

        Example:
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.height
        4.0
        """
        return self.upper - self.lower

class _TestRectangle(unittest.TestCase):

    def test___init__1(self):
        """
        Test exceptions are thrown correctly.

        Test tag: <tc>#tests#Rectangle.__init__</tc>
        """
        try:
            r = Rectangle(1, 1, -1, 5) # right < left
        except ArithmeticError:
            pass
        else:
            self.fail()

        try:
            r = Rectangle(1, 1, 5, -1) # upper < lower
        except ArithmeticError:
            pass
        else:
            self.fail()

    def test_set_centroid1(self):
        """
        Test with rectangles of zero width or height.

        Test tag: <tc>#tests#Rectangle.set_centroid</tc>
        """
        r = Rectangle(5, 5, 5, 10) # Zero width
        r.set_centroid(Point((0, 0)))
        self.assertEquals(r.left, 0)
        self.assertEquals(r.lower, -2.5)
        self.assertEquals(r.right, 0)
        self.assertEquals(r.upper, 2.5)

        r = Rectangle(10, 5, 20, 5) # Zero height
        r.set_centroid(Point((40, 40)))
        self.assertEquals(r.left, 35)
        self.assertEquals(r.lower, 40)
        self.assertEquals(r.right, 45)
        self.assertEquals(r.upper, 40)
        
        r = Rectangle(0, 0, 0, 0) # Zero width and height
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
        r = Rectangle(5, 5, 5, 10) # Zero width
        r.set_scale(2) 
        self.assertEquals(r.left, 5)
        self.assertEquals(r.lower, 2.5)
        self.assertEquals(r.right, 5)
        self.assertEquals(r.upper, 12.5)

        r = Rectangle(10, 5, 20, 5) # Zero height
        r.set_scale(2) 
        self.assertEquals(r.left, 5)
        self.assertEquals(r.lower, 5)
        self.assertEquals(r.right, 25)
        self.assertEquals(r.upper, 5)
        
        r = Rectangle(0, 0, 0, 0) # Zero width and height
        r.set_scale(100)
        self.assertEquals(r.left, 0)
        self.assertEquals(r.lower, 0)
        self.assertEquals(r.right, 0)
        self.assertEquals(r.upper, 0)

        r = Rectangle(0, 0, 0, 0) # Zero width and height
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
        r = Rectangle(5, 5, 5, 10) # Zero width
        self.assertEquals(r.area, 0)

        r = Rectangle(10, 5, 20, 5) # Zero height
        self.assertEquals(r.area, 0)
        
        r = Rectangle(0, 0, 0, 0) # Zero width and height
        self.assertEquals(r.area, 0)

    def test_height1(self):
        """
        Test rectangles with zero height.

        Test tag: <tc>#tests#Rectangle.height</tc>
        """
        r = Rectangle(10, 5, 20, 5) # Zero height
        self.assertEquals(r.height, 0)

    def test_width1(self):
        """
        Test rectangles with zero width.

        Test tag: <tc>#tests#Rectangle.width</tc>
        """
        r = Rectangle(5, 5, 5, 10) # Zero width
        self.assertEquals(r.width, 0)

def _test():
    import doctest
    doctest.testmod()
    unittest.main()

if __name__ == '__main__':
    _test()
