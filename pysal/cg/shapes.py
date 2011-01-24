"""
Computational geometry code for PySAL: Python Spatial Analysis Library.

"""

__author__  = "Sergio J. Rey, Xinyue Ye, Charles Schmidt, Andrew Winslow"
__credits__ = "Copyright (c) 2005-2009 Sergio J. Rey"

import doctest
import math

__all__ = ['Point', 'LineSegment', 'Line', 'Ray', 'Chain', 'Polygon', 'Rectangle']

class Point(object):
    """
    Geometric class for point objects.

    Attributes
    ----------
    None
    """
    def __init__(self, loc):
        """
        Returns an instance of a Point object.

        __init__((number, number)) -> Point 

        Test tag: <tc>#is#Point.__init__</tc>    
        Test tag: <tc>#tests#Point.__init__</tc>    
 
        Parameters
        ----------
        loc : tuple location (number x-tuple, x > 1) 

        Attributes
        ----------

        Examples
        --------
        >>> p = Point((1, 3)) 
        """
        self.__loc = tuple(map(float, loc))

    def __eq__(self,other):
        """
        Tests if the Point is equal to another object.

        __eq__(x) -> bool

        Parameters
        ----------
        other : an object to test equality against

        Attributes
        ----------

        Examples
        --------
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

        Parameters
        ----------
        other : an object to test equality against

        Attributes
        ----------

        Examples
        --------
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

        Parameters
        ----------
        None

        Attributes
        ----------

        Examples
        --------
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

        Parameters
        ----------
        i : index of the desired dimension.

        Attributes
        ----------

        Examples
        --------
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

        Parameters
        ----------
        i : index to start slice
        j : index to end slice (excluded).

        Attributes
        ----------

        Examples
        --------
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

        Parameters
        ----------
        None

        Attributes
        ----------

        Examples
        --------
        >>> len(Point((1,2)))
        2
        """
        return len(self.__loc)

    def __repr__(self):
        """
        Returns the string representation of the Point

        __repr__() -> string
        
        Parameters
        ----------
        None

        Attributes
        ----------

        Examples
        --------
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
 
        Attributes
        ----------

        Examples
        --------
        >>> p = Point((1, 3))
        >>> str(p)
        '(1.0, 3.0)'
        """
        return str(self.__loc)

class LineSegment:
    """
    Geometric representation of line segment objects.
    ...

    Parameters
    ----------

    start_pt     : Point 
                   Point where segment begins
    end_pt       : Point 
                   Point where segment ends

    Attributes
    ----------

    p1              : Point
                      Starting point
    p2              : Point
                      Ending point
    bounding_box    : tuple
                      The bounding box of the segment (number 4-tuple)
    len             : float
                      The length of the segment
    line            : Line
                      The line on which the segment lies

    """
 
    def __init__(self, start_pt, end_pt):
        """
        Creates a LineSegment object.
 
        __init__(Point, Point) -> LineSegment

        Test tag: <tc>#is#LineSegment.__init__</tc>
        Test tag: <tc>#tests#LineSegment.__init__</tc>


        Attributes
        ----------

        Examples
        --------
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        """
        self._p1 = start_pt
        self._p2 = end_pt
        self._reset_props()
    def __eq__(self,other):
        """
        Returns true if self and other are the same line segment

        Examples
        ________
        >>> l1 = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> l2 = LineSegment(Point((5, 6)), Point((1, 2)))
        >>> l1 == l2
        True
        >>> l2 == l1
        True
        """
        if not isinstance(other,self.__class__):
            return False
        if (other.p1 == self._p1 and other.p2 == self._p2):
            return True
        elif (other.p2 == self._p1 and other.p1 == self._p2):
            return True
        return False
    def _reset_props(self):
        """
        HELPER METHOD. DO NOT CALL.

        Resets attributes which are functions of other attributes. The getters for these attributes (implemented as
        properties) then recompute their values if they have been reset since the last call to the getter.

        _reset_props() -> None

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Parameters
        ----------
        pt : point lying ccw or cw of a segment
 
        Attributes
        ----------

        Examples
        --------
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

        Parameters
        ----------
        pt : point lying ccw or cw of a segment
 
        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

class Line:
    """
    Geometric representation of line objects.

    Attributes
    ----------
    m       : float
              slope
    b       : float
              y-intercept

    """

    def __init__(self, m, b):
        """
        Returns a Line object.
 
        __init__(number, number) -> Line

        Test tag: <tc>#is#Line.__init__</tc>
        Test tag: <tc>#tests#Line.__init__</tc>
 
        Parameters
        ----------
        m : the slope of the line
        b : the y-intercept of the line

        Attributes
        ----------

        Examples
        --------
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

        Parameters
        ----------
        x : the x-value to compute y at

        Attributes
        ----------

        Examples
        --------
        >>> l = Line(1, 0)
        >>> l.y(1)
        1
        """
        if self.m == 0:
            return self.b
        return self.m*x + self.b  

class Ray:
    """
    Geometric representation of ray objects.

    Attributes
    ----------

    o       : Point
              Origin (point where ray originates)
    p       : Point
              Second point on the ray (not point where ray originates)
    """

    def __init__(self, origin, second_p):
        """
        Returns a ray with the values specified.
 
        __init__(Point, Point) -> Ray

        Parameters
        ----------
        origin   : the point where the ray originates
        second_p : the second point specifying the ray (not the origin) 

        Attributes
        ----------

        Examples
        --------
        >>> l = Ray(Point((0, 0)), Point((1, 0)))
        >>> str(l.o)
        '(0.0, 0.0)'
        >>> str(l.p)
        '(1.0, 0.0)'
        """
        self.o = origin
        self.p = second_p


class Chain(object):
    """
    Geometric representation of a chain, also known as a polyline.

    Attributes
    ----------

    vertices    : list
                  List of Points of the vertices of the chain in order.
    len         : float
                  The geometric length of the chain.

    """

    def __init__(self, vertices):
        """
        Returns a chain created from the points specified.
 
        __init__(Point list or list of Point lists) -> Chain

        Parameters
        ----------
        vertices : list -- Point list or list of Point lists.

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

class Polygon(object):
    """
    Geometric representation of polygon objects.

    Attributes
    ----------

    vertices        : list
                      List of Points with the vertices of the Polygon in
                      clockwise order
    len             : int
                      Number of verticies including holes
    perimeter       : float
                      Geometric length of the perimeter of the Polygon
    bounding_box    : Rectangle
                      Bounding box of the polygon
    area            : float
                      Area enclosed by the polygon
    centroid        : tuple
                      The 'center of gravity', i.e. the mean point of the polygon.
    """

    def __init__(self, vertices, holes=None):
        """
        Returns a polygon created from the objects specified.
 
        __init__(Point list or list of Point lists, holes list ) -> Polygon

        Parameters
        ----------
        vertices : list -- a list of vertices or a list of lists of vertices.
        holes    : list -- a list of sub-polygons to be considered as holes.

        Attributes
        ----------

        Examples
        --------
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
        Returns the number of vertices in the polygon.

        len -> int

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
        >>> p1 = Polygon([Point((0, 0)), Point((0, 1)), Point((1, 1)), Point((1, 0))])
        >>> len(p1.vertices)
        4
        """
        return sum([part for part in self._vertices], []) + sum([part for part in self._holes], [])

    @property
    def holes(self):
        """
        Returns the holes of the polygon in clockwise order.
        
        holes -> Point list

        Attributes
        ----------

        Examples
        --------
        >>> p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))], [Point((1, 2)), Point((2, 2)), Point((2, 1)), Point((1, 1))])
        >>> len(p.holes)
        1
        """
        return [[v for v in part] for part in self._holes]

    @property
    def parts(self):
        """
        Returns the parts of the polygon in clockwise order.
        
        parts -> Point list

        Attributes
        ----------

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

class Rectangle:
    """
    Geometric representation of rectangle objects.

    Attributes
    ----------
    
    left    : float
              Minimum x-value of the rectangle
    lower   : float
              Minimum y-value of the rectangle
    right   : float
              Maximum x-value of the rectangle
    upper   : float
              Maximum y-value of the rectangle
    """

    def __init__(self, left, lower, right, upper):
        """
        Returns a Rectangle object.
 
        __init__(number, number, number, number) -> Rectangle

        Parameters
        ----------
        left  : the minimum x-value of the rectangle
        lower : the minimum y-value of the rectangle
        right : the maximum x-value of the rectangle
        upper : the maximum y-value of the rectangle

        Attributes
        ----------

        Examples
        --------
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

    def __eq__(self,other):
        if other:
            return self[:] == other[:]
        return False

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

        Parameters
        ----------
        new_center : the new location of the centroid of the polygon

        Attributes
        ----------

        Examples
        --------
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

        Parameters
        ----------
        scale : the ratio of the new scale to the old scale (e.g. 1.0 is current size)

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Attributes
        ----------

        Examples
        --------
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

        Examples
        --------
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.height
        4.0
        """
        return self.upper - self.lower

def _test():
    doctest.testmod(verbose=True)

import standalone   #moving this to top breaks unit tests !

if __name__ == '__main__':
    _test()
