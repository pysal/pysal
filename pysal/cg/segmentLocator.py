import math
from pysal.cg.shapes import Rectangle, Point, LineSegment

__all__ = ["SegmentGrid"]

class SegmentGrid(object):
    """
    Notes:
        SegmentGrid is a low level Grid class.
        This class does not maintain a copy of the geometry in the grid.
        It returns only approx. Solutions.
        This Grid should be wrapped by a locator.
    """
    def __init__(self, bounds, resolution):
        """
        Returns a grid with specified properties. 

        __init__(Rectangle, number) -> SegmentGrid 

        Parameters
        ----------
        bounds      : the area for the grid to encompass
        resolution  : the diameter of each bin 

        Examples
        --------
        TODO: complete this doctest
        >>> g = SegmentGrid(Rectangle(0, 0, 10, 10), 1)
        """
        if resolution == 0:
            raise Exception, 'Cannot create grid with resolution 0'
        self.res = resolution
        self.hash = {}
        self.x_range = (bounds.left, bounds.right)
        self.y_range = (bounds.lower, bounds.upper)
        try:
            self.i_range = int(math.ceil((self.x_range[1]-self.x_range[0])/self.res))
            self.j_range = int(math.ceil((self.y_range[1]-self.y_range[0])/self.res))
        except Exception:
            raise Exception, ('Invalid arguments for SegmentGrid(): (' + str(self.x_range) + ', ' + str(self.y_range) + ', ' + str(self.res) + ')')
    def in_grid(self, loc):
        """
        Returns whether a 2-tuple location _loc_ lies inside the grid bounds.
        """
        return (self.x_range[0] <= loc[0] <= self.x_range[1] and
                self.y_range[0] <= loc[1] <= self.y_range[1])
    def __grid_loc(self, loc):
        i = min(self.i_range, max(int((loc[0] - self.x_range[0])/self.res), 0))
        j = min(self.j_range, max(int((loc[1] - self.y_range[0])/self.res), 0))
        return (i, j)
    def bin_loc(self, loc, id):
        grid_loc = self.__grid_loc(loc)
        if grid_loc not in self.hash:
            self.hash[grid_loc] = set()
        self.hash[grid_loc].add(id)
        return grid_loc
        
    def add(self, segment, id):
        """
        Adds segment to the grid.

        add(segment, id) -> bool

        Parameters
        ----------
        id -- id to be stored int he grid.
        segment -- the segment which identifies where to store 'id' in the grid.

        Examples
        --------
        >>> g = SegmentGrid(Rectangle(0, 0, 10, 10), 1)
        >>> g.add(LineSegment(Point((0.2, 0.7)), Point((4.2, 8.7))), 0)
        True
        """
        if not (self.in_grid(segment.p1) and self.in_grid(segment.p2)):
            raise Exception, 'Attempt to insert item at location outside grid bounds: ' + str(segment)
        i,j = self.bin_loc(segment.p1, id)
        I,J = self.bin_loc(segment.p2, id)
        
        bbox = segment.bounding_box
        left = bbox.left
        lower = bbox.lower
        res = self.res
        line = segment.line
        for i in xrange(min(i,I), max(i,I)):
            x = left + (i*res)
            y = line.y(x)
            self.bin_loc((x,y), id)
        for j in xrange(min(j,J), max(j,J)):
            y = lower + (j*res)
            x = line.x(y)
            self.bin_loc((x,y), id)
        return True
    def remove(self, segment):
        pass
    def nearest(self, pt):
        """
        Return the LineSegment
        """
        pass

