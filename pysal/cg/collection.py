"""
Computational geometry code for PySAL: Python Spatial Analysis Library.

Authors:
Sergio Rey <srey@asu.edu>
Xinyue Ye <xinyue.ye@gmail.com>
Charles Schmidt <schmidtc@gmail.com>
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

class GeometryCollection:
    """
    Represents a collection of geometric objects. Useful for performing various
    queries on the collection as a whole (see attributes/methods).
    """

    def __init__(self, objs):
        """
        Creates a GeometryCollection of a set of objects.

        __init__(list of x) -> GeometryCollection

        objs -- a list of geometric objects (from pysal.cg.shapes).

        Examples
        --------
        >>> import pysal.cg.shapes as shapes
        >>> g = GeometryCollection([shapes.Point((0, 0)), shapes.LineSegment(Point((1, 4)), Point((7, 3)))])
        """
        pass

    @property
    def centroid(self):
        """
        Returns the centroid of the collection.

        centroid -> number n-tuple, n > 1
      
        Examples
        --------
        >>> import pysal.cg.shapes as shapes
        >>> g = GeometryCollection([shapes.Point((0, 0)), shapes.Point((1, 1))])
        >>> g.centroid
        (0.5, 0.5)
        """ 
        pass

    @property
    def _get_bounding_box(self):
        """
        Returns the bounding box of the collection.

        bounding_box -> number 4-tuple
 
        Examples
        --------
        >>> import pysal.cg.shapes as shapes
        >>> g = GeometryCollection([shapes.Point((0, 1)), shapes.Point((4, 5))])
        >>> g.bounding_box
        (0, 1, 4, 5)
        """
        pass 

    def add(self,obj):
        """
        Adds the object _obj_ to the to geometry collection.

        add(Shape obj) -> bool
    
        Examples
        --------
        >>> import pysal.cg.shapes as shapes
        >>> g = GeometryCollection()
        >>> g.add(shapes.Point((0, 1)))
        True
        >>> g.add(shapes.Point((4, 5)))
        True
        """
        pass
    def remove(self,obj):
        """
        Removes the object _obj_ from the geometry collection.

        remove(Shape obj) -> bool
    
        Examples
        --------
        >>> import pysal.cg.shapes as shapes
        >>> g = GeometryCollection([shapes.Point((0, 1)), shapes.Point((4, 5))])
        >>> g.remove(shapes.Point((0, 1)))
        True
        >>> g.remove(shapes.Point((0, 1)))
        False
        """
        pass

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()




