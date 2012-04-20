import pysal

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"

class WorldToViewTransform(object):
    """
    An abstract class modeling a View window.
    Supports Panning, Zooming, Resizing.

    Is observable.

    Parameters:
    worldExtent -- Extent,List -- Extent of the world, left,lower,right,upper in world coords
    pixel_width -- int -- intial width of the view in pixels
    pixel_height -- int -- intial height of the view in pixels

    Notes:
    World coordinates are expected to increase the X and Y direction.
    Pixel coordinates are inverted in the Y direction.

    This class helps tranform world coordinates to screen coordinates.
    To transform a GraphicsMatrix,
    matrix.Scale(1.0/model.scale,-1.0/model.scale)
    matrix.Translate(*model.offset)
    
    The transforms will be applied in reverse order,
    The coordinates will first be translated to the origin (of the current view).
    The coordinates will then be scaled.

    Eg.
    >>> view = WorldToViewTransform([-180,-90,180,90],500,500)
    """
    def __init__(self,worldExtent,pixel_width,pixel_height):
        """ Intialize the view to the extent of the world """
        self.__pixel_width = float(pixel_width)
        self.__pixel_height = float(pixel_height)
        self.__world = worldExtent
        self.extent = worldExtent
        # In World Coords
    def __copy__(self):
        return WorldToViewTransform(self.extent,self.__pixel_width,self.__pixel_height)
    copy = __copy__
    def __get_offset(self):
        """ 
        Returns the offset of the top left corner of the current view in world coords.
        Move the world this many units to aling it with the view.
        """
        return self.__offset
    def __set_offset(self,value):
        """
        Set the Offset of the top left corner in world coords.
        """
        assert len(value) == 2
        self.__offset = value
    offset = property(fget=__get_offset,fset=__set_offset)
    def __get_scale(self):
        """ Returns the current scale in units/pixel """
        return self.__scale
    def __set_scale(self,value):
        """ Sets the current scale in units/pixel """
        self.__scale = value
    scale = property(fget=__get_scale,fset=__set_scale)
    def __get_extent(self):
        """Returns the extent of the current view in World Coordinates."""
        left,upper = self.pixel_to_world(0,0)
        right,lower = self.pixel_to_world(self.__pixel_width,self.__pixel_height)
        return pysal.cg.Rectangle(left,lower,right,upper)
    def __set_extent(self,value):
        """ Set the extent of the current view in World Coordinates.
            Preserve fixed scale, take the max of (sx,sy).

            Use this to zoom to a sepcific region when you know the region's 
            bbox in world coords.
        """
        left,lower,right,upper = value
        width = abs(right-left)
        height = abs(upper-lower)
        sx = width/self.__pixel_width
        sy = height/self.__pixel_height
        self.__scale = max(sx,sy)

        #The offset translate the world to the origin.
        #The X offset + world.left == 0
        #The Y offset + world.upper == 0

        # Move the offset a little, so that the center of the extent is in the center of the view.
        oleft = (left+(width/2.0)) - (self.__pixel_width*self.__scale/2.0)
        oupper = (upper-height/2.0) + (self.__pixel_height*self.__scale/2.0)

        #self.__offset = (-left,-upper) # in world coords
        self.__offset = (-oleft,-oupper) # in world coords
    extent = property(fget=__get_extent,fset=__set_extent)
    def __get_width(self):
        """ Returns the width of the current view in world coords """
        return self.__pixel_width*self.scale
    def __set_width(self, value):
        """
        Sets the width of the current view, value in pixels
        
        Eg.
        >>> view = WorldToViewTransform([0,0,100,100],500,500)
        >>> view.extent[:]
        [0.0, 0.0, 100.0, 100.0]
        >>> view.width = 250
        >>> view.extent[:]
        [0.0, 0.0, 50.0, 100.0]
        """
        if self.__pixel_width != value:
            self.__pixel_width = value
    width = property(fget=__get_width,fset=__set_width)
    def __get_height(self):
        """ Returns the height of the current view in world coords """
        return self.__pixel_height*self.scale
    def __set_height(self, value):
        """
        Sets the height of the current view, value in pixels
        
        Eg.
        >>> view = WorldToViewTransform([0,0,100,100],500,500)
        >>> view.extent[:]
        [0.0, 0.0, 100.0, 100.0]
        >>> view.height = 250
        >>> view.extent[:]
        [0.0, 50.0, 100.0, 100.0]
        """
        if self.__pixel_height != value:
            self.__pixel_height = value
    height = property(fget=__get_height,fset=__set_height)
    def __get_pixel_size(self):
        """
        Set and Return the current size of the view in pixels.
        """
        return self.__pixel_width,self.__pixel_height
    def __set_pixel_size(self,value):
        w,h = value
        if self.__pixel_width != w:
            self.__pixel_width = w
        if self.__pixel_height != h:
            self.__pixel_height = h
    pixel_size = property(fget=__get_pixel_size,fset=__set_pixel_size)

    def pan(self,dpx,dpy):
        """ 
        Pan the view by (dpx,dpy) pixel coordinates.
        
        Positive deltas move the world right and down.
        Negative deltas move the world left and up.

        Eg.
        >>> view = WorldToViewTransform([0,0,100,100],500,500)
        >>> view.pan(500,0)
        >>> view.extent[:]
        [-100.0, 0.0, 0.0, 100.0]
        >>> view.pan(-500,500)
        >>> view.extent[:]
        [0.0, 100.0, 100.0, 200.0]
        >>> view.pan(0,-500)
        >>> view.extent[:]
        [0.0, 0.0, 100.0, 100.0]
        >>> view.pan(490,490)
        >>> view.extent[:]
        [-98.0, 98.0, 2.0, 198.0]
        >>> view.pan(-490,-490)
        >>> view.extent[:]
        [0.0, 0.0, 100.0, 100.0]
        """
        ogx,ogy = self.__offset
        s = self.scale
        self.__offset = ogx+(dpx*s),ogy-(dpy*s)
    def pan_to(self,extent):
        initScale = self.scale
        self.extent = extent
        self.scale = initScale
    def pixel_to_world(self,px,py):
        """
        Returns the world coordinates of the Pixel (px,py).

        Eg.
        >>> view = WorldToViewTransform([0,0,100,100],500,500)
        >>> view.pixel_to_world(0,0)
        (0.0, 100.0)
        >>> view.pixel_to_world(500,500)
        (100.0, 0.0)
        """
        sx = self.scale
        sy = -sx
        ogx,ogy = self.__offset
        return px*sx - ogx, py*sy - ogy
    def world_to_pixel(self,x,y):
        """
        Returns the pixel of the world coordinate (x,y).

        Eg.
        >>> view = WorldToViewTransform([0,0,100,100],500,500)
        >>> view.world_to_pixel(0,0)
        (0.0, 500.0)
        >>> view.world_to_pixel(100,100)
        (500.0, -0.0)
        """
        sx = self.scale
        sy = -sx
        ogx,ogy = self.__offset
        return (x+ogx)/sx, (y+ogy)/sy

if __name__=="__main__":
    import doctest
    doctest.testmod()
    view = WorldToViewTransform([0,0,100,100],500,500)
