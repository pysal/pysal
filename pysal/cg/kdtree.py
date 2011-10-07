"""
KDTree for PySAL: Python Spatial Analysis Library.

Adds support for Arc Distance to scipy.spatial.KDTree.
"""
import sys
import math
import scipy.spatial
import numpy
from scipy import inf
import sphere

__author__ = "Charles R Schmidt <Charles.R.Schmidt@asu.edu>"
DISTANCE_METRICS = ['Euclidean','Arc']

class Arc_KDTree(scipy.spatial.KDTree):
    def __init__(self, data, radius=1.0, leafsize=10):
        """
        KDTree using Arc Distance instead of Euclidean Distance.

        Returned distances are based on radius.
        For Example, pass in the the radius of earth in miles to get back miles.
        Assumes data are Lng/Lat, does not account for geoids.

        For more information see docs for scipy.spatial.KDTree

        Examples
        --------
        >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
        >>> kd = Arc_KDTree(pts, radius = sphere.RADIUS_EARTH_KM)
        >>> d,i = kd.query((90,0), k=4)
        >>> d
        array([ 10007.54339801,  10007.54339801,  10007.54339801,  10007.54339801])
        >>> circumference = 2*math.pi*sphere.RADIUS_EARTH_KM
        >>> round(d[0],5) == round(circumference/4.0,5)
        True
        """
        self.radius = radius
        scipy.spatial.KDTree.__init__(self, map(sphere.toXYZ, data), leafsize)
    def _toXYZ(self, x):
        if not issubclass(type(x),numpy.ndarray):
            x = numpy.array(x)
        if len(x.shape) == 1:
            x = numpy.array(sphere.toXYZ(x))
        else:
            x = map(sphere.toXYZ, x)
        return x
    def count_neighbors(self, other, r):
        """
        See scipy.spatial.KDTree.count_neighbors

        Examples
        --------
        >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
        >>> kd = Arc_KDTree(pts, radius = sphere.RADIUS_EARTH_KM)
        >>> kd.count_neighbors(kd,0)
        4
        >>> circumference = 2.0*math.pi*sphere.RADIUS_EARTH_KM
        >>> kd.count_neighbors(kd,circumference/2.0)
        16
        """
        r = sphere.arcdist2linear(r,self.radius)
        return scipy.spatial.KDTree.count_neighbors(self, other, r)
    def query(self, x, k=1, eps=0, distance_upper_bound=inf):
        """
        See scipy.spatial.KDTree.query

        Parameters
        ----------
        x : array-like, last dimension self.m
            query points are lng/lat.

        Examples
        --------
        >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
        >>> kd = Arc_KDTree(pts, radius = sphere.RADIUS_EARTH_KM)
        >>> d,i = kd.query((90,0), k=4)
        >>> d
        array([ 10007.54339801,  10007.54339801,  10007.54339801,  10007.54339801])
        >>> circumference = 2*math.pi*sphere.RADIUS_EARTH_KM
        >>> round(d[0],5) == round(circumference/4.0,5)
        True
        """
        d,i = scipy.spatial.KDTree.query(self, self._toXYZ(x), k, eps=eps, distance_upper_bound = distance_upper_bound)
        dims = len(d.shape)
        r = self.radius
        if dims == 1:
            #TODO: implement linear2arcdist on numpy arrays
            d = [sphere.linear2arcdist(x,r) for x in d]
        elif dims == 2:
            d = [[sphere.linear2arcdist(x,r) for x in row] for row in d]
        return numpy.array(d),i
    def query_ball_point(self, x, r, eps=0):
        """
        See scipy.spatial.KDTree.query_ball_point

        Examples
        --------
        >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
        >>> kd = Arc_KDTree(pts, radius = sphere.RADIUS_EARTH_KM)
        >>> circumference = 2*math.pi*sphere.RADIUS_EARTH_KM
        >>> kd.query_ball_point(pts, circumference/4., eps=0.1)
        """
        r = sphere.arcdist2linear(r,self.radius)
        eps = sphere.arcdist2linear(eps,self.radius)
        return scipy.spatial.KDTree.query_ball_point(self, self._toXYZ(x), r, eps=eps)
        
    

def KDTree(data, distance_metric='Euclidean', leafsize=10, radius=1.0):
    if distance_metric == 'Euclidean':
        return scipy.spatial.KDTree(data, leafsize)
    elif distance_metric == 'Arc':
        return Arc_KDTree(data, radius, leafsize)

if __name__=='__main__':
    import doctest
    doctest.testmod()
