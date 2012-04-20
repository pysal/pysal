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

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
DISTANCE_METRICS = ['Euclidean','Arc']
FLOAT_EPS = numpy.finfo(float).eps

class Arc_KDTree(scipy.spatial.KDTree):
    def __init__(self, data, leafsize=10, radius=1.0):
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
        self.circumference = 2*math.pi*radius
        scipy.spatial.KDTree.__init__(self, map(sphere.toXYZ, data), leafsize)
    def _toXYZ(self, x):
        if not issubclass(type(x),numpy.ndarray):
            x = numpy.array(x)
        if len(x.shape) ==2 and x.shape[1] == 3: #assume point is already in XYZ
            return x
        if len(x.shape) == 1:
            x = numpy.array(sphere.toXYZ(x))
        else:
            x = map(sphere.toXYZ, x)
        return x
    def count_neighbors(self, other, r, p=2):
        """
        See scipy.spatial.KDTree.count_neighbors

        Parameters
        ----------
        p: ignored, kept to maintain compatibility with scipy.spatial.KDTree

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
        if r > 0.5*self.circumference:
            raise ValueError, "r, must not exceed 1/2 circumference of the sphere (%f)."%self.circumference*0.5
        r = sphere.arcdist2linear(r,self.radius)
        return scipy.spatial.KDTree.count_neighbors(self, other, r)
    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=inf):
        """
        See scipy.spatial.KDTree.query

        Parameters
        ----------
        x : array-like, last dimension self.m
            query points are lng/lat.
        p: ignored, kept to maintain compatibility with scipy.spatial.KDTree

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
        >>> d,i = kd.query(kd.data, k=3)
        >>> d2,i2 = kd.query(pts, k=3)
        >>> (d == d2).all()
        True
        >>> (i == i2).all()
        True
        """
        eps = sphere.arcdist2linear(eps,self.radius)
        if distance_upper_bound != inf:
            distance_upper_bound = sphere.arcdist2linear(distance_upper_bound, self.radius)
        d,i = scipy.spatial.KDTree.query(self, self._toXYZ(x), k, eps=eps, distance_upper_bound = distance_upper_bound)
        dims = len(d.shape)
        r = self.radius
        if dims == 0:
            return sphere.linear2arcdist(d,r),i
        if dims == 1:
            #TODO: implement linear2arcdist on numpy arrays
            d = [sphere.linear2arcdist(x,r) for x in d]
        elif dims == 2:
            d = [[sphere.linear2arcdist(x,r) for x in row] for row in d]
        return numpy.array(d),i
    def query_ball_point(self, x, r, p=2, eps=0):
        """
        See scipy.spatial.KDTree.query_ball_point

        Parameters
        ----------
        p: ignored, kept to maintain compatibility with scipy.spatial.KDTree

        Examples
        --------
        >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
        >>> kd = Arc_KDTree(pts, radius = sphere.RADIUS_EARTH_KM)
        >>> circumference = 2*math.pi*sphere.RADIUS_EARTH_KM
        >>> kd.query_ball_point(pts, circumference/4.)
        array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=object)
        >>> kd.query_ball_point(pts, circumference/2.)
        array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=object)
        """
        eps = sphere.arcdist2linear(eps,self.radius)
        #scipy.sphere.KDTree.query_ball_point appears to ignore the eps argument.
        # we have some floating point errors moving back and forth between cordinate systems,
        # so we'll account for that be adding some to our radius, 3*float's eps value.
        if r > 0.5*self.circumference:
            raise ValueError, "r, must not exceed 1/2 circumference of the sphere (%f)."%self.circumference*0.5
        r = sphere.arcdist2linear(r,self.radius)+FLOAT_EPS*3
        return scipy.spatial.KDTree.query_ball_point(self, self._toXYZ(x), r, eps=eps)
    def query_ball_tree(self, other, r, p=2, eps=0):
        """
        See scipy.spatial.KDTree.query_ball_tree

        Parameters
        ----------
        p: ignored, kept to maintain compatibility with scipy.spatial.KDTree

        Examples
        --------
        >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
        >>> kd = Arc_KDTree(pts, radius = sphere.RADIUS_EARTH_KM)
        >>> kd.query_ball_tree(kd, kd.circumference/4.)
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        >>> kd.query_ball_tree(kd, kd.circumference/2.)
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        """
        eps = sphere.arcdist2linear(eps,self.radius)
        #scipy.sphere.KDTree.query_ball_point appears to ignore the eps argument.
        # we have some floating point errors moving back and forth between cordinate systems,
        # so we'll account for that be adding some to our radius, 3*float's eps value.
        if self.radius != other.radius:
            raise ValueError, "Both trees must have the same radius."
        if r > 0.5*self.circumference:
            raise ValueError, "r, must not exceed 1/2 circumference of the sphere (%f)."%self.circumference*0.5
        r = sphere.arcdist2linear(r,self.radius)+FLOAT_EPS*3
        return scipy.spatial.KDTree.query_ball_tree(self, other, r, eps=eps)
    def query_pairs(self, r, p=2, eps=0):
        """
        See scipy.spatial.KDTree.query_pairs

        Parameters
        ----------
        p: ignored, kept to maintain compatibility with scipy.spatial.KDTree

        Examples
        --------
        >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
        >>> kd = Arc_KDTree(pts, radius = sphere.RADIUS_EARTH_KM)
        >>> kd.query_pairs(kd.circumference/4.)
        set([(0, 1), (1, 3), (2, 3), (0, 2)])
        >>> kd.query_pairs(kd.circumference/2.)
        set([(0, 1), (1, 2), (1, 3), (2, 3), (0, 3), (0, 2)])
        """
        if r > 0.5*self.circumference:
            raise ValueError, "r, must not exceed 1/2 circumference of the sphere (%f)."%self.circumference*0.5
        r = sphere.arcdist2linear(r,self.radius)+FLOAT_EPS*3
        return scipy.spatial.KDTree.query_pairs(self, r, eps=eps)
    def sparse_distance_matrix(self, other, max_distance, p=2):
        """
        See scipy.spatial.KDTree.sparse_distance_matrix

        Parameters
        ----------
        p: ignored, kept to maintain compatibility with scipy.spatial.KDTree

        Examples
        --------
        >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
        >>> kd = Arc_KDTree(pts, radius = sphere.RADIUS_EARTH_KM)
        >>> kd.sparse_distance_matrix(kd, kd.circumference/4.).todense()
        matrix([[     0.        ,  10007.54339801,  10007.54339801,      0.        ],
                [ 10007.54339801,      0.        ,      0.        ,  10007.54339801],
                [ 10007.54339801,      0.        ,      0.        ,  10007.54339801],
                [     0.        ,  10007.54339801,  10007.54339801,      0.        ]])
        >>> kd.sparse_distance_matrix(kd, kd.circumference/2.).todense()
        matrix([[     0.        ,  10007.54339801,  10007.54339801,  20015.08679602],
                [ 10007.54339801,      0.        ,  20015.08679602,  10007.54339801],
                [ 10007.54339801,  20015.08679602,      0.        ,  10007.54339801],
                [ 20015.08679602,  10007.54339801,  10007.54339801,      0.        ]])
        """
        if self.radius != other.radius:
            raise ValueError, "Both trees must have the same radius."
        if max_distance > 0.5*self.circumference:
            raise ValueError, "max_distance, must not exceed 1/2 circumference of the sphere (%f)."%self.circumference*0.5
        max_distance = sphere.arcdist2linear(max_distance,self.radius)+FLOAT_EPS*3
        D = scipy.spatial.KDTree.sparse_distance_matrix(self, other, max_distance)
        D = D.tocoo()
        #print D.data
        a2l = lambda x: sphere.linear2arcdist(x,self.radius)
        #print map(a2l,D.data)
        return scipy.sparse.coo_matrix( (map(a2l,D.data), (D.row,D.col))).todok()
    

def KDTree(data, leafsize=10, distance_metric='Euclidean', radius=1.0):
    if distance_metric == 'Euclidean':
        return scipy.spatial.KDTree(data, leafsize)
    elif distance_metric == 'Arc':
        return Arc_KDTree(data, leafsize, radius)

if __name__=='__main__':
    import doctest
    doctest.testmod()
