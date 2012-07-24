"""
sphere: Tools for working with spherical distances.

Author: Charles R Schmidt <schmidtc@gmail.com>
"""

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
import math
import random
import numpy
import scipy.spatial,scipy.constants
from scipy.spatial.distance import euclidean
from math import pi,cos,sin,asin

__all__ = ['RADIUS_EARTH_KM', 'RADIUS_EARTH_MILES', 'arcdist', 'arcdist2linear', 'brute_knn', 'fast_knn', 'fast_threshold', 'linear2arcdist', 'toLngLat', 'toXYZ']


RADIUS_EARTH_KM = 6371.0 # Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
RADIUS_EARTH_MILES = (RADIUS_EARTH_KM*scipy.constants.kilo) / scipy.constants.mile


def arcdist(pt0,pt1,radius = RADIUS_EARTH_KM):
    """
    Returns the arc distance between pt0 and pt1 using supplied radius
    pt0 and pt1 are assumed to be in the form (lng,lat)
    
    Examples
    >>> pt0 = (0,0)
    >>> pt1 = (180,0)
    >>> d = arcdist(pt0,pt1,RADIUS_EARTH_MILES)
    >>> d == math.pi*RADIUS_EARTH_MILES
    True
    """
    return linear2arcdist(euclidean(toXYZ(pt0),toXYZ(pt1)), radius)

def arcdist2linear(arc_dist,radius = RADIUS_EARTH_KM):
    """
    Convert an arc distance (spherical earth) to a linear distance (R3) in the unit sphere.

    Examples
    >>> pt0 = (0,0)
    >>> pt1 = (180,0)
    >>> d = arcdist(pt0,pt1,RADIUS_EARTH_MILES)
    >>> d == math.pi*RADIUS_EARTH_MILES
    True
    >>> arcdist2linear(d,RADIUS_EARTH_MILES)
    2.0
    """
    c = 2*math.pi*radius
    d = (2-(2*math.cos(math.radians((arc_dist*360.0)/c)))) ** (0.5)
    return d

def linear2arcdist(linear_dist, radius = RADIUS_EARTH_KM):
    """
    Convert a linear distance in the unit sphere (R3) to an arc distance based on supplied radius

    Examples
    >>> pt0 = (0,0)
    >>> pt1 = (180,0)
    >>> d = arcdist(pt0,pt1,RADIUS_EARTH_MILES)
    >>> d == linear2arcdist(2.0, radius = RADIUS_EARTH_MILES)
    True
    """
    if linear_dist == float('inf'):
        return float('inf')
    elif linear_dist > 2.0:
        raise ValueError, "linear_dist, must not exceed the diameter of the unit sphere, 2.0"
    c = 2*math.pi*radius
    a2 = linear_dist**2
    theta = math.degrees(math.acos((2-a2)/(2.)))
    d = (theta*c)/360.0
    return d
    
def toXYZ(pt):
    """ ASSUMPTION: pt = (lng,lat)
        REASON: pi = 180 degrees,
                theta+(pi/2)....
                theta = 90 degrees,
                180 =  90+180/2"""
    phi,theta = map(math.radians,pt)
    phi,theta = phi+pi,theta+(pi/2)
    x = 1*sin(theta)*cos(phi)
    y = 1*sin(theta)*sin(phi)
    z = 1*cos(theta)
    return x,y,z

def toLngLat(xyz):
    x,y,z = xyz
    if z == -1 or z == 1:
        phi = 0
    else:
        phi = math.atan2(y,x)
        if phi > 0:
            phi = phi-math.pi
        elif phi < 0:
            phi = phi+math.pi
    theta = math.acos(z)-(math.pi/2)
    return phi,theta

def brute_knn(pts,k,mode='arc'):
    """
    valid modes are ['arc','xrz']
    """
    n = len(pts)
    full = numpy.zeros((n,n))
    for i in xrange(n):
        for j in xrange(i+1,n):
            if mode == 'arc':
                lng0,lat0= pts[i]
                lng1,lat1= pts[j]
                dist = arcdist(pts[i],pts[j], radius=RADIUS_EARTH_KM)
            elif mode == 'xyz':
                dist = euclidean(pts[i],pts[j])
            full[i,j] = dist
            full[j,i] = dist
    w = {}
    for i in xrange(n):
        w[i] = full[i].argsort()[1:k+1].tolist()
    return w
def fast_knn(pts,k,return_dist=False):
    pts = numpy.array(pts)
    kd = scipy.spatial.KDTree(pts)
    d,w = kd.query(pts,k+1)
    w = w[:,1:]
    wn = {}
    for i in xrange(len(pts)):
        wn[i] = w[i].tolist()
    if return_dist:
        d = d[:,1:]
        wd = {}
        for i in xrange(len(pts)):
            wd[i] = [linear2arcdist(x, radius=RADIUS_EARTH_MILES) for x in d[i].tolist()]
        return wn,wd
    return wn
def fast_threshold(pts,dist,radius=RADIUS_EARTH_KM):
    d = arcdist2linear(dist,radius)
    kd = scipy.spatial.KDTree(pts)
    r = kd.query_ball_tree(kd,d)
    wd = {}
    for i in xrange(len(pts)):
        l = r[i]
        l.remove(i)
        wd[i] = l
    return wd



if __name__=='__main__':
    def random_ll():
        long = (random.random()*360) - 180
        lat = (random.random()*180) - 90
        return long,lat

    for i in range(1):
        n = 99
        # generate random surface points.
        pts = [random_ll() for i in xrange(n)]
        # convert to unit sphere points.
        pts2 = map(toXYZ, pts)
        
        w = brute_knn(pts,4,'arc')
        w2 = brute_knn(pts2,4,'xyz')
        w3 = fast_knn(pts2,4)
        assert w == w2 == w3
    import doctest
    doctest.testmod()


    ### Make knn1
    import pysal
    f = pysal.open('/Users/charlie/Documents/data/stl_hom/stl_hom.shp','r')
    shapes = f.read()
    pts = [shape.centroid for shape in shapes]
    w0 = brute_knn(pts,4,'xyz')
    w1 = brute_knn(pts,4,'arc')
    pts = map(toXYZ, pts)
    w2 = brute_knn(pts,4,'xyz')
    w3 = fast_knn(pts,4)

    wn,wd = fast_knn(pts,4,True)
    ids = range(1,len(pts)+1)
    
