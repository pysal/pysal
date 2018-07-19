"""
sphere: Tools for working with spherical geometry.

Author(s):
    Charles R Schmidt schmidtc@gmail.com
    Luc Anselin luc.anselin@asu.edu
    Xun Li xun.li@asu.edu

"""

__author__ = "Charles R Schmidt <schmidtc@gmail.com>, Luc Anselin <luc.anselin@asu.edu, Xun Li <xun.li@asu.edu"

import math
import random
import numpy
import scipy.spatial
import scipy.constants
from scipy.spatial.distance import euclidean
from math import pi, cos, sin, asin

__all__ = ['RADIUS_EARTH_KM', 'RADIUS_EARTH_MILES', 'arcdist', 'arcdist2linear', 'brute_knn', 'fast_knn', 'fast_threshold', 'linear2arcdist', 'toLngLat', 'toXYZ', 'lonlat','harcdist','geointerpolate','geogrid']


RADIUS_EARTH_KM = 6371.0
RADIUS_EARTH_MILES = (
    RADIUS_EARTH_KM * scipy.constants.kilo) / scipy.constants.mile


def arcdist(pt0, pt1, radius=RADIUS_EARTH_KM):
    """
    Parameters
    ----------
    pt0 : point
        assumed to be in form (lng,lat)
    pt1 : point
        assumed to be in form (lng,lat)
    radius : radius of the sphere
        defaults to Earth's radius

        Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html

    Returns
    -------
    The arc distance between pt0 and pt1 using supplied radius

    Examples
    --------
    >>> pt0 = (0,0)
    >>> pt1 = (180,0)
    >>> d = arcdist(pt0,pt1,RADIUS_EARTH_MILES)
    >>> d == math.pi*RADIUS_EARTH_MILES
    True
    """
    return linear2arcdist(euclidean(toXYZ(pt0), toXYZ(pt1)), radius)


def arcdist2linear(arc_dist, radius=RADIUS_EARTH_KM):
    """
    Convert an arc distance (spherical earth) to a linear distance (R3) in the unit sphere.

    Examples
    --------
    >>> pt0 = (0,0)
    >>> pt1 = (180,0)
    >>> d = arcdist(pt0,pt1,RADIUS_EARTH_MILES)
    >>> d == math.pi*RADIUS_EARTH_MILES
    True
    >>> arcdist2linear(d,RADIUS_EARTH_MILES)
    2.0
    """
    c = 2 * math.pi * radius
    d = (2 - (2 * math.cos(math.radians((arc_dist * 360.0) / c)))) ** (0.5)
    return d


def linear2arcdist(linear_dist, radius=RADIUS_EARTH_KM):
    """
    Convert a linear distance in the unit sphere (R3) to an arc distance based on supplied radius

    Examples
    --------
    >>> pt0 = (0,0)
    >>> pt1 = (180,0)
    >>> d = arcdist(pt0,pt1,RADIUS_EARTH_MILES)
    >>> d == linear2arcdist(2.0, radius = RADIUS_EARTH_MILES)
    True
    """
    if linear_dist == float('inf'):
        return float('inf')
    elif linear_dist > 2.0:
        raise ValueError("linear_dist, must not exceed the diameter of the unit sphere, 2.0")
    c = 2 * math.pi * radius
    a2 = linear_dist ** 2
    theta = math.degrees(math.acos((2 - a2) / (2.)))
    d = (theta * c) / 360.0
    return d


def toXYZ(pt):
    """
    Parameters
    ----------
    pt0 : point
        assumed to be in form (lng,lat)
    pt1 : point
        assumed to be in form (lng,lat)

    Returns
    -------
    x, y, z
    """
    phi, theta = list(map(math.radians, pt))
    phi, theta = phi + pi, theta + (pi / 2)
    x = 1 * sin(theta) * cos(phi)
    y = 1 * sin(theta) * sin(phi)
    z = 1 * cos(theta)
    return x, y, z


def toLngLat(xyz):
    x, y, z = xyz
    if z == -1 or z == 1:
        phi = 0
    else:
        phi = math.atan2(y, x)
        if phi > 0:
            phi = phi - math.pi
        elif phi < 0:
            phi = phi + math.pi
    theta = math.acos(z) - (math.pi / 2)
    return phi, theta


def brute_knn(pts, k, mode='arc'):
    """
    valid modes are ['arc','xrz']
    """
    n = len(pts)
    full = numpy.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if mode == 'arc':
                lng0, lat0 = pts[i]
                lng1, lat1 = pts[j]
                dist = arcdist(pts[i], pts[j], radius=RADIUS_EARTH_KM)
            elif mode == 'xyz':
                dist = euclidean(pts[i], pts[j])
            full[i, j] = dist
            full[j, i] = dist
    w = {}
    for i in range(n):
        w[i] = full[i].argsort()[1:k + 1].tolist()
    return w


def fast_knn(pts, k, return_dist=False):
    """
    Computes k nearest neighbors on a sphere.

    Parameters
    ----------
    pts :  list of x,y pairs
    k   :  int
        Number of points to query
    return_dist : bool
        Return distances in the 'wd' container object

    Returns
    -------
    wn  :  list
        list of neighbors
    wd  : list
        list of neighbor distances (optional)

    """
    pts = numpy.array(pts)
    kd = scipy.spatial.KDTree(pts)
    d, w = kd.query(pts, k + 1)
    w = w[:, 1:]
    wn = {}
    for i in range(len(pts)):
        wn[i] = w[i].tolist()
    if return_dist:
        d = d[:, 1:]
        wd = {}
        for i in range(len(pts)):
            wd[i] = [linear2arcdist(x,
                                    radius=RADIUS_EARTH_MILES) for x in d[i].tolist()]
        return wn, wd
    return wn


def fast_threshold(pts, dist, radius=RADIUS_EARTH_KM):
    d = arcdist2linear(dist, radius)
    kd = scipy.spatial.KDTree(pts)
    r = kd.query_ball_tree(kd, d)
    wd = {}
    for i in range(len(pts)):
        l = r[i]
        l.remove(i)
        wd[i] = l
    return wd


########### new functions

def lonlat(pointslist):
    """
    Converts point order from lat-lon tuples to lon-lat (x,y) tuples

    Parameters
    ----------

    pointslist : list of lat-lon tuples (Note, has to be a list, even for one point)

    Returns
    -------

    newpts      : list with tuples of points in lon-lat order

    Example
    -------
    >>> points = [(41.981417, -87.893517), (41.980396, -87.776787), (41.980906, -87.696450)]
    >>> newpoints = lonlat(points)
    >>> newpoints
    [(-87.893517, 41.981417), (-87.776787, 41.980396), (-87.69645, 41.980906)]

    """
    newpts = [(i[1],i[0]) for i in pointslist]
    return newpts

def haversine(x):
    """
    Computes the haversine formula

    Parameters
    ----------
    x    : angle in radians

    Returns
    -------
         : square of sine of half the radian (the haversine formula)

    Example
    -------
    >>> haversine(math.pi)     # is 180 in radians, hence sin of 90 = 1
    1.0

    """

    x = math.sin(x/2)
    return x*x

# Lambda functions

# degree to radian conversion
d2r = lambda x: x * math.pi / 180.0

# radian to degree conversion
r2d = lambda x: x * 180.0 / math.pi

def radangle(p0,p1):
    """
    Radian angle between two points on a sphere in lon-lat (x,y)

    Parameters
    ----------
    p0    : first point as a lon,lat tuple
    p1    : second point as a lon,lat tuple

    Returns
    -------
    d     : radian angle in radians

    Example
    -------
    >>> p0 = (-87.893517, 41.981417)
    >>> p1 = (-87.519295, 41.657498)
    >>> radangle(p0,p1)
    0.007460167953189258

    Note
    ----
    Uses haversine formula, function haversine and degree to radian
    conversion lambda function d2r

    """
    x0, y0 = d2r(p0[0]),d2r(p0[1])
    x1, y1 = d2r(p1[0]),d2r(p1[1])
    d = 2.0 * math.asin(math.sqrt(haversine(y1 - y0) +
                        math.cos(y0) * math.cos(y1)*haversine(x1 - x0)))
    return d

def harcdist(p0,p1,lonx=True,radius=RADIUS_EARTH_KM):
    """
    Alternative arc distance function, uses haversine formula

    Parameters
    ----------
    p0       : first point as a tuple in decimal degrees
    p1       : second point as a tuple in decimal degrees
    lonx     : boolean to assess the order of the coordinates,
               for lon,lat (default) = True, for lat,lon = False
    radius   : radius of the earth at the equator as a sphere
               default: RADIUS_EARTH_KM (6371.0 km)
               options: RADIUS_EARTH_MILES (3959.0 miles)
                        None (for result in radians)

    Returns
    -------
    d        : distance in units specified, km, miles or radians (for None)

    Example
    -------
    >>> p0 = (-87.893517, 41.981417)
    >>> p1 = (-87.519295, 41.657498)
    >>> harcdist(p0,p1)
    47.52873002976876
    >>> harcdist(p0,p1,radius=None)
    0.007460167953189258

    Note
    ----
    Uses radangle function to compute radian angle

    """
    if not(lonx):
        p = lonlat([p0,p1])
        p0 = p[0]
        p1 = p[1]

    d = radangle(p0,p1)
    if radius is not None:
        d = d*radius
    return d

def geointerpolate(p0,p1,t,lonx=True):
    """
    Finds a point on a sphere along the great circle distance between two points
    on a sphere
    also known as a way point in great circle navigation

    Parameters
    ----------
    p0       : first point as a tuple in decimal degrees
    p1       : second point as a tuple in decimal degrees
    t        : proportion along great circle distance between p0 and p1
               e.g., t=0.5 would find the mid-point
    lonx     : boolean to assess the order of the coordinates,
               for lon,lat (default) = True, for lat,lon = False

    Returns
    -------
    x,y      : tuple in decimal degrees of lon-lat (default) or lat-lon,
               depending on setting of lonx; in other words, the same
               order is used as for the input

    Example
    -------
    >>> p0 = (-87.893517, 41.981417)
    >>> p1 = (-87.519295, 41.657498)
    >>> geointerpolate(p0,p1,0.1)          # using lon-lat
    (-87.85592403438788, 41.949079912574796)
    >>> p3 = (41.981417, -87.893517)
    >>> p4 = (41.657498, -87.519295)
    >>> geointerpolate(p3,p4,0.1,lonx=False)   # using lat-lon
    (41.949079912574796, -87.85592403438788)

    """

    if not(lonx):
        p = lonlat([p0,p1])
        p0 = p[0]
        p1 = p[1]

    d = radangle(p0,p1)
    k = 1.0 / math.sin(d)
    t = t*d
    A = math.sin(d-t) * k
    B = math.sin(t) * k

    x0, y0 = d2r(p0[0]),d2r(p0[1])
    x1, y1 = d2r(p1[0]),d2r(p1[1])

    x = A * math.cos(y0) * math.cos(x0) + B * math.cos(y1) * math.cos(x1)
    y = A * math.cos(y0) * math.sin(x0) + B * math.cos(y1) * math.sin(x1)
    z = A * math.sin(y0) + B * math.sin(y1)

    newpx = r2d(math.atan2(y, x))
    newpy = r2d(math.atan2(z, math.sqrt(x*x + y*y)))
    if not(lonx):
        return newpy,newpx
    return newpx,newpy

def geogrid(pup,pdown,k,lonx=True):
    """
    Computes a k+1 by k+1 set of grid points for a bounding box in lat-lon
    uses geointerpolate

    Parameters
    ----------
    pup     : tuple with lat-lon or lon-lat for upper left corner of bounding box
    pdown   : tuple with lat-lon or lon-lat for lower right corner of bounding box
    k       : number of grid cells (grid points will be one more)
    lonx    : boolean to assess the order of the coordinates,
              for lon,lat (default) = True, for lat,lon = False

    Returns
    -------
    grid    : list of tuples with lat-lon or lon-lat for grid points, row by row,
              starting with the top row and moving to the bottom; coordinate tuples
              are returned in same order as input

    Example
    -------
    >>> pup = (42.023768,-87.946389)    # Arlington Heights IL
    >>> pdown = (41.644415,-87.524102)  # Hammond, IN
    >>> geogrid(pup,pdown,3,lonx=False)
    [(42.023768, -87.946389), (42.02393997819538, -87.80562679358316), (42.02393997819538, -87.66486420641684), (42.023768, -87.524102), (41.897317, -87.94638900000001), (41.8974888973743, -87.80562679296166), (41.8974888973743, -87.66486420703835), (41.897317, -87.524102), (41.770866000000005, -87.94638900000001), (41.77103781320412, -87.80562679234043), (41.77103781320412, -87.66486420765956), (41.770866000000005, -87.524102), (41.644415, -87.946389), (41.64458672568646, -87.80562679171955), (41.64458672568646, -87.66486420828045), (41.644415, -87.524102)]

    """
    if lonx:
        corners = [pup,pdown]
    else:
        corners = lonlat([pup,pdown])
    tpoints = [float(i)/k for i in range(k)[1:]]
    leftcorners = [corners[0],(corners[0][0],corners[1][1])]
    rightcorners = [(corners[1][0],corners[0][1]),corners[1]]
    leftside = [leftcorners[0]]
    rightside = [rightcorners[0]]
    for t in tpoints:
        newpl = geointerpolate(leftcorners[0],leftcorners[1],t)
        leftside.append(newpl)
        newpr = geointerpolate(rightcorners[0],rightcorners[1],t)
        rightside.append(newpr)
    leftside.append(leftcorners[1])
    rightside.append(rightcorners[1])

    grid = []
    for i in range(len(leftside)):
        grid.append(leftside[i])
        for t in tpoints:
            newp = geointerpolate(leftside[i],rightside[i],t)
            grid.append(newp)
        grid.append(rightside[i])
    if not(lonx):
        grid = lonlat(grid)
    return grid
