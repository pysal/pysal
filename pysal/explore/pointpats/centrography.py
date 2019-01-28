"""
Centrographic measures for point patterns

TODO

- testing
- documentation

"""

__author__ = "Serge Rey sjsrey@gmail.com"
__all__ = ['mbr', 'hull', 'mean_center', 'weighted_mean_center',
           'manhattan_median', 'std_distance', 'euclidean_median', 'ellipse',
           'skyum', 'dtot',"_circle"]


import sys
import numpy as np
import warnings
import copy
from math import pi as PI
from scipy.spatial import ConvexHull
from pysal.lib.cg import get_angle_between, Ray, is_clockwise
from scipy.spatial import distance as dist
from scipy.optimize import minimize

not_clockwise = lambda x: not is_clockwise(x)

MAXD = sys.float_info.max
MIND = sys.float_info.min


def mbr(points):
    """
    Find minimum bounding rectangle of a point array.

    Parameters
    ----------
    points : arraylike
             (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    min_x  : float
             leftmost value of the vertices of minimum bounding rectangle.
    min_y  : float
             downmost value of the vertices of minimum bounding rectangle.
    max_x  : float
             rightmost value of the vertices of minimum bounding rectangle.
    max_y  : float
             upmost value of the vertices of minimum bounding rectangle.

    """
    points = np.asarray(points)
    min_x = min_y = MAXD
    max_x = max_y = MIND
    for point in points:
        x, y = point
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y
    return min_x, min_y, max_x, max_y


def hull(points):
    """
    Find convex hull of a point array.

    Parameters
    ----------
    points: arraylike
            (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _     : array
            (h,2), points defining the hull in counterclockwise order.
    """

    points = np.asarray(points)
    h = ConvexHull(points)
    return points[h.vertices]


def mean_center(points):
    """
    Find mean center of a point array.

    Parameters
    ----------
    points: arraylike
            (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _     : array
            (2,), (x,y) coordinates of the mean center.
    """

    points = np.asarray(points)
    return points.mean(axis=0)


def weighted_mean_center(points, weights):
    """
    Find weighted mean center of a marked point pattern.

    Parameters
    ----------
    points  : arraylike
              (n,2), (x,y) coordinates of a series of event points.
    weights : arraylike
              a series of attribute values of length n.

    Returns
    -------
    _      : array
             (2,), (x,y) coordinates of the weighted mean center.
    """


    points, weights = np.asarray(points), np.asarray(weights)
    w = weights * 1. / weights.sum()
    w.shape = (1, len(points))
    return np.dot(w, points)[0]


def manhattan_median(points):
    """
    Find manhattan median of a point array.

    Parameters
    ----------
    points  : arraylike
              (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _      : array
             (2,), (x,y) coordinates of the manhattan median.
    """

    points = np.asarray(points)
    if not len(points) % 2:
        s = "Manhattan Median is not unique for even point patterns."
        warnings.warn(s)
    return np.median(points, axis=0)


def std_distance(points):
    """
    Calculate standard distance of a point array.

    Parameters
    ----------
    points  : arraylike
              (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _      : float
             standard distance.
    """

    points = np.asarray(points)
    n, p = points.shape
    m = points.mean(axis=0)
    return np.sqrt(((points*points).sum(axis=0)/n - m*m).sum())


def ellipse(points):
    """
    Calculate parameters of standard deviational ellipse for a point pattern.

    Parameters
    ----------
    points : arraylike
             (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _      : float
             semi-major axis.
    _      : float
             semi-minor axis.
    theta  : float
             clockwise rotation angle of the ellipse.

    Notes
    -----
    Implements approach from:

    https://www.icpsr.umich.edu/CrimeStat/files/CrimeStatChapter.4.pdf
    """

    points = np.asarray(points)
    n, k = points.shape
    x = points[:, 0]
    y = points[:, 1]
    xd = x - x.mean()
    yd = y - y.mean()
    xss = (xd * xd).sum()
    yss = (yd * yd).sum()
    cv = (xd * yd).sum()
    num = (xss - yss) + np.sqrt((xss - yss)**2 + 4 * (cv)**2)
    den = 2 * cv
    theta = np.arctan(num / den)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    n_2 = n - 2
    sd_x = (2 * (xd * cos_theta - yd * sin_theta)**2).sum() / n_2
    sd_y = (2 * (xd * sin_theta - yd * cos_theta)**2).sum() / n_2
    return np.sqrt(sd_x), np.sqrt(sd_y), theta


def dtot(coord, points):
    """
    Sum of Euclidean distances between event points and a selected point.

    Parameters
    ----------
    coord   : arraylike
              (x,y) coordinates of a point.
    points  : arraylike
              (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    d       : float
              sum of Euclidean distances.

    """
    points = np.asarray(points)
    xd = points[:, 0] - coord[0]
    yd = points[:, 1] - coord[1]
    d = np.sqrt(xd*xd + yd*yd).sum()
    return d

def euclidean_median(points):
    """
    Calculate the Euclidean median for a point pattern.

    Parameters
    ----------
    points: arraylike
            (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _     : array
            (2,), (x,y) coordinates of the Euclidean median.

    """
    points = np.asarray(points)
    start = mean_center(points)
    res = minimize(dtot, start, args=(points,))
    return res['x']

def skyum(points, not_hull=True):
    """
    Implements Skyum (1990)'s algorithm for the minimum bounding circle in R^2.

    0. Store points clockwise.
    1. Find p in S that maximizes angle(prec(p), p, succ(p) THEN radius(prec(p),
    p, succ(p)). This is also called the lexicographic maximum, and is the last
    entry of a list of (radius, angle) in lexicographical order.
    2a. If angle(prec(p), p, succ(p)) <= 90 degrees, then finish.
    2b. If not, remove p from set.
    """
    points = hull(points).tolist()
    if not_clockwise(points):
        points.reverse()
        if not_clockwise(points):
            raise Exception('Points are neither clockwise nor counterclockwise')
    POINTS = copy.deepcopy(points)
    removed = []
    i=0
    while True:
        angles = [_angle(_prec(p, points), p, _succ(p, points)) for p in points]
        circles = [_circle(_prec(p, points), p, _succ(p, points)) for p in points]
        radii = [c[0] for c in circles]
        lexord = np.lexsort((radii, angles)) #confusing as hell defaults...
        lexmax = lexord[-1]
        candidate = (_prec(points[lexmax], points),
                     points[lexmax],
                     _succ(points[lexmax], points))
        if angles[lexmax] <= PI/2.0:
            #print("Constrained by points: {}".format(candidate))
            return _circle(*candidate), points, removed, candidate
        else:
            try:
                removed.append((points.pop(lexmax), i))
            except IndexError:
                raise Exception("Construction of Minimum Bounding Circle failed!")
        i+=1

def _angle(p,q,r):
    """
    compute the positive angle formed by PQR
    """
    return np.abs(get_angle_between(Ray(q,p),Ray(q,r)))

def _prec(p,l):
    """
    retrieve the predecessor of p in list l
    """
    pos = l.index(p)
    if pos-1 < 0:
        return l[-1]
    else:
        return l[pos-1]

def _succ(p,l):
    """
    retrieve the successor of p in list l
    """
    pos = l.index(p)
    if pos+1 >= len(l):
        return l[0]
    else:
        return l[pos+1]

def _circle(p,q,r, dmetric=dist.euclidean):
    """
    Returns (radius, (center_x, center_y)) of the circumscribed circle by the
    triangle pqr.

    note, this does not assume that p!=q!=r
    """
    px,py = p
    qx,qy = q
    rx,ry = r
    if np.allclose(np.abs(_angle(p,q,r)), PI):
        radius = dmetric(p,r)/2.
        center_x = (px + rx)/2.
        center_y = (py + ry)/2.
    elif np.allclose(np.abs(_angle(p,q,r)), 0):
        radius = dmetric(p,q)/2.
        center_x = (px + qx)/2.
        center_y = (py + qy)/2.
    else:
        D = 2*(px*(qy - ry) + qx*(ry - py) + rx*(py - qy))
        center_x = ((px**2 + py**2)*(qy-ry) + (qx**2 + qy**2)*(ry-py)
                  + (rx**2 + ry**2)*(py-qy)) / float(D)
        center_y = ((px**2 + py**2)*(rx-qx) + (qx**2 + qy**2)*(px-rx)
                  + (rx**2 + ry**2)*(qx-px)) / float(D)
        radius = dmetric((center_x, center_y), p)
    return radius, (center_x, center_y)
