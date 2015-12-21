"""
Centrographic measures for point patterns

Author: Serge Rey <sjsrey@gmail.com>


TODO

dispersion
ellipses
euclidean median
testing
documentation

"""
import sys
import numpy as np
import warnings
import shapely
from scipy.spatial import ConvexHull


MAXD = sys.float_info.max
MIND = sys.float_info.min


def mbr(points):
    """
    Minimum bounding rectangle
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
    Find convex hull of a point array

    Arguments
    ---------

    points: array (n x 2)

    Returns
    -------
    _ : array (h x 2)
        points defining the hull in counterclockwise order
    """
    points = np.asarray(points)
    h = ConvexHull(points)
    return points[h.vertices]


def mean_center(points):
    points = np.asarray(points)
    points.mean(axis=0)


def weighted_mean_center(points, weights):
    points, weights = np.asarray(points), np.asarray(weights)
    w = weights * 1. / weights.sum()
    w.shape = (1, len(points))
    return np.dot(w, points)[0]


def manhattan_median(points):
    points = np.asarray(points)
    if not len(points) % 2:
        s = "Manhattan Median is not unique for even point patterns."
        warnings.warn(s)
    return np.median(points, axis=0)
