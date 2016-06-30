import numpy as np
from pysal.common import RTOL
from .shapes import Point, Chain, Polygon, Geometry

#### This approach is invalid :(
#### OGC has a different definition of equality, so we can't be ogc-compliant
#### by focusing on vertex set relationships. For example:

# from shapely import geometry as geom
# p1 = geom.Polygon([(0,0),(0,1),(1,0)])
# p2 = geom.Polygon([(0,0),(0,1),(.5,.5),(1,0)])
# p1.equals(p2) #is true
# no relationship on point set (without simplifying boundaries) will make these
# approaches work. 

ATOL = 1e-12

def _point_equal(a,b):
    """
    Test that a point is exactly equal to another point, using numpy array
    comparison
    """
    return np.array_equal(a._Point__loc, b._Point__loc)

def _point_almost_equal(a,b, rtol=RTOL, atol=ATOL):
    """
    test that one point is equal to another point, up to a specified relative or
    absolute tolerance.
    """
    return np.allclose(a._Point__loc, b._Point__loc, 
                       rtol=rtol, atol=atol)

def _chain_equal(a,b):
    """
    Test that two chains are equal. This considers reversed-incident chains,
    chains with the same pointset but in reverse order, as different chains. 
    """
    for a_part, b_part in zip(a.parts, b.parts):
        for a_seg, b_seg in zip(a_part, b_part):
            if not np.array_equal(a_seg, b_seg):
                return False
    return True 

def _chain_almost_equal(a,b, rtol=RTOL, atol=ATOL):
    """
    Test that two chains are equal, up to a specified relative or absolute
    tolerance. This considers reversed-incident chains,
    chains with the same pointset but in reverse order, as different chains. 
    """
    for a_part, b_part in zip(a.parts, b.parts):
        for a_seg, b_seg in zip(a_part, b_part):
            if not np.allclose(a_seg, b_seg, 
                               rtol=RTOL, atol=ATOL):
                return False
    return True

def _poly_exactly_equal(a,b):
    """
    Test that two polygons are equal. This is a straightfoward linear comparison
    of parts and holes. That is, it is assumed that the parts and holes are
    conformal.

    Thus, shapes with the same parts or holes, but listed in a different order,
    will be considered different. Solving this will require some way to relate
    ring/hole sets
    """
    for a_hole, b_hole in zip(a.holes, b.holes):
        if not np.array_equal(a_hole, b_hole):
            return False
    for a_part, b_part in zip(a.parts, b.parts):
        if not np.array_equal(a_part, b_part):
            return False
    return True

def _poly_exactly_coincident(a,b):
    """
    Check that two polygons have coincident boundaries 
    
    Thus, this returns True when two polygons have the same holes & parts in any
    order with potentially-reversed path directions
    """
    n_holes = len(a.holes)
    n_parts = len(a.parts)
    if n_holes != len(b.holes):
        return False
    if n_parts != len(b.parts):
        return False
    b_in_a = [None]*n_holes
    a_in_b = [None]*n_holes
    for i, a_hole in a.holes:
        for j, b_hole in b.holes:
            i_j = coincident(a_hole, b_hole)
            if i_j:
                b_in_a[j] = True
                a_in_b[i] = True
                break
        if not a_in_b[i]:
            return False
    if any(in_b):
        return False
    for b_hole in b.holes:
        in_a = [not coincident(b_hole, a_hole) for a_hole in a.holes]
    if any(in_a):
        return False
    return True

def _coincident(a,b):
    """
    Check if two vertex lists are equal, either forwards or backwards. Thus,
    this checks for equality of the path or equality when one path is reversed. 
    """
    return np.array_equal(a, b) or np.array_equal(np.flipud(a),b)

def _almost_coincident(a,b, rtol=RTOL, atol=ATOL):
    """
    Check if two vertex lists are equal to within a given relative and absolute
    tolernance, either forwards or backwards. Thus,
    this checks for equality of the path or equality when one path is reversed. 
    """
    return (np.allclose(a, b, rtol=RTOL, atol=ATOL) 
            or np.allclose(np.flipud(a),b, rtol=RTOL, atol=ATOL))

def _poly_almost_equal(a,b, rtol=RTOL, atol=ATOL):
    is_equal = True
    for a_hole, b_hole in zip(a.holes, b.holes):
        is_equal &= np.allclose(a_hole, b_hole,
                                rtol=rtol, atol=atol)
    for a_part, b_part in zip(a.parts, b.parts):
        is_equal &= np.allclose(a_part, b_part, 
                                rtol=rtol, atol=atol)
    return is_equal

_EQ_MAP = {Point:_point_equal, 
           Polygon:_poly_equal,
           Chain: _chain_equal}
_AEQ_MAP = {Point:_point_almost_equal, 
            Polygon:_poly_almost_equal,
            Chain: _chain_almost_equal}

def equal(a,b):
    ta, tb = type(a), type(b)
    if not all([isinstance(a, tb), isinstance(b, ta)]):
        return False
    return _EQ_MAP[ta](a,b)

def almost_equal(a,b, rtol=RTOL, atol=ATOL):
    ta, tb = type(a), type(b)
    if not all([isinstance(a, tb), isinstance(b, ta)]):
        return False
    return _AEQ_MAP[ta](a,b)

def is_shape(a):
    return isinstance(a, Geometry)
