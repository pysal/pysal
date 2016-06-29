import numpy as np
from pysal.common import RTOL
from .shapes import Point, Chain, Polygon, Geometry

ATOL = 1e-12

def _point_equal(a,b):
    return np.array_equal(a._Point__loc, b._Point__loc)

def _point_almost_equal(a,b, rtol=RTOL, atol=ATOL):
    return np.allclose(a._Point__loc, b._Point__loc, 
                       rtol=rtol, atol=atol)

def _chain_equal(a,b):
    is_equal = True
    for a_part, b_part in zip(a.parts, b.parts):
        for a_seg, b_seg in zip(a_part, b_part):
            is_equal &= np.array_equal(a_seg, b_seg)
    return is_equal 

def _chain_almost_equal(a,b, rtol=RTOL, atol=ATOL):
    is_equal = True
    for a_part, b_part in zip(a.parts, b.parts):
        for a_seg, b_seg in zip(a_part, b_part):
            is_equal &= np.allclose(a_seg, b_seg, 
                                    rtol=RTOL, atol=ATOL)

def _poly_equal(a,b):
    is_equal = True
    for a_holes, b_holes in zip(a.holes, b.holes):
        for a_hole, b_hole in zip(a_holes, b_holes):
            is_equal &= np.array_equal(a_hole, b_hole)
    for a_parts, b_parts in zip(a.parts, b.parts):
        for a_part, b_part in zip(a_parts, b_parts):
            is_equal &= np.array_equal(a_part, b_part)
    return is_equal

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
