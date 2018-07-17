from shapely import geometry as geom, ops as shops, __version__ as shapely_version
_basegeom = geom.base.BaseGeometry
from .shapes import asShape
__all__ = ["to_wkb", "to_wkt", "area", "distance", "length", "boundary", "bounds", "centroid", "representative_point", "convex_hull", "envelope", "buffer", "simplify", "difference", "intersection", "symmetric_difference", "union", "unary_union", "cascaded_union", "has_z", "is_empty", "is_ring", "is_simple", "is_valid", "relate", "contains", "crosses", "disjoint", "equals", "intersects", "overlaps", "touches", "within", "equals_exact", "almost_equals", "project", "interpolate"]


def to_wkb(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.to_wkb()

def to_wkt(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.to_wkt()

# Real-valued properties and methods
# ----------------------------------
def area(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.area

def distance(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.distance(o2)

def length(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.length

# Topological properties
# ----------------------
def boundary(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    res = o.boundary
    return asShape(res)

def bounds(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.bounds

def centroid(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    res = o.centroid
    return asShape(res)

def representative_point(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    res = o.representative_point()
    return asShape(res)

def convex_hull(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    res = o.convex_hull
    return asShape(res)

def envelope(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    res = o.envelope
    return asShape(res)

def buffer(shape, radius, resolution=16):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    res = o.buffer(radius, resolution)
    return asShape(res)

def simplify(shape, tolerance, preserve_topology=True):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    res = o.simplify(tolerance, preserve_topology)
    return asShape(res)

# Binary operations
# -----------------
def difference(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    res = o.difference(o2)
    return asShape(res)

def intersection(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    res = o.intersection(o2)
    return asShape(res)

def symmetric_difference(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    res = o.symmetric_difference(o2)
    return asShape(res)

def union(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    res = o.union(o2)
    return asShape(res)

def cascaded_union(shapes):
    o = []
    for shape in shapes:
        if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
        o.append(geom.asShape(shape))
    res = shops.cascaded_union(o)
    return asShape(res)

def unary_union(shapes):
    # seems to be the same as cascade_union except that it handles multipart polygons
    if shapely_version < '1.2.16':
        raise Exception("shapely 1.2.16 or higher needed for unary_union; upgrade shapely or try cascade_union instead")
    o = []
    for shape in shapes:
        if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
        o.append(geom.asShape(shape))
    res = shops.unary_union(o)
    return asShape(res)

# Unary predicates
# ----------------
def has_z(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.has_z

def is_empty(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.is_empty

def is_ring(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.is_ring

def is_simple(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.is_simple

def is_valid(shape):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    return o.is_valid

# Binary predicates
# -----------------
def relate(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.relate(o2)

def contains(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.contains(o2)

def crosses(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.crosses(o2)

def disjoint(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.disjoint(o2)

def equals(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.equals(o2)

def intersects(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.intersects(o2)

def overlaps(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.overlaps(o2)

def touches(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.touches(o2)

def within(shape, other):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.within(o2)

def equals_exact(shape, other, tolerance):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.equals_exact(o2, tolerance)

def almost_equals(shape, other, decimal=6):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.almost_equals(o2, decimal)

# Linear referencing
# ------------------

def project(shape, other, normalized=False):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    if not hasattr(other,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    o2 = geom.asShape(other)
    return o.project(o2, normalized)

def interpolate(shape, distance, normalized=False):
    if not hasattr(shape,'__geo_interface__'): raise TypeError("%r does not appear to be a shape"%shape)
    o = geom.asShape(shape)
    res = o.interpolate(distance, normalized)
    return asShape(res)


# Copy doc strings from shapely
for method in __all__:
    if hasattr(_basegeom, method):
        locals()[method].__doc__ = getattr(_basegeom,method).__doc__
