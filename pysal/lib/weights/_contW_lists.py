from ..cg.shapes import Polygon
import itertools as it
from sys import version_info
import collections
QUEEN = 1
ROOK = 2
if version_info[0] == 2:
    zip = it.izip
    range = xrange

__author__ = "Jay Laura jlaura@asu.edu"

def _get_verts(pgon):
    if isinstance(pgon, Polygon):
        return pgon.vertices
    else:
        return _get_boundary_points(pgon)

def _get_boundary_points(pgon):
    """
    Recursively handle polygons vs. multipolygons to
    extract the boundary point set from each. 
    """
    if pgon.type.lower() == 'polygon':
        bounds = pgon.boundary
        if bounds.type.lower() == 'linestring':
            return list(map(tuple, list(zip(*bounds.coords.xy))))
        elif bounds.type.lower() == 'multilinestring':
            return list(it.chain(*(list(zip(*bound.coords.xy))
                                     for bound in bounds)))
        else:
            raise TypeError('Input Polygon has unrecognized boundary type: {}'
                            ''.format(bounds.type))
    elif pgon.type.lower() == 'multipolygon':
        return list(it.chain(*(_get_boundary_points(part) 
                               for part in pgon)))
    else:
        raise TypeError('Input shape must be Polygon or Multipolygon and was '
                        'instead: {}'.format(pgon.type))


class ContiguityWeightsLists:
    """
    Contiguity for a collection of polygons using high performance
    list, set, and dict containers
    """
    def __init__(self, collection, wttype=1):
        """
        Arguments
        =========

        collection: PySAL PolygonCollection

        wttype: int
                1: Queen
                2: Rook
        """
        self.collection = list(collection)
        self.wttype = wttype
        self.jcontiguity()

    def jcontiguity(self):
        numPoly = len(self.collection)

        w = {}
        for i in range(numPoly):
            w[i] = set()

        geoms = []
        offsets = []
        c = 0  # PolyID Counter

        if self.wttype == QUEEN:
            for n in range(numPoly):
                    verts = _get_verts(self.collection[n])
                    offsets += [c] * len(verts)
                    geoms += (verts)
                    c += 1

            items = collections.defaultdict(set)
            for i, vertex in enumerate(geoms):
                items[vertex].add(offsets[i])

            shared_vertices = []
            for item, location in list(items.items()):
                if len(location) > 1:
                    shared_vertices.append(location)

            for vert_set in shared_vertices:
                for v in vert_set:
                    w[v] = w[v] | vert_set
                    try:
                        w[v].remove(v)
                    except:
                        pass

        elif self.wttype == ROOK:
            for n in range(numPoly):
                verts = _get_verts(self.collection[n])
                for v in range(len(verts) - 1):
                    geoms.append(tuple(sorted([verts[v], verts[v + 1]])))
                offsets += [c] * (len(verts) - 1)
                c += 1

            items = collections.defaultdict(set)
            for i, item in enumerate(geoms):
                items[item].add(offsets[i])

            shared_vertices = []
            for item, location in list(items.items()):
                if len(location) > 1:
                    shared_vertices.append(location)

            for vert_set in shared_vertices:
                for v in vert_set:
                    w[v] = w[v] | vert_set
                    try:
                        w[v].remove(v)
                    except:
                        pass
        else:
            raise Exception('Weight type {} Not Understood!'.format(self.wttype))
        self.w = w
