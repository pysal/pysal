from ..cg.shapes import Polygon, Chain
import itertools as it
import collections
QUEEN = 1
ROOK = 2

__author__ = "Jay Laura jlaura@asu.edu"

def _get_verts(shape):
    if isinstance(shape, (Polygon, Chain)):
        return shape.vertices
    else:
        return _get_boundary_points(shape)

def _get_boundary_points(shape):
    """
    Recursively handle polygons vs. multipolygons to
    extract the boundary point set from each. 
    """
    if shape.type.lower() == 'polygon':
        shape = shape.boundary
        return _get_boundary_points(shape)
    elif shape.type.lower() == 'linestring':
        return list(map(tuple, list(zip(*shape.coords.xy))))
    elif shape.type.lower() == 'multilinestring':
        return list(it.chain(*(list(zip(*shape.coords.xy))
                                 for shape in shape)))
    elif shape.type.lower() == 'multipolygon':
        return list(it.chain(*(_get_boundary_points(part.boundary) 
                               for part in shape)))
    else:
        raise TypeError('Input shape must be a Polygon, Multipolygon, LineString, '
                        ' or MultiLinestring and was '
                        ' instead: {}'.format(shape.type))


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
