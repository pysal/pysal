import pysal as ps
import collections
QUEEN = 1
ROOK = 2

__author__ = "Jay Laura jlaura@asu.edu"

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
        self.collection = collection
        self.wttype = wttype
        self.jcontiguity()

    def jcontiguity(self):
        if not isinstance(self.collection[0], ps.cg.Polygon):
            return False

        numPoly = len(self.collection)

        w = {}
        for i in range(numPoly):
            w[i] = set()

        geoms = []
        offsets = []
        c = 0  # PolyID Counter

        if self.wttype == QUEEN:
            for n in range(numPoly):
                    verts = self.collection[n].vertices
                    offsets += [c] * len(verts)
                    geoms += (verts)
                    c += 1

            items = collections.defaultdict(set)
            for i, vertex in enumerate(geoms):
                items[vertex].add(offsets[i])

            shared_vertices = []
            for item, location in items.iteritems():
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
                verts = self.collection[n].vertices
                for v in range(len(verts) - 1):
                    geoms.append(tuple(sorted([verts[v], verts[v + 1]])))
                offsets += [c] * (len(verts) - 1)
                c += 1

            items = collections.defaultdict(set)
            for i, item in enumerate(geoms):
                items[item].add(offsets[i])

            shared_vertices = []
            for item, location in items.iteritems():
                if len(location) > 1:
                    shared_vertices.append(location)

            for vert_set in shared_vertices:
                for v in vert_set:
                    w[v] = w[v] | vert_set
                    try:
                        w[v].remove(v)
                    except:
                        pass
        self.w = w
