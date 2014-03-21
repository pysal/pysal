import pysal.cg.rtree as rtree
from pysal.cg.standalone import get_shared_segments
#Order by Degree of connectivity, i.e. rook is more connected then queen.
QUEEN = 1
ROOK = 2

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ["QUEEN", "ROOK", "ContiguityWeights_rtree"]

Q_TARGET_MEM_SIZE = 250 * 1024 * 1024  # 250mb


class _PolyQ(dict):
    def __init__(self):
        dict.__init__(self)
        self.size = 20  # use the first 20 objects to calculate the average Size.
        self.ids = []

    def __checkSize(self):
        """
        Use the objects in the Q to calculate the average size of the objects
        Adjust Q.size to hold Q_TARGET_MEM_SIZE/avgSize object
        This is as many average size object that fit into Q_TARGET_MEM_SIZE
        """
        if len(self.ids) > 50:
            return True
        return False

    def add(self, poly):
        if poly.id not in self:
            if len(self.ids) >= self.size:
                if self.__checkSize():
                    del self[self.ids.pop(0)]
            self[poly.id] = poly
            self.ids.append(poly.id)


class ContiguityWeights_rtree:
    def __init__(self, geoObj, joinType=ROOK):
        self.index = rtree.Rtree()
        self.geoObj = geoObj
        self.joinType = joinType
        self.w = {}
        self.Q = _PolyQ()
        self.cache_hits = 0
        self.cache_misses = 0
        self.create()
        #print "Misses: ",self.cache_misses
        #print "Hits: ",self.cache_hits

    def create(self):
        for id, poly in enumerate(self.geoObj):
            poly.id = id
            self.append(poly)

    def append(self, poly):
        self.Q.add(poly)
        b = poly.bounding_box
        bbox = [b.left, b.lower, b.right, b.upper]
        for id in self.index.intersection(bbox):
            id = int(id)
            if self.check(id, poly) >= self.joinType:
                self.setW(id, poly.id)
        if poly.id not in self.w:  # add the null cases
            self.w[poly.id] = set()
        self.index.add(poly.id, bbox)

    def setW(self, id0, id1):
        "updates the W matrix seting two polygon's as neighbors"
        w = self.w
        if id0 not in w:
            w[id0] = set()
        if id1 not in w:
            w[id1] = set()
        w[id0].add(id1)
        w[id1].add(id0)

    def check(self, id0, poly1):
        "Check's if two polygon's are neighbors"
        if id0 in self.Q:
            self.cache_hits += 1
            poly0 = self.Q[id0]
        else:
            self.cache_misses += 1
            poly0 = self.geoObj.get(id0)
            poly0.id = id0
            self.Q.add(poly0)
        common = set(poly0.vertices).intersection(set(poly1.vertices))
        if len(common) > 1 and self.joinType == ROOK:
            #double check rook
            if get_shared_segments(poly0, poly1, True):
                return ROOK
            return False
            #for vert in common:
            #    idx = poly0.vertices.index(vert)
            #    IDX = poly1.vertices.index(vert)
            #    try:
            #        if poly0.vertices[idx+1] == poly1.vertices[IDX+1] or poly0.vertices[idx+1] == poly1.vertices[IDX-1]\
            #        or poly0.vertices[idx-1] == poly1.vertices[IDX+1] or poly0.vertices[idx-1] == poly1.vertices[IDX-1]:
            #            return ROOK
            #    except IndexError:
            #        pass
            #return False
        elif len(common) > 0:
            return QUEEN
        else:
            return False

if __name__ == '__main__':
    import pysal
    import time
    t0 = time.time()
    shp = pysal.open(pysal.examples.get_path('10740.shp'), 'r')
    w = ContiguityWeights_rtree(shp, QUEEN)
    t1 = time.time()
    print "Completed in: ", t1 - t0, "seconds using rtree"
