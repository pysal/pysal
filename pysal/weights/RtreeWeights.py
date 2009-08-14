import rtree
#Order by Degree of connectivity, i.e. rook is more connected then queen.
QUEEN = 1
ROOK = 2

Q_TARGET_MEM_SIZE = 5 * 1024 * 1024 #5mb
class PolyQ(dict):
    def __init__(self):
        dict.__init__(self)
        self.size = 20 # user the first 20 object to calc the average Size.
        self.ids = []
    def __checkSize(self):
        """ Use the objects in the Q to calculate the average size of the objects
            Adjust Q.size to hold Q_TARGET_MEM_SIZE/avgSize object
            This is as many average size object that fit into Q_TARGET_MEM_SIZE """
        size = sum(map(len,self.values())) * 2 * 8 # *(2 doubles/point) * (8 bytes/double)
        if size < Q_TARGET_MEM_SIZE:
            avgSize = size/float(len(self))
            self.size = Q_TARGET_MEM_SIZE/avgSize
            return False #make the Q larger
        else:
            return True #continue with purge
        
    def add(self,poly):
        if poly.id not in self:
            if len(self.ids) >= self.size:
                if self.__checkSize():
                    del self[self.ids.pop(0)]
            self[poly.id] = poly
            self.ids.append(poly.id)
class ContWeights:
    def __init__(self,geoObj,joinType=ROOK):
        self.index = rtree.Rtree()
        self.geoObj = geoObj
        self.joinType = joinType
        self.w = {}
        self.Q = PolyQ()
    def create(self):
        for poly in self.geoObj:
            self.append(poly)
    def append(self,poly):
        self.Q.add(poly)
        for id in self.index.intersection(poly.bbox):
            if self.check(id,poly) >= self.joinType:
                self.setW(id,poly.id)
        if poly.id not in self.w: #add the null cases
            self.w[poly.id] = set()
        self.index.add(poly.bbox.id,poly.bbox)
    def setW(self,id0,id1):
        "updates the W matrix seting two polygon's as neighbors"
        w = self.w
        if id0 not in w:
            w[id0] = set()
        if id1 not in w:
            w[id1] = set()
        w[id0].add(id1)
        w[id1].add(id0)
    def check(self,id0,poly1):
        "Check's if two polygon's are neighbors"
        if id0 in self.Q:
            poly0 = self.Q[id0]
        else:
            poly0 = self.geoObj.get(id0)
        #poly1 = self.geoObj.get(id1)
        common = set(poly0).intersection(set(poly1))
        #common = poly0.intersection(poly1)
        if len(common) > 1:
            #double check rook
            return ROOK
        elif len(common) == 1:
            return QUEEN
        else:
            return False

if __name__=='__main__':
    #from pysal import GeoIO
    import pysal
    shp = pysal.open('examples/usCounties/usa.shp','r')
    #shp = GeoIO.GeoIO('examples/usCounties/co99_d00.shp')
    w = ContWeights(shp,QUEEN)
    import time
    import cProfile
    t0 = time.time()
    #cProfile.run('w.create()')
    w.create()
    t1 = time.time()
    print "Completed in: ",t1-t0,"seconds"
