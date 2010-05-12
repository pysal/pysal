import pysal.core.FileIO as FileIO
from pysal.weights import W


class GwtIO(FileIO.FileIO):

    FORMATS = ['gwt']
    MODES = ['r', 'w']

    def __init__(self,*args,**kwargs):
        FileIO.FileIO.__init__(self,*args,**kwargs)
        self.file = open(self.dataPath,self.mode)

    def read(self,n=-1):
        return self._read()

    def seek(self,pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0
    def _read(self):
        if self.pos > 0:
            raise StopIteration

        flag,n,shp,id_var = self.file.readline().strip().split()
        self.flag=flag
        self.n=n
        self.shp=shp
        self.id_var=id_var
        weights={}
        neighbors={}
        for line in self.file.readlines():
            i,j,w=line.strip().split()
            #i=int(i)
            #j=int(j)
            w=float(w)
            if i not in weights:
                weights[i]=[]
                neighbors[i]=[]
            weights[i].append(w)
            neighbors[i].append(j)
        self.original_neighbors=neighbors
        zo=self.__zero_offset(neighbors,weights)
        self.neighbors=zo['new_neighbors']
        self.original_ids=zo['old_ids']
        self.ids = zo['new_ids']
        self.n=len(weights)
        self.weights=zo['new_weights']

        self.pos += 1
        return W(neighbors,weights)

    def write(self, obj):
        """.write(weightsObject)
        write a weights object to the opened file
        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj),W):
            IDS = obj.id_order
            header = '%s %i %s %s\n' % ('', obj.n, '', '')
            self.file.write(header)
            for id in IDS:
                neighbors = zip(obj.neighbors[id], obj.weights[id])
                for neighbor, weight in neighbors:
                    self.file.write('%s %s %d\n' % (id, neighbor, weight))
                    self.pos += 1
        else:
            raise TypeError, "Expected a pysal weights object, got: %s" % (type(obj))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)

    @staticmethod
    def __zero_offset(neighbors,weights,original_ids=None):
        if not original_ids:
            original_ids=neighbors.keys()
        old_weights=weights
        new_weights={}
        new_ids={}
        old_ids={}
        new_neighbors={}
        for i in original_ids:
            new_i=original_ids.index(i)
            new_ids[new_i]=i
            old_ids[i]=new_i
            neighbors_i=neighbors[i]
            new_neighbors_i=[original_ids.index(j) for j in neighbors_i]
            new_neighbors[new_i]=new_neighbors_i
            new_weights[new_i]=weights[i]
        info={}
        info['new_ids']=new_ids
        info['old_ids']=old_ids
        info['new_neighbors']=new_neighbors
        info['new_weights']=new_weights
        return info


