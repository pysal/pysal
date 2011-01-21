import pysal
import os.path
import pysal.core.FileIO as FileIO
from pysal.weights import W
from warnings import warn

__author__ = "Charles R Schmidt <Charles.R.Schmidt@asu.edu>"
__all__ = ["GwtIO"]

class GwtIO(FileIO.FileIO):

    FORMATS = ['gwt']
    MODES = ['r', 'w']

    def __init__(self,*args,**kwargs):
        self._varName = 'Unknown'
        self._shpName = 'Unknown'
        FileIO.FileIO.__init__(self,*args,**kwargs)
        self.file = open(self.dataPath,self.mode)

    def _set_varName(self,val):
        if issubclass(type(val),basestring):
            self._varName=val
    def _get_varName(self):
        return self._varName
    varName = property(fget=_get_varName,fset=_set_varName)
    def _set_shpName(self,val):
        if issubclass(type(val),basestring):
            self._shpName=val
    def _get_shpName(self):
        return self._shpName
    shpName = property(fget=_get_shpName,fset=_set_shpName)

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
        self.shpName = shp
        self.varName = id_var
        id_order = None
        id_type = str
        try:
            base = os.path.split(self.dataPath)[0]
            dbf = os.path.join(base,self.shpName.replace('.shp','')+'.dbf')
            if os.path.exists(dbf):
                db = pysal.open(dbf,'r')
                if id_var in db.header:
                    id_order = db.by_col(id_var)
                    id_type = type(id_order[0])
                else:
                    warn("ID_VAR:'%s' was in in the DBF header, proceeding with unordered string ids."%(id_var), RuntimeWarning)
            else:
                warn("DBF relating to GWT was not found, proceeding with unordered string ids.", RuntimeWarning)
        except:
            warn("Exception occurred will reading DBF, proceeding with unordered string ids.", RuntimeWarning)
        self.flag=flag
        self.n=n
        self.shp=shp
        self.id_var=id_var
        weights={}
        neighbors={}
        for line in self.file.readlines():
            i,j,w=line.strip().split()
            i=id_type(i)
            j=id_type(j)
            w=float(w)
            if i not in weights:
                weights[i]=[]
                neighbors[i]=[]
            weights[i].append(w)
            neighbors[i].append(j)

        self.pos += 1
        return W(neighbors,weights,id_order)

    def write(self, obj):
        """.write(weightsObject)
        write a weights object to the opened file
        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj),W):
            header = '%s %i %s %s\n' % ('0', obj.n, self.shpName, self.varName)
            self.file.write(header)
            for id in obj.id_order:
                neighbors = zip(obj.neighbors[id], obj.weights[id])
                for neighbor, weight in neighbors:
                    self.file.write('%s %s %6G\n' % (str(id), str(neighbor), weight))
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


