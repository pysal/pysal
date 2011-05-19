import pysal
import os.path
import pysal.core.FileIO as FileIO
from pysal.weights import W
from warnings import warn

__author__ = "Charles R Schmidt <Charles.R.Schmidt@asu.edu>"
__all__ = ["GwtIO"]

class GwtIO(FileIO.FileIO):

    FORMATS = ['gwt']
    MODES = ['r']

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

    def read(self, n=-1):
        self._complain_ifclosed(self.closed)
        return self._read()
    def seek(self,pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _readlines(self, id_type):
        """
        Reads the main body of gwt-like weights files 
        into two dictionaries containing weights and neighbors.
        This code part is repeatedly used for many weight file formats.
        Header lines, however, are different from format to format. 
        So, for code reusability, this part is separated out from 
        _read function by Myunghwa Hwang.
        """
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
        return weights, neighbors        

    def _read(self):
        """Reads .gwt file
        Returns a pysal.weights.weights.W object

        Examples
        --------

        Type 'dir(f)' at the interpreter to see what methods are supported.
        Open .gwt file and read it into a pysal weights object

        >>> f = pysal.open('../../examples/juvenile.gwt','r').read()

        Get the number of observations from the header

        >>> f.n
        168

        Get the mean number of neighbors

        >>> f.mean_neighbors
        16.678571428571427

        Get neighbor distances for a single observation

        >>> f[1]
        {2: 14.1421356}


        """
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
                    warn("ID_VAR:'%s' was not in the DBF header, proceeding with unordered string ids."%(id_var), RuntimeWarning)
            else:
                warn("DBF relating to GWT was not found, proceeding with unordered string ids.", RuntimeWarning)
        except:
            warn("Exception occurred will reading DBF, proceeding with unordered string ids.", RuntimeWarning)
        self.flag=flag
        self.n=n
        self.shp=shp
        self.id_var=id_var
        weights, neighbors = self._readlines(id_type)

        self.pos += 1
        w = W(neighbors,weights,id_order)
        w.transform = 'b'
        warn("Weights have been converted to binary. To retrieve original values use w.transform='o'", RuntimeWarning)
        return w

    def _writelines(self, obj):
        """
        Writes  the main body of gwt-like weights files. 
        This code part is repeatedly used for many weight file formats.
        Header lines, however, are different from format to format. 
        So, for code reusability, this part is separated out from 
        write function by Myunghwa Hwang.
        """
        for id in obj.id_order:
            neighbors = zip(obj.neighbors[id], obj.weights[id])
            for neighbor, weight in neighbors:
                self.file.write('%s %s %6G\n' % (str(id), str(neighbor), weight))
                self.pos += 1

    def write(self, obj):
        """ 

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        ------

        a GWT file
        write a weights object to the opened GWT file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open('../../examples/juvenile.gwt','r')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.gwt')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'w')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created gwt file

        >>> wnew =  pysal.open(fname,'r').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)
        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj),W):
            header = '%s %i %s %s\n' % ('0', obj.n, self.shpName, self.varName)
            self.file.write(header)
            self._writelines(obj)

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

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
