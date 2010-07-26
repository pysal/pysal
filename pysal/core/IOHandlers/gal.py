import pysal.core.FileIO as FileIO
from pysal.weights import W

__author__='Charles R Schmidt <Charles.R.Schmidt@asu.edu>'

class GalIO(FileIO.FileIO):
    """
    Opens, reads, and writes file objects in GAL format.


    Parameters
    ----------


    Returns
    -------


    Notes
    -----


    Examples
    --------

    >>> import tempfile, pysal, os
    >>> w = pysal.open('../../examples/sids2.gal','r').read()
    >>> f = tempfile.NamedTemporaryFile(suffix='.gal')
    >>> fname = f.name
    >>> f.close()
    >>> o = pysal.open(fname,'w')
    >>> o.write(w)
    >>> o.close()
    >>> wnew =  pysal.open(fname,'r').read()
    >>> wnew.pct_nonzero == w.pct_nonzero
    True
    >>> os.remove(fname)

    """
    FORMATS = ['gal']
    MODES = ['r','w']

    def __init__(self,*args,**kwargs):
        FileIO.FileIO.__init__(self,*args,**kwargs)
        self.file = open(self.dataPath, self.mode)

    def read(self, n=-1):
        return self._read()

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _read(self):
        """
        Parameters
        ----------
        reads in a GalIO object

        Returns
        -------
        returns a W object
        """
        if self.pos > 0:
            raise StopIteration
        weights={}
        neighbors={}
        ids=[]
        # handle case where more than n is specified in first line
        header=self.file.readline().strip().split()
        header_n= len(header)
        n=int(header[0])
        if header_n > 1:
            n=int(header[1])
        w={}
        for i in range (n):
            id,n_neighbors=self.file.readline().strip().split()
            n_neighbors = int(n_neighbors)
            neighbors_i = self.file.readline().strip().split()
            weights[id]=[1]*n_neighbors
            neighbors[id]=neighbors_i
            ids.append(id)

        self.pos += 1 
        return W(neighbors,weights,ids)

    def write(self,obj):
        """ 

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        ------

        a GAL file
        write a weights object to the opened GAL file.
        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj),W):
            IDS = obj.id_order
            self.file.write('%d\n'%(obj.n))
            for id in IDS:
                neighbors = obj.neighbors[id]
                self.file.write('%s %d\n'%(str(id),len(neighbors)))
                self.file.write(' '.join(map(str,neighbors))+'\n')
            self.pos += 1
        else:
            raise TypeError,"Expected a pysal weights object, got: %s"%(type(obj))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)

def _test():
    import doctest, unittest
    doctest.testmod(verbose=True)
    unittest.main()

if __name__=='__main__':
    _test()



