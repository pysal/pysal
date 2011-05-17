import pysal
import os.path
from numpy import fromstring
from numpy import array
from struct import pack
import pysal.core.FileIO as FileIO
from pysal.weights import W
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["ArcGISSwmIO"]

class ArcGISSwmIO(FileIO.FileIO):
    """
    Opens, reads, and writes weights file objects in ArcGIS swm format.
    
    Spatial weights objects in the ArcGIS swm format are used in 
    ArcGIS Spatial Statistics tools.
    
    An exemplary structure of an ArcGIS swm file is as follows:
    [ID_VAR_NAME];[ESRI_SRS]\n[NO_OBS][ROW_STD][WGT_1]...[WGT_i]...[WGT_n]
    where [WGT_i] consists of [ORG_i][NO_NGH_i][NGHS_i] 
    and [NGHS_i] cosists of [DSTS_i][WS_i][W_SUM_i].
    Here, n is the number of observations.

    The specifics of each part of the above structure is as follows:
    Part	    Data type		   Desription				Length
    [ID_VAR_NAME]   ASCII TEXT		   ID variable name			Flexible (Up to the 1st ;)
    [ESRI_SRS]	    ASCII TEXT		   ESRI spatial reference system	Flexible (Btw the 1st ; and \n)
    [NO_OBS]	    little endian integer  Number of observations		4
    [ROW_STD]	    little endian integer  Whether or not row-standardized 	4
    [WGT_i]
      [ORG_i]  	    little endian integer  ID of observaiton i 		   	4
      [NO_NGH_i]    little endian integer  Number of neighbors for obs. i (m)	4
      [NGHS_i]
        [DSTS_i]    little endian integer  IDs of all neighbors of obs. i	4*m
        [WS_i]      little endian float    Weights for obs. i and its neighbors 8*m		
        [W_SUM_i]   little endian float    Sum of weights for "                 8

    References
    ----------
    ArcGIS 9.3 SWM2Table.py
    
    """

    FORMATS = ['swm']
    MODES = ['r', 'w']

    def __init__(self, *args, **kwargs):
        self._varName = 'Unknown'
        FileIO.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode + 'b')

    def _set_varName(self, val):
        if issubclass(type(val), basestring):
            self._varName = val
    def _get_varName(self):
        return self._varName
    varName = property(fget=_get_varName, fset=_set_varName)

    def read(self, n=-1):
        self._complain_ifclosed(self.closed)
        return self._read()

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _read(self):
        """Reads ArcGIS swm file
        Returns a pysal.weights.weights.W object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open an ArcGIS swm file and read it into a pysal weights object

        >>> w = pysal.open('../../examples/ohio.swm','r').read()

        Get the number of observations from the header

        >>> w.n
        88

        Get the mean number of neighbors

        >>> w.mean_neighbors
        5.25

        Get neighbor distances for a single observation

        >>> w[1]
        {2: 1.0, 11: 1.0, 6: 1.0, 7: 1.0}

        """
        if self.pos > 0:
            raise StopIteration

        header01 = self.file.readline()
        id_var, srs = header01[:-1].split(';')
        self.varName = id_var            
        self.header_len = len(header01) + 8
        no_obs, row_std = tuple(fromstring(self.file.read(8), '<i'))

        neighbors = {}
        weights = {}
        for i in xrange(no_obs):
            origin, no_nghs = tuple(fromstring(self.file.read(8), '<i'))
            neighbors[origin] = []
            weights[origin] = []
            if no_nghs > 0:
                neighbors[origin]  = list(fromstring(self.file.read(4*no_nghs), '<i'))
                weights[origin] = list(fromstring(self.file.read(8*no_nghs), '<d'))
                w_sum = list(fromstring(self.file.read(8), '<d'))[0]

        self.pos += 1
        return W(neighbors,weights)

    def write(self, obj):
        """ 

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        ------

        an ArcGIS swm file
        write a weights object to the opened swm file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open('../../examples/ohio.swm','r')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.swm')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'w')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created text file

        >>> wnew =  pysal.open(fname,'r').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)

        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj),W):
            self.file.write('%s;Unknown\n' % self.varName)
            self.file.write(pack('<l', obj.n))
            self.file.write(pack('<l', obj.transform.upper() == 'R'))
            for obs in obj.weights:
                self.file.write(pack('<l', obs))
                self.file.write(pack('<l', len(obj.weights[obs])))
                self.file.write(array(obj.neighbors[obs], '<l').tostring())
                self.file.write(array(obj.weights[obs], '<d').tostring())
                self.file.write(pack('<d', sum(obj.weights[obs])))
            self.pos += 1

        else:
            raise TypeError, "Expected a pysal weights object, got: %s" % (type(obj))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
