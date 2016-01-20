import pysal
import os.path
import numpy as np
from struct import pack, unpack
import pysal.core.FileIO as FileIO
from pysal.weights import W
from pysal.weights.util import remap_ids
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["ArcGISSwmIO"]


class ArcGISSwmIO(FileIO.FileIO):
    """
    Opens, reads, and writes weights file objects in ArcGIS swm format.

    Spatial weights objects in the ArcGIS swm format are used in
    ArcGIS Spatial Statistics tools.
    Particularly, this format can be directly used with the tools under
    the category of Mapping Clusters.

    The values for [ORG_i] and [DST_i] should be integers,
    as ArcGIS Spatial Statistics tools support only unique integer IDs.
    For the case where a weights object uses non-integer IDs,
    ArcGISSwmIO allows users to use internal ids corresponding to record numbers,
    instead of original ids.

    The specifics of each part of the above structure is as follows.

  .. table:: ArcGIS SWM Components

    ============ ============ ==================================== ================================
        Part      Data type           Description                   Length                        
    ============ ============ ==================================== ================================
     ID_VAR_NAME  ASCII TEXT  ID variable name                     Flexible (Up to the 1st ;)     
     ESRI_SRS     ASCII TEXT  ESRI spatial reference system        Flexible (Btw the 1st ; and \\n)  
     NO_OBS       l.e. int    Number of observations               4                         
     ROW_STD      l.e. int    Whether or not row-standardized      4                         
     WGT_i                                                                                   
     ORG_i        l.e. int    ID of observaiton i                  4                         
     NO_NGH_i     l.e. int    Number of neighbors for obs. i (m)   4                         
     NGHS_i                                                                                  
     DSTS_i       l.e. int    IDs of all neighbors of obs. i       4*m                       
     WS_i         l.e. float  Weights for obs. i and its neighbors 8*m                       
     W_SUM_i      l.e. float  Sum of weights for "                 8                         
    ============ ============ ==================================== ================================

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
        """
        Reads ArcGIS swm file.
        Returns a pysal.weights.weights.W object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open an ArcGIS swm file and read it into a pysal weights object

        >>> w = pysal.open(pysal.examples.get_path('ohio.swm'),'r').read()

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
        header01 = header01.decode()
        id_var, srs = header01[:-1].split(';')
        self.varName = id_var
        self.header_len = len(header01) + 8
        no_obs, row_std = tuple(unpack('<2l', self.file.read(8)))

        neighbors = {}
        weights = {}
        for i in xrange(no_obs):
            origin, no_nghs = tuple(unpack('<2l', self.file.read(8)))
            neighbors[origin] = []
            weights[origin] = []
            if no_nghs > 0:
                neighbors[origin] = list(unpack('<%il' %
                                                no_nghs, self.file.read(4 * no_nghs)))
                weights[origin] = list(unpack('<%id' %
                                              no_nghs, self.file.read(8 * no_nghs)))
                w_sum = list(unpack('<d', self.file.read(8)))[0]

        self.pos += 1
        return W(neighbors, weights)

    def write(self, obj, useIdIndex=False):
        """
        Writes a spatial weights matrix data file in swm format.

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        -------

        an ArcGIS swm file
        write a weights object to the opened swm file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open(pysal.examples.get_path('ohio.swm'),'r')
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

        >>> wnew = pysal.open(fname,'r').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname) """

        self._complain_ifclosed(self.closed)
        if issubclass(type(obj), W):
            if not (type(obj.id_order[0]) in (np.int32, np.int64, int)) and not useIdIndex:
                raise TypeError("ArcGIS SWM files support only integer IDs")
            if useIdIndex:
                id2i = obj.id2i
                obj = remap_ids(obj, id2i)
            unk = str('%s;Unknown\n' % self.varName).encode()
            self.file.write(unk)
            self.file.write(pack('<l', obj.n))
            self.file.write(pack('<l', obj.transform.upper() == 'R'))
            for obs in obj.weights:
                self.file.write(pack('<l', obs))
                no_nghs = len(obj.weights[obs])
                self.file.write(pack('<l', no_nghs))
                self.file.write(pack('<%il' % no_nghs, *obj.neighbors[obs]))
                self.file.write(pack('<%id' % no_nghs, *obj.weights[obs]))
                self.file.write(pack('<d', sum(obj.weights[obs])))
            self.pos += 1

        else:
            raise TypeError("Expected a pysal weights object, got: %s" % (
                type(obj)))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)
