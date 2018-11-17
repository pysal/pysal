import numpy as np
from struct import pack, unpack
from .. import fileio
from ...weights import W
from ...weights.util import remap_ids

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["ArcGISSwmIO"]


class ArcGISSwmIO(fileio.FileIO):
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
        self._srs = "Unknow"
        fileio.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode + 'b')

    def _set_varName(self, val):
        if issubclass(type(val), str):
            self._varName = val

    def _get_varName(self):
        return self._varName

    varName = property(fget=_get_varName, fset=_set_varName)

    def _set_srs(self, val):
        if issubclass(type(val), str):
            self._srs = val

    def _get_srs(self):
        return self._srs

    srs = property(fget=_get_srs, fset=_set_srs)

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
        Returns a libpysal.weights.weights.W object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open an ArcGIS swm file and read it into a pysal weights object

        >>> import libpysal
        >>> w = libpysal.io.open(libpysal.examples.get_path('ohio.swm'),'r').read()

        Get the number of observations from the header

        >>> w.n
        88

        Get the mean number of neighbors

        >>> w.mean_neighbors
        5.25

        Get neighbor distances for a single observation

        >>> w[1] == dict({2: 1.0, 11: 1.0, 6: 1.0, 7: 1.0})
        True

        """
        if self.pos > 0:
            raise StopIteration

        header = self.file.readline()
        header = header.decode()

        if header.upper().strip().startswith("VERSION@"):
            #  deal with the new SWM version
            return self.read_new_version(header)
        else:
            #  deal with the old SWM version
            return self.read_old_version(header)

    def read_old_version(self, header):
        """
        Read the old version of ArcGIS(<10.1) swm file
        :param header:
        :return:
        """
        id_var, srs = header[:-1].split(';')
        self.varName = id_var
        self.srs = srs
        self.header_len = len(header) + 8
        no_obs, row_std = tuple(unpack('<2l', self.file.read(8)))
        neighbors = {}
        weights = {}
        for i in range(no_obs):
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

    def read_new_version(self, header_line):
        """
        Read the new version of ArcGIS(<10.1) swm file, which contains more parameters
        and records weights in two ways: fixed or variable
        :param header_line: str, the firs line of the swm file, which contains a lot of parameters.
                            The parameters are divided by ";" and the key-value of each parameter is divided by "@"
        :return:
        """
        headerDict = {}
        for item in header_line.split(";"):
            key, value = item.split("@")
            headerDict[key] = value
        # for the reader, in order to generate the PySAL Weight class, only a few of the parameters are needed.
        self.varName = headerDict["UNIQUEID"]
        self.srs = headerDict["SPATIALREFNAME"]

        fixedWeights = False
        if "FIXEDWEIGHTS" in headerDict:
            fixedWeights = headerDict["FIXEDWEIGHTS"].upper().strip() == 'TRUE'

        no_obs, row_std = tuple(unpack('<2l', self.file.read(8)))
        is_row_standard = row_std == 1

        neighbors = {}
        weights = {}
        for i in range(no_obs):
            origin, no_nghs = tuple(unpack('<2l', self.file.read(8)))
            neighbors[origin] = []
            weights[origin] = []
            if no_nghs > 0:
                neighbors[origin] = list(unpack('<%il' %
                                                no_nghs, self.file.read(4 * no_nghs)))
                if fixedWeights:
                    weights[origin] = list(unpack('<d', self.file.read(8))) * no_nghs
                else:
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

        Add properities to the file to write

        >>> o.varName = testfile.varName
        >>> o.srs = testfile.srs

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
        if not issubclass(type(obj), W):
            raise TypeError("Expected a pysal weights object, got: %s" % (
                type(obj)))
        if not (type(obj.id_order[0]) in (np.int32, np.int64, int)) and not useIdIndex:
            raise TypeError("ArcGIS SWM files support only integer IDs")
        if useIdIndex:
            id2i = obj.id2i
            obj = remap_ids(obj, id2i)

        unk = str('%s;%s\n' % (self.varName, self.srs)).encode()
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

    def close(self):
        self.file.close()
        fileio.FileIO.close(self)
