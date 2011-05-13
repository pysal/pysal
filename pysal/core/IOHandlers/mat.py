import pysal
import os.path
from numpy import fromstring
from numpy import array
from struct import pack, unpack, calcsize
import zlib
from cStringIO import StringIO
from time import ctime
import pysal.core.FileIO as FileIO
from pysal.weights import W
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["MatIO"]

class MatIO(FileIO.FileIO):
    """
    Opens, reads, and writes weights file objects in MATLAB Level 5 MAT format.

    MAT files are used in Dr. LeSage's MATLAB Econometrics library.

    PySAL can read only MATLAB array and compressed data types.
    For MATLAB array data type, PySAL can read only MATLAB sparse, 
    double-precision, single-precision, and all integer arrays.
    The MAT file format can handle both full and sparse matrices, 
    and it allows for a matrix dimension greater than 256. 
    In PySAL, row and column headers of a MATLAB array are ignored. 
    Also, in PySAL, it is assumed that the input mat file includes 
    only one MATLAB array or a compressed data item that 
    condensed one MATLAB array.
    PySAL writes a spatial weights object in a mat file using 
    the compressed data item that condenses a MATLAB sparase array.
    
    The detailed specifications of the MAT file format are fully described in 
    MathWorks (2011) "MATLAB 7 MAT-File Format" at
    http://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf.

    Notes
    -----
    This module is based upon the following source code:
    http://abel.ee.ucla.edu/cvxopt/examples/extra-utilities/matfile.py/view?searchterm=mat

    References
    ----------
    MathWorks (2011) "MATLAB 7 MAT-File Format" at
    http://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf.
    
    """

    FORMATS = ['mat']
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
        """Reads MATLAB mat file
        Returns a pysal.weights.weights.W object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open a MATLAB mat file and read it into a pysal weights object

        >>> w = pysal.open('../../examples/spat-sym-us.mat','r').read()

        Get the number of observations from the header

        >>> w.n
        46

        Get the mean number of neighbors

        >>> w.mean_neighbors
        4.0869565217391308

        Get neighbor distances for a single observation

        >>> w[1]
        {25: 1, 3: 1, 28: 1, 39: 1}

        """
        if self.pos > 0:
            raise StopIteration

        header = self.file.read(126)
        E = '<'
        if unpack('<2s', self.file.read(2))[0] == 'MI':
            E = '>'
        platform, version = unpack(E+'124s1H', header)
        if version != 0x0100: 
            raise ValueError, "This file is not a MATLAB Level 5 MAT file."

        neighbors = {}
        weights = {}
        def unpack_array(array_data):
            f = StringIO()
            f.write(array_data)
            f.seek(0)
            # array flag subelement
            aftype, aflen, aflags, max_nonzero = unpack(E+'4i', f.read(16))
            # dimensions array
            dimtype, dimlen = unpack(E+'2i', f.read(8))
            if dimlen > 2*4:
                raise ValueError, 'This array has %i dimensions. Only 2-dimensional arrays are supported.' % dimlen/4
            dimvalue = unpack(E+'%ii' % (dimlen/4), f.read(dimlen))
            # array name
            antype, anlen = unpack(E+'2i', f.read(8))
            anmaxlen = 8
            cls = aflags
            if cls == 5:
                anlen, antype = antype, anlen
                anmaxlen = 4
            aname = unpack(E+'%is' % anlen, f.read(anlen))
            if (anlen % anmaxlen) > 0:
                f.read(anmaxlen - (anlen % anmaxlen))
            # array class
            if cls < 5: 
                raise ValueError, 'This type of array is not supported' 
            elif cls == 5: # sparse array
                if aflags & 8:
                    raise ValueError, 'Sparse array with complex numbers is not supported'
                rdata_type, rdata_len = unpack(E+'2i', f.read(8))
                rdata = unpack(E+'%ii' % (rdata_len/4), f.read(rdata_len))
                cdata_type, cdata_len = unpack(E+'2i', f.read(8))
                cdata = unpack(E+'%ii' % (cdata_len/4), f.read(cdata_len))
                celldata_type, celldata_len = unpack(E+'2i', f.read(8))
                celldata = unpack(E+'%id' % (celldata_len/8), f.read(celldata_len))
                for i, j, w in zip(rdata, cdata, celldata):
                    ngh = neighbors.setdefault(i,[])
                    wgt = weights.setdefault(i,[])
                    ngh.append(j)
                    wgt.append(w)
            else:
                ndata_type, ndata_len = unpack(E+'2i', f.read(8))
                dtype_map = {1:(1,'b'),2:(1,'B'),3:(2,'h'),4:(2,'H'),5:(4,'i'),6:(4,
                             'I'),9:(8,'d')}
                if ndata_type not in dtype_map.keys():
                    raise ValueError, 'The data type used in the input file is not supported'
                dtlen, dtcode = dtype_map[ndata_type]
                ndata = unpack(E+'%i%s' % (ndata_len/dtlen, dtcode), f.read(ndata_len))
                n = dimvalue[0]
                for i in xrange(n):
                    js = array(ndata[i*n:(i+1)*n])
                    js_nonzero = js.nonzero()[0]
                    neighbors[i] = list(js_nonzero)
                    weights[i] = list(js[js_nonzero]) 
            f.close()
 
        data_header = self.file.read(8)
        if data_header == '':
            raise ValueError, 'No data in your intput mat file'
        dtype, dlen = unpack(E+'2i', data_header)
        if dtype not in [14, 15]: # MATLAB Array, compressed data
            raise ValueError, 'PySAL does not support this type of MATLAB data'
        elif dtype == 14:
            unpack_array(self.file.read(dlen))
        else: # dtype == 15
            data = zlib.decompress(self.file.read(dlen))
            dtype, dlen = unpack(E+'2i', data[:8])
            if dtype != 14:
                raise ValueError, 'PySAL does not support this type of MATLAB data'
            unpack_array(data[8:])

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

        a MATLAB mat file
        write a weights object to the opened mat file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open('../../examples/spat-sym-us.mat','r')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.mat')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'w')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created mat file

        >>> wnew =  pysal.open(fname,'r').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)

        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj),W):
            f = self.file
            n = obj.n
            bof = 'MATLAB 5.0 MAT-file, Created by PySAL on ' + ctime()
            endian = 256*ord('M') + ord('I')
            f.write(pack('124s2H',bof + ' '*124,0x0100,endian))

            ar_name = 'WGHT'
            ar_name_len = len(ar_name)
            ar_name_blen = int(8*(ar_name_len/8.0))
            w = obj.sparse

            matlab_sparse = ''
            # array flags
            arflags, cls, max_nonzero = 0, 5, obj.max_neighbors
            matlab_sparse = pack('4i',6,8,arflags*256+cls,max_nonzero)
            # dimensions array
            matlab_sparse += pack('4i',5,8,w.shape[0],w.shape[1])
            # array name
            matlab_sparse += pack('2i4s',ar_name_len,1,ar_name)

            w = w.tocoo()
            # row indices
            matlab_sparse += pack('2i',5,len(w.row)*4)
            row_data = tuple(['%ii' % len(w.row)] + w.row.tolist())
            matlab_sparse += pack(*row_data)
            # column indices
            matlab_sparse += pack('2i',5,len(w.col)*4)
            col_data = tuple(['%ii' % len(w.col)] + w.col.tolist())
            matlab_sparse += pack(*col_data)
            # cell data
            matlab_sparse += pack('2i',5,len(w.data)*8)
            cell_data = tuple(['%id' % len(w.data)] + w.data.tolist())
            matlab_sparse += pack(*cell_data)
            matlab_sparse = pack('2i',14,len(matlab_sparse)) + matlab_sparse
            
            matlab_sparse = zlib.compress(matlab_sparse, 9)
            f.write(pack('2i',15,len(matlab_sparse)))
            f.write(matlab_sparse)

            self.pos += 1

        else:
            raise TypeError, "Expected a pysal weights object, got: %s" % (type(obj))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
