import pysal
import os.path
import struct
import pysal.core.FileIO as FileIO
from pysal.weights import W
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["Wk1IO"]


class Wk1IO(FileIO.FileIO):
    """
    MATLAB wk1read.m and wk1write.m that were written by Brian M. Bourgault in 10/22/93

    Opens, reads, and writes weights file objects in Lotus Wk1 format.

    Lotus Wk1 file is used in Dr. LeSage's MATLAB Econometrics library.

    A Wk1 file holds a spatial weights object in a full matrix form
    without any row and column headers.
    The maximum number of columns supported in a Wk1 file is 256.
    Wk1 starts the row (column) number from 0 and
    uses little endian binary endcoding.
    In PySAL, when the number of observations is n,
    it is assumed that each cell of a n\*n(=m) matrix either is a blank or
    have a number.

    The internal structure of a Wk1 file written by PySAL is as follows:
    [BOF][DIM][CPI][CAL][CMODE][CORD][SPLIT][SYNC][CURS][WIN]
    [HCOL][MRG][LBL][CELL_1]...[CELL_m][EOF]
    where [CELL_k] equals to [DTYPE][DLEN][DFORMAT][CINDEX][CVALUE].
    The parts between [BOF] and [CELL_1] are variable according to the software
    program used to write a wk1 file. While reading a wk1 file,
    PySAL ignores them.
    Each part of this structure is detailed below.

 .. table:: Lotus WK1 fields

   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |Part         |Description          |Data Type                |Length |Value                        |
   +=============+=====================+=========================+=======+=============================+
   |[BOF]        |Begining of field    |unsigned character       |6      |0,0,2,0,6,4                  |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[DIM]        |Matrix dimension                                                                     |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [DIMDTYPE] |Type of dim. rec     |unsigned short           |2      |6                            |
   |  [DIMLEN]   |Length of dim. rec   |unsigned short           |2      |8                            |
   |  [DIMVAL]   |Value of dim. rec    |unsigned short           |8      |0,0,n,n                      |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[CPI]        |CPI                                                                                  |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [CPITYPE]  |Type of cpi rec      |unsigned short           |2      |150                          |
   |  [CPILEN]   |Length of cpi rec    |unsigned short           |2      |6                            |
   |  [CPIVAL]   |Value of cpi rec     |unsigned char            |6      |0,0,0,0,0,0                  |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[CAL]        |calcount                                                                             |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [CALTYPE]  |Type of calcount rec |unsigned short           |2      |47                           |
   |  [CALLEN]   |Length calcount rec  |unsigned short           |2      |1                            |
   |  [CALVAL]   |Value of calcount rec|unsigned char            |1      |0                            |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[CMODE]      |calmode                                                                              |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [CMODETYP] |Type of calmode rec  |unsigned short           |2      |2                            |
   |  [CMODELEN] |Length of calmode rec|unsigned short           |2      |1                            |
   |  [CMODEVAL] |Value of calmode rec |signed char              |1      |0                            |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[CORD]       |calorder                                                                             |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [CORDTYPE] |Type of calorder rec |unsigned short           |2      |3                            |
   |  [CORDLEN]  |Length calorder rec  |unsigned short           |2      |1                            |
   |  [CORDVAL]  |Value of calorder rec|signed char              |1      |0                            |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[SPLIT]      |split                                                                                |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [SPLTYPE]  |Type of split rec    |unsigned short           |2      |4                            |
   |  [SPLLEN]   |Length of split rec  |unsigned short           |2      |1                            |
   |  [SPLVAL]   |Value of split rec   |signed char              |1      |0                            |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[SYNC]       |sync                                                                                 |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [SYNCTYP]  |Type of sync rec     |unsigned short           |2      |5                            |
   |  [SYNCLEN]  |Length of sync rec   |unsigned short           |2      |1                            |
   |  [SYNCVAL]  |Value of sync rec    |singed char              |1      |0                            |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[CURS]       |cursor                                                                               |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [CURSTYP]  |Type of cursor rec   |unsigned short           |2      |49                           |
   |  [CURSLEN]  |Length of cursor rec |unsigned short           |2      |1                            |
   |  [CURSVAL]  |Value of cursor rec  |signed char              |1      |1                            |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[WIN]        |window                                                                               |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [WINTYPE]  |Type of window rec   |unsigned short           |2      |7                            |
   |  [WINLEN]   |Length of window rec |unsigned short           |2      |32                           |
   |  [WINVAL1]  |Value 1 of window rec|unsigned short           |4      |0,0                          |
   |  [WINVAL2]  |Value 2 of window rec|signed char              |2      |113,0                        |
   |  [WINVAL3]  |Value 3 of window rec|unsigned short           |26     |10,n,n,0,0,0,0,0,0,0,0,72,0  |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[HCOL]       |hidcol                                                                               |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [HCOLTYP]  |Type of hidcol rec   |unsigned short           |2      |100                          |
   |  [HCOLLEN]  |Length of hidcol rec |unsigned short           |2      |32                           |
   |  [HCOLVAL]  |Value of hidcol rec  |signed char              |32     |0*32                         |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[MRG]        |margins                                                                              |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [MRGTYPE]  |Type of margins rec  |unsigned short           |2      |40                           |
   |  [MRGLEN]   |Length of margins rec|unsigned short           |2      |10                           |
   |  [MRGVAL]   |Value of margins rec |unsigned short           |10     |4,76,66,2,2                  |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[LBL]        |labels                                                                               |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [LBLTYPE]  |Type of labels rec   |unsigned short           |2      |41                           |
   |  [LBLLEN]   |Length of labels rec |unsigned short           |2      |1                            |
   |  [LBLVAL]   |Value of labels rec  |char                     |1      |'                            |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |[CELL_k]                                                                                           |
   +-------------+---------------------+-------------------------+-------+-----------------------------+
   |  [DTYPE]    |Type of cell data    |unsigned short           |2      |[DTYPE][0]==0: end of file   |
   |             |                     |                         |       |          ==14: number       |
   |             |                     |                         |       |          ==16: formula      |
   |             |                     |                         |       |          ==13: integer      |
   |             |                     |                         |       |          ==11: nrange       |
   |             |                     |                         |       |          ==else: unknown    |
   |  [DLEN]     |Length of cell data  |unsigned short           |2      |                             |
   |  [DFORMAT]  |Format of cell data  |not sure                 |1      |                             |
   |  [CINDEX]   |Row, column of cell  |unsigned short           |4      |                             |
   |  [CVALUE]   |Value of cell        |double, [DTYPE][0]==14   |8      |                             |
   |             |                     |formula,[DTYPE][0]==16   |8 +    |[DTYPE][1] - 13              |
   |             |                     |integer,[DTYPE][0]==13   |2      |                             |
   |             |                     |nrange, [DTYPE][0]==11   |24     |                             |
   |             |                     |else,   [DTYPE][0]==else |       |[DTYPE][1]                   |
   |  [EOF]      |End of file          |unsigned short           |4      |1,0,0,0                      |
   +-------------+---------------------+-------------------------+-------+-----------------------------+


    """

    FORMATS = ['wk1']
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
        Reads Lotus Wk1 file

        Returns
        -------
        A pysal.weights.weights.W object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open a Lotus Wk1 file and read it into a pysal weights object

        >>> w = pysal.open(pysal.examples.get_path('spat-sym-us.wk1'),'r').read()

        Get the number of observations from the header

        >>> w.n
        46

        Get the mean number of neighbors

        >>> w.mean_neighbors
        4.0869565217391308

        Get neighbor distances for a single observation

        >>> w[1]
        {25: 1.0, 3: 1.0, 28: 1.0, 39: 1.0}

        """
        if self.pos > 0:
            raise StopIteration

        bof = struct.unpack('<6B', self.file.read(6))
        if bof != (0, 0, 2, 0, 6, 4):
            raise ValueError('The header of your file is wrong!')

        neighbors = {}
        weights = {}
        dtype, dlen = struct.unpack('<2H', self.file.read(4))
        while(dtype != 1):
            if dtype in [13, 14, 16]:
                self.file.read(1)
                row, column = struct.unpack('2H', self.file.read(4))
                format, length = '<d', 8
                if dtype == 13:
                    format, length = '<h', 2
                value = float(struct.unpack(format, self.file.read(length))[0])
                if value > 0:
                    ngh = neighbors.setdefault(row, [])
                    ngh.append(column)
                    wgt = weights.setdefault(row, [])
                    wgt.append(value)
                if dtype == 16:
                    self.file.read(dlen - 13)
            elif dtype == 11:
                self.file.read(24)
            else:
                self.file.read(dlen)
            dtype, dlen = struct.unpack('<2H', self.file.read(4))

        self.pos += 1
        return W(neighbors, weights)

    def write(self, obj):
        """

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        ------

        a Lotus wk1 file
        write a weights object to the opened wk1 file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open(pysal.examples.get_path('spat-sym-us.wk1'),'r')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.wk1')

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
        if issubclass(type(obj), W):
            f = self.file
            n = obj.n
            if n > 256:
                raise ValueError('WK1 file format supports only up to 256 observations.')
            pack = struct.pack
            f.write(pack('<6B', 0, 0, 2, 0, 6, 4))
            f.write(pack('<6H', 6, 8, 0, 0, n, n))
            f.write(pack('<2H6B', 150, 6, 0, 0, 0, 0, 0, 0))
            f.write(pack('<2H1B', 47, 1, 0))
            f.write(pack('<2H1b', 2, 1, 0))
            f.write(pack('<2H1b', 3, 1, 0))
            f.write(pack('<2H1b', 4, 1, 0))
            f.write(pack('<2H1b', 5, 1, 0))
            f.write(pack('<2H1b', 49, 1, 1))
            f.write(pack('<4H2b13H', 7, 32, 0, 0, 113, 0, 10,
                         n, n, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0))
            hidcol = tuple(['<2H32b', 100, 32] + [0] * 32)
            f.write(pack(*hidcol))
            f.write(pack('<7H', 40, 10, 4, 76, 66, 2, 2))
            f.write(pack('<2H1c', 41, 1, "'".encode()))

            id2i = obj.id2i
            for i, w_i in enumerate(obj):
                row = [0.0] * n
                for k in w_i[1]:
                    row[id2i[k]] = w_i[1][k]
                for c, v in enumerate(row):
                    cell = tuple(['<2H1b2H1d', 14, 13, 113, i, c, v])
                    f.write(pack(*cell))
            f.write(pack('<4B', 1, 0, 0, 0))
            self.pos += 1

        else:
            raise TypeError("Expected a pysal weights object, got: %s" % (
                type(obj)))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)


