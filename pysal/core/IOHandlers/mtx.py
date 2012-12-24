import pysal
import os.path
import scipy.io as sio
import pysal.core.FileIO as FileIO
from pysal.weights import W, WSP
from pysal.weights.util import full, full2W
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["MtxIO"]


class MtxIO(FileIO.FileIO):
    """
    Opens, reads, and writes weights file objects in Matrix Market MTX format.

    The Matrix Market MTX format is used to facilitate the exchange of matrix data.
    In PySAL, it is being tested as a new file format for delivering
    the weights information of a spatial weights matrix.
    Although the MTX format supports both full and sparse matrices with different
    data types, it is assumed that spatial weights files in the mtx format always
    use the sparse (or coordinate) format with real data values.
    For now, no additional assumption (e.g., symmetry) is made of the structure
    of a weights matrix.

    With the above assumptions,
    the structure of a MTX file containing a spatial weights matrix
    can be defined as follows:
    %%MatrixMarket matrix coordinate real general <--- header 1 (constant)
    % Comments starts                             <---
    % ....                                           | 0 or more comment lines
    % Comments ends                               <---
    M    N    L                                   <--- header 2, rows, columns, entries
    I1   J1   A(I1,J1)                            <---
    ...                                              | L entry lines
    IL   JL   A(IL,JL)                            <---

    In the MTX foramt, the index for rows or columns starts with 1.

    PySAL uses mtx io tools in scipy.
    Thus, it is subject to all limits that scipy currently has.
    Reengineering might be required, since scipy currently reads in
    the entire entry into memory.

    References
    ----------
    MTX format specification
    http://math.nist.gov/MatrixMarket/formats.html

    scipy matlab io
    http://docs.scipy.org/doc/scipy/reference/tutorial/io.html

    """

    FORMATS = ['mtx']
    MODES = ['r', 'w']

    def __init__(self, *args, **kwargs):
        FileIO.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode + 'b')

    def read(self, n=-1, sparse=False):
        """
        sparse: boolean
                if true, return pysal WSP object
                if false, return pysal W object
        """
        self._sparse = sparse
        self._complain_ifclosed(self.closed)
        return self._read()

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _read(self):
        """Reads MatrixMarket mtx file
        Returns a pysal.weights.weights.W or pysal.weights.weights.WSP object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open a MatrixMarket mtx file and read it into a pysal weights object

        >>> f = pysal.open(pysal.examples.get_path('wmat.mtx'),'r')

        >>> w = f.read()

        Get the number of observations from the header

        >>> w.n
        49

        Get the mean number of neighbors

        >>> w.mean_neighbors
        4.7346938775510203

        Get neighbor weights for a single observation

        >>> w[1]
        {2: 0.33329999999999999, 5: 0.33329999999999999, 6: 0.33329999999999999}

        >>> f.close()

        >>> f = pysal.open(pysal.examples.get_path('wmat.mtx'),'r')

        >>> wsp = f.read(sparse=True)

        Get the number of observations from the header

        >>> wsp.n
        49

        Get row from the weights matrix. Note that the first row in the sparse
        matrix (the 0th row) corresponds to ID 1 from the original mtx file
        read in.

        >>> print wsp.sparse[0].todense()
        [[ 0.      0.3333  0.      0.      0.3333  0.3333  0.      0.      0.      0.
           0.      0.      0.      0.      0.      0.      0.      0.      0.      0.
           0.      0.      0.      0.      0.      0.      0.      0.      0.      0.
           0.      0.      0.      0.      0.      0.      0.      0.      0.      0.
           0.      0.      0.      0.      0.      0.      0.      0.      0.    ]]

        """
        if self.pos > 0:
            raise StopIteration
        mtx = sio.mmread(self.file)
        ids = range(1, mtx.shape[0] + 1)  # matrix market indexes start at one
        wsp = WSP(mtx, ids)
        if self._sparse:
            w = wsp
        else:
            w = pysal.weights.WSP2W(wsp)
        self.pos += 1
        return w

    def write(self, obj):
        """

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        ------

        a MatrixMarket mtx file
        write a weights object to the opened mtx file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open(pysal.examples.get_path('wmat.mtx'),'r')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.mtx')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'w')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created mtx file

        >>> wnew =  pysal.open(fname,'r').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)

        Go to the beginning of the test file

        >>> testfile.seek(0)

        Create a sparse weights instance from the test file

        >>> wsp = testfile.read(sparse=True)

        Open the new file in write mode

        >>> o = pysal.open(fname,'w')

        Write the sparse weights object into the open file

        >>> o.write(wsp)
        >>> o.close()

        Read in the newly created mtx file

        >>> wsp_new =  pysal.open(fname,'r').read(sparse=True)

        Compare values from old to new

        >>> wsp_new.s0 == wsp.s0
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)

        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj), W) or issubclass(type(obj), WSP):
            w = obj.sparse
            sio.mmwrite(self.file, w, comment='Generated by PySAL',
                        field='real', precision=7)
            self.pos += 1
        else:
            raise TypeError("Expected a pysal weights object, got: %s" % (
                type(obj)))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)


