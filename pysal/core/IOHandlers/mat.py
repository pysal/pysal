import pysal
import os.path
import scipy.io as sio
import pysal.core.FileIO as FileIO
from pysal.weights import W
from pysal.weights.util import full, full2W
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["MatIO"]


class MatIO(FileIO.FileIO):
    """
    Opens, reads, and writes weights file objects in MATLAB Level 4-5 MAT format.

    MAT files are used in Dr. LeSage's MATLAB Econometrics library.
    The MAT file format can handle both full and sparse matrices,
    and it allows for a matrix dimension greater than 256.
    In PySAL, row and column headers of a MATLAB array are ignored.

    PySAL uses matlab io tools in scipy.
    Thus, it is subject to all limits that loadmat and savemat in scipy have.

    Notes
    -----
    If a given weights object contains too many observations to
    write it out as a full matrix,
    PySAL writes out the object as a sparse matrix.

    References
    ----------
    MathWorks (2011) "MATLAB 7 MAT-File Format" at
    http://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf.

    scipy matlab io
    http://docs.scipy.org/doc/scipy/reference/tutorial/io.html

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

        >>> w = pysal.open(pysal.examples.get_path('spat-sym-us.mat'),'r').read()

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

        mat = sio.loadmat(self.file)
        mat_keys = [k for k in mat if not k.startswith("_")]
        full_w = mat[mat_keys[0]]

        self.pos += 1
        return full2W(full_w)

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
        >>> testfile = pysal.open(pysal.examples.get_path('spat-sym-us.mat'),'r')
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
        if issubclass(type(obj), W):
            try:
                w = full(obj)[0]
            except ValueError:
                w = obj.sparse
            sio.savemat(self.file, {'WEIGHT': w})
            self.pos += 1
        else:
            raise TypeError("Expected a pysal weights object, got: %s" % (
                type(obj)))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)

