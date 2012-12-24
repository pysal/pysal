import pysal
import os.path
import gwt
from pysal.weights import W
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["DatIO"]


class DatIO(gwt.GwtIO):
    """
    Opens, reads, and writes file objects in DAT format.

    Spatial weights objects in DAT format are used in
    Dr. LeSage's MatLab Econ library.
    This DAT format is a simple text file with DAT or dat extension.
    Without header line, it includes three data columns
    for origin id, destination id, and weight values as follows:

    [Line 1]    2    1    0.25
    [Line 2]    5    1    0.50
    ...

    Origin/destination IDs in this file format are simply record
    numbers starting with 1. IDs are not necessarily integers.
    Data values for all columns should be numeric.

    """

    FORMATS = ['dat']
    MODES = ['r', 'w']

    def _read(self):
        """Reads .dat file
        Returns a pysal.weights.weights.W object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open .dat file and read it into a pysal weights object

        >>> w = pysal.open(pysal.examples.get_path('wmat.dat'),'r').read()

        Get the number of observations from the header

        >>> w.n
        49

        Get the mean number of neighbors

        >>> w.mean_neighbors
        4.7346938775510203

        Get neighbor distances for a single observation

        >>> w[1]
        {2.0: 0.3333, 5.0: 0.3333, 6.0: 0.3333}

        """
        if self.pos > 0:
            raise StopIteration

        id_type = float
        weights, neighbors = self._readlines(id_type)

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

        a DAT file
        write a weights object to the opened DAT file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open(pysal.examples.get_path('wmat.dat'),'r')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.dat')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'w')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created dat file

        >>> wnew =  pysal.open(fname,'r').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)
        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj), W):
            self._writelines(obj)
        else:
            raise TypeError("Expected a pysal weights object, got: %s" % (
                type(obj)))

