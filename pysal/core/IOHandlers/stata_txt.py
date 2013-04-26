import pysal
import os.path
import pysal.core.FileIO as FileIO
from pysal.weights import W
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["StataTextIO"]


class StataTextIO(FileIO.FileIO):
    """
    Opens, reads, and writes weights file objects in STATA text format.

    Spatial weights objects in the STATA text format are used in
    STATA sppack library through the spmat command.
    This format is a simple text file delimited by a whitespace.
    The spmat command does not specify which file extension to use.
    But, txt seems the default file extension, which is assumed in PySAL.

    The first line of the STATA text file  is
    a header including the number of observations.
    After this header line, it includes at least one data column that contains
    unique ids or record numbers of observations.
    When an id variable is not specified for the original spatial weights
    matrix in STATA, record numbers are used to identify individual observations,
    and the record numbers start with one.
    The spmat command seems to allow only integer IDs,
    which is also assumed in PySAL.

    A STATA text file can have one of the following structures according to
    its export options in STATA.

    Structure 1: encoding using the list of neighbor ids
    [Line 1]    [Number_of_Observations]
    [Line 2]    [ID_of_Obs_1] [ID_of_Neighbor_1_of_Obs_1] [ID_of_Neighbor_2_of_Obs_1] .... [ID_of_Neighbor_m_of_Obs_1]
    [Line 3]    [ID_of_Obs_2]
    [Line 4]    [ID_of_Obs_3] [ID_of_Neighbor_1_of_Obs_3] [ID_of_Neighbor_2_of_Obs_3]
    ...
    Note that for island observations their IDs are still recorded.

    Structure 2: encoding using a full matrix format
    [Line 1]    [Number_of_Observations]
    [Line 2]    [ID_of_Obs_1] [w_11] [w_12] ... [w_1n]
    [Line 3]    [ID_of_Obs_2] [w_21] [w_22] ... [w_2n]
    [Line 4]    [ID_of_Obs_3] [w_31] [w_32] ... [w_3n]
    ...
    [Line n+1]  [ID_of_Obs_n] [w_n1] [w_n2] ... [w_nn]
    where w_ij can be a form of general weight.
    That is, w_ij can be both a binary value or a general numeric value.
    If an observation is an island, all of its w columns contains 0.

    References
    ----------
    Drukker D.M., Peng H., Prucha I.R., and Raciborski R. (2011)
    "Creating and managing spatial-weighting matrices using the spmat command"

    Notes
    -----
    The spmat command allows users to add any note to a spatial weights matrix object in STATA.
    However, all those notes are lost when the matrix is exported.
    PySAL also does not take care of those notes.

    """

    FORMATS = ['stata_text']
    MODES = ['r', 'w']

    def __init__(self, *args, **kwargs):
        args = args[:2]
        FileIO.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode)

    def read(self, n=-1):
        self._complain_ifclosed(self.closed)
        return self._read()

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _read(self):
        """Reads STATA Text file
        Returns a pysal.weights.weights.W object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open a text file and read it into a pysal weights object

        >>> w = pysal.open(pysal.examples.get_path('stata_sparse.txt'),'r','stata_text').read()
        WARNING: there are 7 disconnected observations
        Island ids:  [5, 9, 10, 11, 12, 14, 15]

        Get the number of observations from the header

        >>> w.n
        56

        Get the mean number of neighbors

        >>> w.mean_neighbors
        4.0

        Get neighbor distances for a single observation

        >>> w[1]
        {53: 1.0, 51: 1.0, 45: 1.0, 54: 1.0, 7: 1.0}

        """
        if self.pos > 0:
            raise StopIteration

        n = int(self.file.readline().strip())
        line1 = self.file.readline().strip()
        obs_01 = line1.split(' ')
        matrix_form = False
        if len(obs_01) == 1 or float(obs_01[1]) != 0.0:
            def line2wgt(line):
                row = [int(i) for i in line.strip().split(' ')]
                return row[0], row[1:], [1.0] * len(row[1:])
        else:
            matrix_form = True

            def line2wgt(line):
                row = line.strip().split(' ')
                obs = int(float(row[0]))
                ngh, wgt = [], []
                for i in range(n):
                    w = float(row[i + 1])
                    if w > 0:
                        ngh.append(i)
                        wgt.append(w)
                return obs, ngh, wgt

        id_order = []
        weights, neighbors = {}, {}
        l = line1
        for i in range(n):
            obs, ngh, wgt = line2wgt(l)
            id_order.append(obs)
            neighbors[obs] = ngh
            weights[obs] = wgt
            l = self.file.readline()
        if matrix_form:
            for obs in neighbors:
                neighbors[obs] = [id_order[ngh] for ngh in neighbors[obs]]

        self.pos += 1
        return W(neighbors, weights)

    def write(self, obj, matrix_form=False):
        """

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        ------

        a STATA text file
        write a weights object to the opened text file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open(pysal.examples.get_path('stata_sparse.txt'),'r','stata_text')
        >>> w = testfile.read()
        WARNING: there are 7 disconnected observations
        Island ids:  [5, 9, 10, 11, 12, 14, 15]

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.txt')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'w','stata_text')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created text file

        >>> wnew =  pysal.open(fname,'r','stata_text').read()
        WARNING: there are 7 disconnected observations
        Island ids:  [5, 9, 10, 11, 12, 14, 15]

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)
        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj), W):
            header = '%s\n' % obj.n
            self.file.write(header)
            if matrix_form:
                def wgt2line(obs_id, neighbor, weight):
                    w = ['0.0'] * obj.n
                    for ngh, wgt in zip(neighbor, weight):
                        w[obj.id2i[ngh]] = str(wgt)
                    return [str(obs_id)] + w
            else:
                def wgt2line(obs_id, neighbor, weight):
                    return [str(obs_id)] + [str(ngh) for ngh in neighbor]
            for id in obj.id_order:
                line = wgt2line(id, obj.neighbors[id], obj.weights[id])
                self.file.write('%s\n' % ' '.join(line))
        else:
            raise TypeError("Expected a pysal weights object, got: %s" % (
                type(obj)))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)

