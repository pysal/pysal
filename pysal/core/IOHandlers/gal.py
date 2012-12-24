import pysal.core.FileIO as FileIO
from pysal.weights import W, WSP
from scipy import sparse
import numpy as np

__author__ = 'Charles R Schmidt <schmidtc@gmail.com>'
__all__ = ['GalIO']


class GalIO(FileIO.FileIO):
    """
    Opens, reads, and writes file objects in GAL format.


    """
    FORMATS = ['gal']
    MODES = ['r', 'w']

    def __init__(self, *args, **kwargs):
        self._typ = str
        FileIO.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode)

    def read(self, n=-1, sparse=False):
        """

        sparse: boolean
               If true return scipy sparse object
               If false return pysal w object
        """
        self._sparse = sparse
        self._complain_ifclosed(self.closed)
        return self._read()

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _get_data_type(self):
        return self._typ

    def _set_data_type(self, typ):
        if callable(typ):
            self._typ = typ
        else:
            raise TypeError("Expecting a callable")
    data_type = property(fset=_set_data_type, fget=_get_data_type)

    def _read(self):
        """
        Parameters
        ----------
        reads in a GalIO object

        Returns
        -------
        returns a W object

        Examples
        --------

        >>> import tempfile, pysal, os

        Read in a file GAL file

        >>> testfile = pysal.open(pysal.examples.get_path('sids2.gal'),'r')

        Return a W object

        >>> w = testfile.read()
        >>> w.n == 100
        True
        >>> w.sd == 1.5151237573214935
        True
        >>> testfile = pysal.open(pysal.examples.get_path('sids2.gal'),'r')

        Return a sparse matrix for the w information

        >>> wsp = testfile.read(sparse=True)
        >>> wsp.sparse.nnz
        462

        """
        if self._sparse:
            if self.pos > 0:
                raise StopIteration

            header = self.file.readline().strip().split()
            header_n = len(header)
            n = int(header[0])
            if header_n > 1:
                n = int(header[1])
            ids = []
            idsappend = ids.append
            row = []
            extend = row.extend    # avoid dot in loops
            col = []
            append = col.append
            counter = 0
            typ = self.data_type
            for i in xrange(n):
                id, n_neighbors = self.file.readline().strip().split()
                id = typ(id)
                n_neighbors = int(n_neighbors)
                neighbors_i = map(typ, self.file.readline().strip().split())
                nn = len(neighbors_i)
                extend([id] * nn)
                counter += nn
                for id_neigh in neighbors_i:
                    append(id_neigh)
                idsappend(id)
            self.pos += 1
            row = np.array(row)
            col = np.array(col)
            data = np.ones(counter)
            ids = np.unique(row)
            row = np.array([np.where(ids == j)[0] for j in row]).flatten()
            col = np.array([np.where(ids == j)[0] for j in col]).flatten()
            spmat = sparse.csr_matrix((data, (row, col)), shape=(n, n))
            return WSP(spmat)

        else:
            if self.pos > 0:
                raise StopIteration
            neighbors = {}
            ids = []
            # handle case where more than n is specified in first line
            header = self.file.readline().strip().split()
            header_n = len(header)
            n = int(header[0])
            if header_n > 1:
                n = int(header[1])
            w = {}
            typ = self.data_type
            for i in range(n):
                id, n_neighbors = self.file.readline().strip().split()
                id = typ(id)
                n_neighbors = int(n_neighbors)
                neighbors_i = map(typ, self.file.readline().strip().split())
                neighbors[id] = neighbors_i
                ids.append(id)
            self.pos += 1
            return W(neighbors, id_order=ids)

    def write(self, obj):
        """

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        ------

        a GAL file
        write a weights object to the opened GAL file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open(pysal.examples.get_path('sids2.gal'),'r')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.gal')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'w')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created gal file

        >>> wnew =  pysal.open(fname,'r').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)
        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj), W):
            IDS = obj.id_order
            self.file.write('%d\n' % (obj.n))
            for id in IDS:
                neighbors = obj.neighbors[id]
                self.file.write('%s %d\n' % (str(id), len(neighbors)))
                self.file.write(' '.join(map(str, neighbors)) + '\n')
            self.pos += 1
        else:
            raise TypeError("Expected a pysal weights object, got: %s" %
                            (type(obj)))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)


