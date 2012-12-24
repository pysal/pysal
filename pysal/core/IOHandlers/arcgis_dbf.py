import pysal
import os.path
import pysal.core.FileIO as FileIO
from pysal.weights import W
from pysal.weights.util import remap_ids
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["ArcGISDbfIO"]


class ArcGISDbfIO(FileIO.FileIO):
    """
    Opens, reads, and writes weights file objects in ArcGIS dbf format.

    Spatial weights objects in the ArcGIS dbf format are used in
    ArcGIS Spatial Statistics tools.
    This format is the same as the general dbf format,
    but the structure of the weights dbf file is fixed unlike other dbf files.
    This dbf format can be used with the "Generate Spatial Weights Matrix" tool,
    but not with the tools under the "Mapping Clusters" category.

    The ArcGIS dbf file is assumed to have three or four data columns.
    When the file has four columns,
    the first column is meaningless and will be ignored in PySAL
    during both file reading and file writing.
    The next three columns hold origin IDs, destinations IDs, and weight values.
    When the file has three columns,
    it is assumed that only these data columns exist in the stated order.
    The name for the orgin IDs column should be the name of
    ID variable in the original source data table.
    The names for the destination IDs and weight values columns are NID
    and WEIGHT, respectively.
    ArcGIS Spatial Statistics tools support only unique integer IDs.
    Therefore, the values for origin and destination ID columns should
    be integer.
    For the case where the IDs of a weights object are not integers,
    ArcGISDbfIO allows users to use internal id values corresponding to
    record numbers, instead of original ids.

    An exemplary structure of an ArcGIS dbf file is as follows:
    [Line 1]    Field1    RECORD_ID    NID    WEIGHT
    [Line 2]    0         72           76     1
    [Line 3]    0         72           79     1
    [Line 4]    0         72           78     1
    ...

    Unlike the ArcGIS text format, this format does not seem to include self-neighbors.

    References
    ----------
    http://webhelp.esri.com/arcgisdesktop/9.3/index.cfm?TopicName=Convert_Spatial_Weights_Matrix_to_Table_(Spatial_Statistics)

    """

    FORMATS = ['arcgis_dbf']
    MODES = ['r', 'w']

    def __init__(self, *args, **kwargs):
        self._varName = 'Unknown'
        args = args[:2]
        FileIO.FileIO.__init__(self, *args, **kwargs)
        self.file = pysal.open(self.dataPath, self.mode)

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
        self.file.seek(pos)
        self.pos = self.file.pos

    def _read(self):
        """Reads ArcGIS dbf file
        Returns a pysal.weights.weights.W object

        Examples
        --------

        Type 'dir(w)' at the interpreter to see what methods are supported.
        Open an ArcGIS dbf file and read it into a pysal weights object

        >>> w = pysal.open(pysal.examples.get_path('arcgis_ohio.dbf'),'r','arcgis_dbf').read()

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

        id_var = self.file.header[1]
        startPos = len(self.file.header)

        if startPos == 3:
            startPos = 0
        elif startPos == 4:
            startPos = 1
        else:
            raise ValueError("Wrong structure, a weights dbf file requires at least three data columns")

        self.varName = id_var
        id_type = int
        id_spec = self.file.field_spec[startPos]
        if id_spec[0] != 'N':
            raise TypeError('The data type for ids should be integer.')
        self.id_var = id_var

        weights = {}
        neighbors = {}
        for row in self.file:
            i, j, w = tuple(row)[startPos:]
            i = id_type(i)
            j = id_type(j)
            w = float(w)
            if i not in weights:
                weights[i] = []
                neighbors[i] = []
            weights[i].append(w)
            neighbors[i].append(j)
            self.pos = self.file.pos

        return W(neighbors, weights)

    def write(self, obj, useIdIndex=False):
        """

        Parameters
        ----------
        .write(weightsObject)
        accepts a weights object

        Returns
        ------

        an ArcGIS dbf file
        write a weights object to the opened dbf file.

        Examples
        --------

        >>> import tempfile, pysal, os
        >>> testfile = pysal.open(pysal.examples.get_path('arcgis_ohio.dbf'),'r','arcgis_dbf')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.dbf')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'w','arcgis_dbf')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created text file

        >>> wnew =  pysal.open(fname,'r','arcgis_dbf').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)

        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj), W):
            self.file.header = [self.varName, 'NID', 'WEIGHT']

            id_type = type(obj.id_order[0])
            if id_type is not int and not useIdIndex:
                raise TypeError("ArcGIS DBF weight files support only integer IDs")
            if useIdIndex:
                id2i = obj.id2i
                obj = remap_ids(obj, id2i)

            id_spec = ('N', len(str(max(obj.id_order))), 0)
            self.file.field_spec = [id_spec, id_spec, ('N', 13, 6)]

            for id in obj.id_order:
                neighbors = zip(obj.neighbors[id], obj.weights[id])
                for neighbor, weight in neighbors:
                    self.file.write([id, neighbor, weight])
                    self.pos = self.file.pos

        else:
            raise TypeError("Expected a pysal weights object, got: %s" % (
                type(obj)))

    def flush(self):
        self._complain_ifclosed(self.closed)
        self.file.flush()

    def close(self):
        self.file.close()

