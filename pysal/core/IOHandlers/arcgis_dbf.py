import pysal
import os.path
import pysal.core.FileIO as FileIO
from pysal.weights import W
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
    MODES = ['rw', 'ww']

    def __init__(self, *args, **kwargs):
        self._varName = 'Unknown'
        FileIO.FileIO.__init__(self, *args, **kwargs)
        self.file = pysal.open(self.dataPath, self.mode[0])

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

        >>> w = pysal.open('../../examples/arcgis_ohio.dbf','rw').read()

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
            raise ValueError, "Wrong structure, a weights dbf file requires at least three data columns"

        self.varName = id_var
        id_type = str
        id_spec = self.file.field_spec[startPos]
        if id_spec[0] in ['N', 'F']:
            if id_spec[2] == 0:
                id_type = int
            else:
                id_type = float

        self.id_var = id_var
        
        weights = {}
        neighbors = {}
        for row in self.file:
            i,j,w = tuple(row)[startPos:]
            i = id_type(i)
            j = id_type(j)
            w = float(w)
            if i not in weights:
                weights[i] = []
                neighbors[i] = []
            weights[i].append(w)
            neighbors[i].append(j)
            self.pos = self.file.pos

        return W(neighbors,weights)

    def write(self, obj):
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
        >>> testfile = pysal.open('../../examples/arcgis_ohio.dbf','rw')
        >>> w = testfile.read()

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.dbf')

        Reassign to new var

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Open the new file in write mode

        >>> o = pysal.open(fname,'ww')

        Write the Weights object into the open file

        >>> o.write(w)
        >>> o.close()

        Read in the newly created text file

        >>> wnew =  pysal.open(fname,'rw').read()

        Compare values from old to new

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example

        >>> os.remove(fname)

        """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj),W):
            self.file.header = [self.varName, 'NID', 'WEIGHT']

            id_type = type(obj.id_order[0])
            id_precision = 0
            keys = [(len(str(id)), str(id)) for id in obj.id_order]
            keys.sort()
            id_len = keys[-1][0]
            if id_type == str:
                id_type = 'C'
            elif id_type in [int, float]:
                id_type = 'N'
                digits = keys[-1][1].split('.')
                if len(digits) == 2:
                    id_precision = len(digits[1])
            id_spec = (id_type, id_len, id_precision)
            self.file.field_spec = [id_spec, id_spec, ('N', 13, 6)]

            for id in obj.id_order:
                neighbors = zip(obj.neighbors[id], obj.weights[id])
                for neighbor, weight in neighbors:
                    self.file.write([id, neighbor, weight])
                    self.pos = self.file.pos

        else:
            raise TypeError, "Expected a pysal weights object, got: %s" % (type(obj))

    def flush(self):
        self._complain_ifclosed(self.closed)
        self.file.flush()

    def close(self):
        self.file.close()

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
