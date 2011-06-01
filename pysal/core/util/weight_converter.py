import os
import pysal

class WeightConverter(object):

    """
    Open and reads a weights file in a format. 
    Then, writes the file in other formats.
    
    WeightConverter can read a weights file in the following formats:
    GAL, GWT, ArcGIS DBF/SWM/Text, DAT, MAT, MTX, WK1, GeoBUGS Text, and STATA Text.
    It can convert the input file into all of the formats listed above, except GWT.
    Currently, PySAL does not support writing a weights object in the GWT format.

    When an input weight file includes multiple islands and 
    the format of an output weight file is ArcGIS DBF/SWM/TEXT, DAT, or WK1,
    the number of observations in the new weights file will be 
    the original number of observations substracted by the number of islands.
    This is because ArcGIS DBF/SWM/TEXT, DAT, WK1 formats ignore islands.

    """

    def __init__(self, inputPath='', dataFormat=None):
        self.inputPath = inputPath
        self.inputDataFormat = dataFormat
        self._setW()

    def _setW(self):
        """
        Reads a weights file and sets a pysal.weights.weights.W object as an attribute

        Examples
        --------
        
        Create a WeightConvert object

        >>> wc = WeightConverter('../../examples/arcgis_ohio.dbf',dataFormat='arcgis_dbf')

        Check whether or not the W object is set as an attribute

        >>> wc.w_set()
        True

        Get the number of observations included in the W object
        
        >>> wc.w.n
        88

        """
        try:
            if self.inputDataFormat:
                f = pysal.open(self.inputPath, 'r', self.inputDataFormat)
            else:
                f = pysal.open(self.inputPath, 'r')
        except:
            raise IOError, 'A problem occurred while reading the input file.'
        else:
            try:
                self.w = f.read()
            except:
                raise RuntimeError, 'A problem occurred while creating a weights object.'
            finally:
                f.close()

    def w_set(self):
        """
        Checks if a source w object is set
        """
        return hasattr(self, 'w')

    def write(self, outputPath, **kwargs):
        """
        Parameters 
        ----------
        outputPath: string
                    path to the output weights file
        kwargs: dictionary
                other options for writing the output file
                Available options are as follows:
                dataFormat: string
                            'arcgis_dbf' for ArcGIS DBF format
                            'arcgis_text' for ArcGIS Text format
                            'geobugs_text' for GeoBUGS Text format
                            'stata_text' for STATA Text format
                useIdIndex: True or False 
                            ArcGIS DBF/SWM/Text formats
                matrix_form: True or False
                             STATA Text format

        Returns
        -------
        A weights file is created

        Examples
        --------
        >>> import tempfile, os, pysal
 
        Create a WeightConverter object

        >>> wc = WeightConverter('../../examples/sids2.gal')

        Check whether or not the W object is set as an attribute

        >>> wc.w_set()
        True

        Create a temporary file for this example

        >>> f = tempfile.NamedTemporaryFile(suffix='.dbf')
  
        Reassign to new variable

        >>> fname = f.name

        Close the temporary named file

        >>> f.close()

        Write the input gal file in the ArcGIS dbf format

        >>> wc.write(fname, dataFormat='arcgis_dbf', useIdIndex=True)                

        Create a new weights object from the converted dbf file

        >>> wnew = pysal.open(fname, 'r', 'arcgis_dbf').read()

        Compare the number of observations in two W objects

        >>> wc.w.n == wnew.n
        True
       
        Clean up the temporary file

        >>> os.remove(fname)

        """
        ext = os.path.splitext(outputPath)[1]
        ext = ext.replace('.', '')
        if ext.lower() == 'gwt':
            raise TypeError, 'Currently, PySAL does not support writing a weights object into a gwt file.'
        
        if not self.w_set():
            raise RuntimeError, 'There is no weights object to write out.'

        try:
            if kwargs.has_key('dataFormat'):
                dataFormat = kwargs['dataFormat']
                del kwargs['dataFormat']
                o = pysal.open(outputPath, 'w', dataFormat)
            else:
                o = pysal.open(outputPath, 'w')
        except:
            raise IOError, 'A problem occurred while creating the output file.'
        else:
            try:
                if len(kwargs) == 0:
                    o.write(self.w)
                else:
                    o.write(self.w, **kwargs)
            except:
                raise RuntimeError, 'A problem occurred while writing out the weights object'
            finally:
                o.close()

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
  
            
    
    
