import clusterpy as _clusterpy
import pysal as ps
import struct

__author__ = "Sergio Rey <sjsrey@gmail.com>"

__ALL__= ['Layer', 'loadArcData', 'importCsvData', 'addRook2Layer', 'addQueen2Layer', 'addArray2Layer', 'addW2Layer']


def _importArcData(filename):
    """Creates a new Layer from a shapefile (<file>.shp)

    This function wraps and extends a core clusterPy function to utilize PySAL
    W constructors and dbf readers.


    Parameters
    ==========

    filename: string
              suffix of shapefile (fileName not fileName.shp)


    Returns
    =======
    layer: clusterpy layer instance



    """
    layer = _clusterpy.Layer()
    layer.name = filename.split('/')[-1]
    #print "Loading " + filename + ".dbf"
    dbf = ps.open(filename+".dbf")
    fields = dbf.header
    #data, fields, specs = importDBF(filename + '.dbf')
    data = {}
    #print "Loading " + filename + ".shp"
    if fields[0] != "ID":
        fields = ["ID"] + fields
        for y in range(dbf.n_records):
            data[y] = [y] + dbf.by_row(y)
    else:
        for y in range(dbf.n_records):
            data[y] = dbf.by_row_(y)

    layer.fieldNames = fields
    layer.Y = data
    shpf = filename+".shp"
    layer.shpType = 5
    #print 'pysal reader'
    layer.Wrook = ps.rook_from_shapefile(filename+".shp").neighbors
    layer.Wqueen = ps.queen_from_shapefile(filename+".shp").neighbors
    #print "Done"
    return layer

_clusterpy.importArcData = _importArcData

################# Public functions #######################

def Layer():
    """Provide a clusterpy Layer instance

    Parameters
    ==========

    none

    Returns
    =======

    layer: clusterpy.Layer instance

    Examples
    ========
    >>> import pysal.contrib.clusterpy as cp
    >>> l = cp.Layer()
    >>> type(l)
    <type 'instance'>
    >>> l.Wrook
    {}

    """
    return _clusterpy.Layer()

def loadArcData(shapeFileName):
    """
    Handler to use PySAL W and dbf readers in place of clusterpy's

    Parameters
    ==========
    shapeFileName: string
                   filename including .shp extension

    Returns
    =======
    layer: clusterpy layer instance

    

    Examples
    ========
    >>> import pysal.contrib.clusterpy as cp
    >>> import pysal as ps
    >>> shpFile = ps.examples.get_path('columbus.shp')
    >>> columbus = cp.loadArcData(shpFile)
    >>> columbus.Wrook[0]
    [1, 2]
    >>> columbus.Wrook[1]
    [0, 2, 3]
    >>> columbus.fieldNames[0:10]
    ['ID', 'AREA', 'PERIMETER', 'COLUMBUS_', 'COLUMBUS_I', 'POLYID', 'NEIG', 'HOVAL', 'INC', 'CRIME']
    """
    base = shapeFileName.split(".")[0]
    return _clusterpy.importArcData(base)

def importCsvData(filename, layer=None):
    """
    Read a csv file of attributes into a layer

    Parameters
    ==========

    filename: string
              csf file to load

    layer: clusterpy layer instance (default: None)
           if a layer is passed, new attributes, Ws are attached to the layer.
               Otherwise a new layer is created and returned
           


    Returns
    =======
    layer: clusterpy layer instance


    Examples
    ========
    >>> import pysal.contrib.clusterpy as cp
    >>> l = cp.Layer()
    >>> mexico = cp.importCsvData(ps.examples.get_path('mexico.csv'))
    >>> mexico.fieldNames
    ['ID', 'State', 'pcgdp1940', 'pcgdp1950', 'pcgdp1960', 'pcgdp1970', 'pcgdp1980', 'pcgdp1990', 'pcgdp2000', 'hanson03', 'hanson98', 'esquivel99', 'inegi', 'inegi2']

    Notes
    =====

    This assumes the csv file is organized with records on the rows and attributes on the columns


    """

    if not layer:
        layer = Layer()
    csv = ps.open(filename,'r')
    fields = csv.header
    data = {}
    if fields[0] != "ID":
        fields = ["ID"] + fields
        for i, rec in enumerate(csv.data):
            data[i] = [i] + csv.by_row(i)
    else:
        for i, rec in enumerate(csv.data):
            data[i] = csv.by_row(i) 
    layer.Y = data
    layer.fieldNames = fields
    return layer

def addGal2Layer(galfile, layer, contiguity='rook'):
    """
    Attach an adjacency object to a layer

    Parameters
    ==========
    galfile: string
             galfile

    layer: clusterpy layer

    contiguity: type of contguity ['rook'|'queen']


    Returns
    =======
    None

    Examples
    ========
    >>> import pysal as ps
    >>> import pysal.contrib.clusterpy as cp
    >>> csvfile = ps.examples.get_path('mexico.csv')
    >>> galfile = ps.examples.get_path('mexico.gal')
    >>> mexico = cp.importCsvData(csvfile)
    >>> cp.addRook2Layer(galfile, mexico)
    >>> mexico.Wrook[0]
    [31, 13]


    """
    gal = ps.open(galfile).read().neighbors
    w = {}
    for key in gal:
        w[int(key)] =  map(int, gal[key]) 
    
    if contiguity.upper()== "ROOK":
        layer.Wrook = w
    elif contiguity.upper() == "QUEEN":
        layer.Wqueen = w
    else:
        print 'Unsupported contiguity type: ', contiguity

def addRook2Layer(galfile, layer):
    """
    User function for adding rook to layer

    See addGal2Layer
    """
    addGal2Layer(galfile, layer)

def addQueen2Layer(galfile, layer):
    """
    User function for adding queen to layer

    See addGal2Layer
    """
    addGal2Layer(galfile, layer, contiguity='QUEEN')

def addArray2Layer(array, layer, names=None):
    """
    Add a numpy array to a clusterpy layer


    Parameters
    ==========
    array: nd-array
           nxk with n observations on k attributes

    layer: clusterpy layer object

    names: list
           k strings for attribute names

    Returns
    =======
    None

    Examples
    ========
    # Note this will report as fail since clusterpy prints 'Adding variables
    # for each variable added. But the variables will be correctly added
    >>> #import pysal as ps
    >>> #import pysal.contrib.clusterpy as cp
    >>> #import numpy as np
    >>> #uscsv = ps.examples.get_path("usjoin.csv")
    >>> #f = ps.open(uscsv)
    >>> #pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)]).T 
    >>> #usy = cp.Layer()
    >>> #names = ["Y_%d"%v for v in range(1929,2010)]
    >>> #cp.addArray2Layer(pci, usy, names)

    """
    n,k = array.shape
    if not names:
        names = ["X_%d"% v for v in range(k)]
        
    for j,name in enumerate(names):
        v = {}
        for i in xrange(n):
            v[i] = array[i,j]
        layer.addVariable([name], v)

def addW2Layer(w, layer, contiguity='rook'):
    '''
    Attach a contiguity PySAL W object to a layer
    
    NOTE: given clusterpy's requirements, this method only extracts the
    `neighbors` dictionary.

    ...

    Parameters
    ----------
    w         : ps.W
                PySAL weights object
    layer     : clusterpy.Layer
                Layer to attach the weights to
    contiguity: str ['rook'|'queen']
                Type of contguity expressed in `w`

    Returns
    -------
    None

    Example
    -------
    >>> import pysal as ps
    >>> import pysal.contrib.clusterpy as cp
    >>> w = ps.queen_from_shapefile(ps.examples.get_path('columbus.shp'))
    >>> layer = cp.Layer()
    >>> cp.addW2Layer(w, layer, contiguity='queen')
    >>> layer.Wqueen[0]
    [1, 2]
    '''
    if contiguity.upper()== "ROOK":
        layer.Wrook = w.neighbors
    elif contiguity.upper() == "QUEEN":
        layer.Wqueen = w.neighbors
    else:
        print 'Unsupported contiguity type: ', contiguity
    return None

if __name__ == '__main__':

    import numpy as np

    w = ps.queen_from_shapefile(ps.examples.get_path('columbus.shp'))
    db = ps.open(ps.examples.get_path('columbus.dbf'))
    vars = ['CRIME',  'HOVAL']
    x = np.array([db.by_col(i) for i in vars]).T
    layer = Layer()
    _ = addArray2Layer(x, layer, names=vars)
    _ = addW2Layer(w, layer)
    layer.cluster('arisel', ['CRIME',  'CONSTANT'], 2, dissolve=0, std=0)
    '''
    columbus = loadArcData(ps.examples.get_path('columbus.shp'))
    n = len(columbus.Wqueen)
    columbus.dataOperation("CONSTANT = 1")
    np.random.seed(12345)
    columbus.cluster('maxpTabu', ['CRIME',  'CONSTANT'], threshold=4, dissolve=0, std=0)
    '''
    #np.random.seed(12345)
    #columbus.cluster('arisel', ['CRIME'], 5, wType='rook', inits=10, dissolve=0)

