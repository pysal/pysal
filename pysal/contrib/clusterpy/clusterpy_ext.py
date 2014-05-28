import clusterpy as _clusterpy
import pysal as ps
import struct

__ALL__= ['Layer', 'loadArcData', 'importCsvData', 'addRook2Layer', 'addQueen2Layer', 'addArray2Layer' ]


def importArcData(filename):
    """Creates a new Layer from a shapefile (<file>.shp)

    modification of original function fro clusterPy to use PySAL W constructor
    
    :param filename: filename without extension 

    **Description**

    `ESRI <http://www.esri.com/>`_ shapefile is a binary file used to
    save and transport maps. During the last times it has become
    the most used format for the spatial scientists around the world.

    On clusterPy's "data_examples" folder you can find some shapefiles. To
    load a shapefile in clusterPy just follow the example bellow.

    **Example** ::

        import clusterpy
        china = clusterpy.importArcData("clusterpy/data_examples/china")

    """
    layer = _clusterpy.Layer()
    layer.name = filename.split('/')[-1]
    print "Loading " + filename + ".dbf"
    dbf = ps.open(filename+".dbf")
    fields = dbf.header
    #data, fields, specs = importDBF(filename + '.dbf')
    data = {}
    print "Loading " + filename + ".shp"
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
    print 'pysal reader'
    layer.Wrook = ps.rook_from_shapefile(filename+".shp").neighbors
    layer.Wqueen = ps.queen_from_shapefile(filename+".shp").neighbors
    print "Done"
    return layer

_clusterpy.importArcData = importArcData

################# Public functions #######################

def Layer():
    return _clusterpy.Layer()

def loadArcData(shapeFileName):
    base = shapeFileName.split(".")[0]
    return _clusterpy.importArcData(base)

def importCsvData(filename, layer=None):

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
    addGal2Layer(galfile, layer)

def addQueen2Layer(galfile, layer):
    addGal2Layer(galfile, layer, contiguity='QUEEN')

def addArray2Layer(array, layer, names=None):
    n,k = array.shape
    if not names:
        names = ["X_%d"% v for v in range(k)]
        
    for j,name in enumerate(names):
        v = {}
        for i in xrange(n):
            v[i] = array[i,j]
        layer.addVariable([name], v)


if __name__ == '__main__':

    columbus = loadArcData(ps.examples.get_path('columbus.shp'))
    n = len(columbus.Wqueen)
    columbus.generateData('Uniform', 'rook', 1, 1, 10)
    columbus.cluster('maxpTabu', ['CRIME',  'Uniform21'], threshold=4, dissolve=0, std=0)
    columbus.cluster('arisel', ['CRIME'], 5, wType='rook', inits=10, dissolve=0)

