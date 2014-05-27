import clusterpy

import clusterpy.core.inputs
from clusterpy.core.inputs import readPoints, readPolylines, readPolygons
import pysal as ps
import struct

def importArcData(filename):
    """Creates a new Layer from a shapefile (<file>.shp)

    overridden by pysal
    
    :param filename: filename without extension 
    :type filename: string
    :rtype: Layer (CP project)

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
    layer = clusterpy.Layer()
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

clusterpy.importArcData = importArcData
