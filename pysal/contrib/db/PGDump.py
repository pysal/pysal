#!/usr/bin/python
__author__ = "Philip Stephens <philip.stephens@asu.edu "

__all__ = ['db2shape']

from django.contrib.gis.gdal.geomtype import OGRGeomType
from osgeo import ogr
import os

def db2shape(connstring, input, output):
    """
    dumps a postgis database table to shapefile 

    Arguments
    ---------
    connstring = A connection string with 4 parameters. Example is 
    "PG: host='localhost' dbname='pysaldb' user='myusername'
    password='my_password'"

    input : the db table

    output : a filename

    Note
    ----

    If a file exists with the same name as 'output', it will be deleted
    before being overwritten.
    """
    conn = ogr.Open(connstring)
    layer = conn.GetLayerByName(input)
    type = layer.GetGeomType()  # returns an int
    geom_type = OGRGeomType._types[type]  # map int to string description

    # Schema definition of SHP file
    out_driver = ogr.GetDriverByName( 'ESRI Shapefile' )
    if os.path.exists(output):
        out_driver.DeleteDataSource(output)
    out_ds = out_driver.CreateDataSource(output)
    out_srs = None
    out_layer = out_ds.CreateLayer(geom_type, out_srs, type)
    fd = ogr.FieldDefn('name',ogr.OFTString)
    out_layer.CreateField(fd)

    #layer = conn.ExecuteSQL(sql)

    feat = layer.GetNextFeature()
    while feat is not None:
        featDef = ogr.Feature(out_layer.GetLayerDefn())
        featDef.SetGeometry(feat.GetGeometryRef())
        #featDef.SetField('name',feat.TITLE)
        out_layer.CreateFeature(featDef)
        feat.Destroy()
        feat = layer.GetNextFeature()
    conn.Destroy()
    out_ds.Destroy()

if __name__ == '__main__':
    import nose
    nose.runmodule()
