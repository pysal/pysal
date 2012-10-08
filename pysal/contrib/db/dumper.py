#!/usr/bin/python
"""
dumps postgis database table to shapefile 
"""
__author__ = "Philip Stephens <philip.stephens@asu.edu "

__all__ = ['db2shape']

from osgeo import ogr
def db2shape(connstring, input, output):
    conn = ogr.Open(connstring)

    # Schema definition of SHP file
    out_driver = ogr.GetDriverByName( 'ESRI Shapefile' )
    out_ds = out_driver.CreateDataSource(output)
    out_srs = None
    out_layer = out_ds.CreateLayer("point", out_srs, ogr.wkbPoint)
    fd = ogr.FieldDefn('name',ogr.OFTString)
    out_layer.CreateField(fd)

    layer = conn.GetLayerByName(input)
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

    connstring = "PG: host='localhost' dbname='pysaldb' user='stephens' password='coorhall'"
    input = 'burkitt'
    output = "dumptest.shp"
    db2shape(connstring, input, output)
