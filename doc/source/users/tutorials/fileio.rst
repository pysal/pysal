.. _fileio:

.. testsetup:: *
        
        import pysal
        import numpy as np

******************************************
An Overview of the FileIO system in PySAL.
******************************************

.. contents::

Introduction
============

PySAL contains a new file input-output API that should be used for all file IO
operations. The goal is to abstract file handling and return native PySAL data
types when reading from known file types. A list of known extensions can be
found by issuing the following command::

    pysal.open.check()

Note that in some cases the FileIO module will peek inside your file to
determine its type. For example "geoda_txt" is just a unique scheme for ".txt"
files, so when opening a ".txt" pysal will peek inside the file to determine it
if has the necessary header information and dispatch accordingly. In the event
that pysal does not understand your file IO operations will be dispatched to
python's internal open.

Examples: Reading files
=======================

Shapefiles
----------

.. doctest::

    >>> import pysal
    >>> shp = pysal.open('../pysal/examples/10740.shp')
    >>> poly = next(shp)
    >>> type(poly)
    <class 'pysal.cg.shapes.Polygon'>
    >>> len(shp)
    195
    >>> shp.get(len(shp)-1).id
    195
    >>> polys = list(shp)
    >>> len(polys)
    195

DBF Files
---------

.. doctest::

    >>> import pysal
    >>> db = pysal.open('../pysal/examples/10740.dbf','r')
    >>> db.header
    ['GIST_ID', 'FIPSSTCO', 'TRT2000', 'STFID', 'TRACTID']
    >>> db.field_spec
    [('N', 8, 0), ('C', 5, 0), ('C', 6, 0), ('C', 11, 0), ('C', 10, 0)]
    >>> db.next()
    [1, '35001', '000107', '35001000107', '1.07']
    >>> db[0]
    [[1, '35001', '000107', '35001000107', '1.07']]
    >>> db[0:3]
    [[1, '35001', '000107', '35001000107', '1.07'], [2, '35001', '000108', '35001000108', '1.08'], [3, '35001', '000109', '35001000109', '1.09']]
    >>> db[0:5,1]
    ['35001', '35001', '35001', '35001', '35001']
    >>> db[0:5,0:2]
    [[1, '35001'], [2, '35001'], [3, '35001'], [4, '35001'], [5, '35001']]
    >>> db[-1,-1]
    ['9712']

CSV Files
---------

.. doctest::

    >>> import pysal
    >>> db = pysal.open('../pysal/examples/stl_hom.csv')
    >>> db.header
    ['WKT', 'NAME', 'STATE_NAME', 'STATE_FIPS', 'CNTY_FIPS', 'FIPS', 'FIPSNO', 'HR7984', 'HR8488', 'HR8893', 'HC7984', 'HC8488', 'HC8893', 'PO7984', 'PO8488', 'PO8893', 'PE77', 'PE82', 'PE87', 'RDAC80', 'RDAC85', 'RDAC90']
    >>> db[0]
    [['POLYGON ((-89.585220336914062 39.978794097900391,-89.581146240234375 40.094867706298828,-89.603988647460938 40.095306396484375,-89.60589599609375 40.136119842529297,-89.6103515625 40.3251953125,-89.269027709960938 40.329566955566406,-89.268562316894531 40.285579681396484,-89.154655456542969 40.285774230957031,-89.152763366699219 40.054969787597656,-89.151618957519531 39.919403076171875,-89.224777221679688 39.918678283691406,-89.411857604980469 39.918041229248047,-89.412437438964844 39.931644439697266,-89.495201110839844 39.933486938476562,-89.4927978515625 39.980186462402344,-89.585220336914062 39.978794097900391))', 'Logan', 'Illinois', 17, 107, 17107, 17107, 2.115428, 1.290722, 1.624458, 4, 2, 3, 189087, 154952, 184677, 5.10432, 6.59578, 5.832951, -0.991256, -0.940265, -0.845005]]
    >>> fromWKT = pysal.core.util.wkt.WKTParser()
    >>> db.cast('WKT',fromWKT)
    >>> type(db[0][0][0])
    <class 'pysal.cg.shapes.Polygon'>
    >>> db[0][0][1:]
    ['Logan', 'Illinois', 17, 107, 17107, 17107, 2.115428, 1.290722, 1.624458, 4, 2, 3, 189087, 154952, 184677, 5.10432, 6.59578, 5.832951, -0.991256, -0.940265, -0.845005]
    >>> polys = db.by_col('WKT')
    >>> from pysal.cg import standalone
    >>> standalone.get_bounding_box(polys)[:]
    [-92.70067596435547, 36.88180923461914, -87.91657257080078, 40.329566955566406]


WKT Files
---------

.. doctest::

    >>> import pysal
    >>> wkt = pysal.open('../pysal/examples/stl_hom.wkt', 'r')
    >>> polys = wkt.read()
    >>> wkt.close()
    >>> print len(polys)
    78
    >>> print polys[1].centroid
    (-91.19578469430738, 39.990883050220845)


GeoDa Text Files
----------------

.. doctest::

    >>> import pysal
    >>> geoda_txt = pysal.open('../pysal/examples/stl_hom.txt', 'r')
    >>> geoda_txt.header
    ['FIPSNO', 'HR8488', 'HR8893', 'HC8488']
    >>> print len(geoda_txt)
    78
    >>> geoda_txt.dat[0]
    ['17107', '1.290722', '1.624458', '2']
    >>> geoda_txt._spec
    [<type 'int'>, <type 'float'>, <type 'float'>, <type 'int'>]
    >>> geoda_txt.close()

GAL Binary Weights Files
------------------------

.. doctest::

    >>> import pysal
    >>> gal = pysal.open('../pysal/examples/sids2.gal','r')
    >>> w = gal.read()
    >>> gal.close()
    >>> w.n
    100

GWT Weights Files
-----------------

.. doctest::

    >>> import pysal
    >>> gwt = pysal.open('../pysal/examples/juvenile.gwt', 'r')
    >>> w = gwt.read()
    >>> gwt.close()
    >>> w.n
    168

ArcGIS Text Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> arcgis_txt = pysal.open('../pysal/examples/arcgis_txt.txt','r','arcgis_text')
    >>> w = arcgis_txt.read()
    >>> arcgis_txt.close()
    >>> w.n
    3

ArcGIS DBF Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> arcgis_dbf = pysal.open('../pysal/examples/arcgis_ohio.dbf','r','arcgis_dbf')
    >>> w = arcgis_dbf.read()
    >>> arcgis_dbf.close()
    >>> w.n
    88

ArcGIS SWM Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> arcgis_swm = pysal.open('../pysal/examples/ohio.swm','r')
    >>> w = arcgis_swm.read()
    >>> arcgis_swm.close()
    >>> w.n
    88

DAT Weights Files
-----------------

.. doctest::

    >>> import pysal
    >>> dat = pysal.open('../pysal/examples/wmat.dat','r')
    >>> w = dat.read()
    >>> dat.close()
    >>> w.n
    49

MATLAB MAT Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> mat = pysal.open('../pysal/examples/spat-sym-us.mat','r')
    >>> w = mat.read()
    >>> mat.close()
    >>> w.n
    46

LOTUS WK1 Weights Files
-----------------------

.. doctest::

    >>> import pysal
    >>> wk1 = pysal.open('../pysal/examples/spat-sym-us.wk1','r')
    >>> w = wk1.read()
    >>> wk1.close()
    >>> w.n
    46

GeoBUGS Text Weights Files
--------------------------

.. doctest::

    >>> import pysal
    >>> geobugs_txt = pysal.open('../pysal/examples/geobugs_scot','r','geobugs_text')
    >>> w = geobugs_txt.read()
    WARNING: there are 3 disconnected observations
    Island ids:  [6, 8, 11]
    >>> geobugs_txt.close()
    >>> w.n
    56

STATA Text Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> stata_txt = pysal.open('../pysal/examples/stata_sparse.txt','r','stata_text')
    >>> w = stata_txt.read()
    WARNING: there are 7 disconnected observations
    Island ids:  [5, 9, 10, 11, 12, 14, 15]
    >>> stata_txt.close()
    >>> w.n
    56

.. _mtx:

MatrixMarket MTX Weights Files
------------------------------

This file format or its variant is currently under consideration of the PySAL team 
to store general spatial weights in a sparse matrix form.

.. doctest::

    >>> import pysal
    >>> mtx = pysal.open('../pysal/examples/wmat.mtx','r')
    >>> w = mtx.read()
    >>> mtx.close()
    >>> w.n
    49

Examples: Writing files
=======================

GAL Binary Weights Files
------------------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> gal = pysal.open('../pysal/examples/virginia_queen.gal','w')
    >>> gal.write(w)
    >>> gal.close()

GWT Weights Files
-----------------

Currently, it is not allowed to write a GWT file.

ArcGIS Text Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> arcgis_txt = pysal.open('../pysal/examples/virginia_queen.txt','w','arcgis_text')
    >>> arcgis_txt.write(w, useIdIndex=True)
    >>> arcgis_txt.close()

ArcGIS DBF Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> arcgis_dbf = pysal.open('../pysal/examples/virginia_queen.dbf','w','arcgis_dbf')
    >>> arcgis_dbf.write(w, useIdIndex=True)
    >>> arcgis_dbf.close()

ArcGIS SWM Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> arcgis_swm = pysal.open('../pysal/examples/virginia_queen.swm','w')
    >>> arcgis_swm.write(w, useIdIndex=True)
    >>> arcgis_swm.close()

DAT Weights Files
-----------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> dat = pysal.open('../pysal/examples/virginia_queen.dat','w')
    >>> dat.write(w)
    >>> dat.close()

MATLAB MAT Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> mat = pysal.open('../pysal/examples/virginia_queen.mat','w')
    >>> mat.write(w)
    >>> mat.close()

LOTUS WK1 Weights Files
-----------------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> wk1 = pysal.open('../pysal/examples/virginia_queen.wk1','w')
    >>> wk1.write(w)
    >>> wk1.close()

GeoBUGS Text Weights Files
--------------------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> geobugs_txt = pysal.open('../pysal/examples/virginia_queen','w','geobugs_text')
    >>> geobugs_txt.write(w)
    >>> geobugs_txt.close()

STATA Text Weights Files
-------------------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> stata_txt = pysal.open('../pysal/examples/virginia_queen.txt','w','stata_text')
    >>> stata_txt.write(w,matrix_form=True)
    >>> stata_txt.close()

MatrixMarket MTX Weights Files
------------------------------

.. doctest::

    >>> import pysal
    >>> w = pysal.queen_from_shapefile('../pysal/examples/virginia.shp',idVariable='FIPS')
    >>> w.n
    136
    >>> mtx = pysal.open('../pysal/examples/virginia_queen.mtx','w')
    >>> mtx.write(w)
    >>> mtx.close()

Examples: Converting the format of spatial weights files
========================================================

PySAL provides a utility tool to convert a weights file from one format to another.

From GAL to ArcGIS SWM format

.. doctest::

    >>> from pysal.core.util.weight_converter import weight_convert
    >>> gal_file = '../pysal/examples/sids2.gal'
    >>> swm_file = '../pysal/examples/sids2.swm'
    >>> weight_convert(gal_file, swm_file, useIdIndex=True)
    >>> wold = pysal.open(gal_file, 'r').read()
    >>> wnew = pysal.open(swm_file, 'r').read()
    >>> wold.n == wnew.n
    True


For further details see the :doc:`FileIO API <../../library/core/FileIO>`.
