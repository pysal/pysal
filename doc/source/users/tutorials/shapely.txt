.. testsetup:: *
        
        import pysal
        import numpy as np

###########################################
Using PySAL with Shapely for GIS Operations
###########################################


.. versionadded:: 1.3

Introduction
============

The `Shapely <http://pypi.python.org/pypi/Shapely>`_ project is a BSD-licensed
Python package for manipulation and analysis of planar geometric objects, and
depends on the widely used GEOS library.

PySAL supports interoperation with the Shapely library through Shapely's Python
Geo Interface. All PySAL geometries provide a __geo_interface__ property which
models the geometries as a GeoJSON object. Shapely geometry objects also export
the __geo_interface__ property and can be adapted to PySAL geometries using
the :py:func:`pysal.cg.asShape` function.

Additionally, PySAL provides an optional contrib module that handles the
conversion between pysal and shapely data strucutures for you.  The module can
be found in at, :py:mod:`pysal.contrib.shapely_ext`.

Installation
============

Please refer to the `Shapely <http://pypi.python.org/pypi/Shapely>`_
website for instructions on installing Shapely and its
dependencies, *without which PySAL's Shapely extension will not work.*

Usage
=====

Using the Python Geo Interface...

.. doctest::

    >>> import pysal
    >>> import shapely.geometry
    >>> # The get_path function returns the absolute system path to pysal's
    >>> # included example files no matter where they are installed on the system.
    >>> fpath = pysal.examples.get_path('stl_hom.shp')
    >>> # Now, open the shapefile using pysal's FileIO
    >>> shps = pysal.open(fpath , 'r')
    >>> # We can read a polygon...
    >>> polygon = shps.next()
    >>> # To use this polygon with shapely we simply convert it with
    >>> # Shapely's asShape method.
    >>> polygon = shapely.geometry.asShape(polygon)
    >>> # now we can operate on our polygons like normal shapely objects...
    >>> print "%.4f"%polygon.area
    0.1701
    >>> # We can do things like buffering...
    >>> eroded_polygon = polygon.buffer(-0.01)
    >>> print "%.4f"%eroded_polygon.area
    0.1533
    >>> # and containment testing...
    >>> polygon.contains(eroded_polygon)
    True
    >>> eroded_polygon.contains(polygon)
    False
    >>> # To go back to pysal shapes we call pysal.cg.asShape...
    >>> eroded_polygon = pysal.cg.asShape(eroded_polygon)
    >>> type(eroded_polygon)
    <class 'pysal.cg.shapes.Polygon'>

Using The PySAL shapely_ext module...

.. doctest::

    >>> import pysal
    >>> from pysal.contrib import shapely_ext
    >>> fpath = pysal.examples.get_path('stl_hom.shp')
    >>> shps = pysal.open(fpath , 'r')
    >>> polygon = shps.next()
    >>> eroded_polygon = shapely_ext.buffer(polygon, -0.01)
    >>> print "%0.4f"%eroded_polygon.area
    0.1533
    >>> shapely_ext.contains(polygon,eroded_polygon)
    True
    >>> shapely_ext.contains(eroded_polygon,polygon)
    False
    >>> type(eroded_polygon)
    <class 'pysal.cg.shapes.Polygon'>


    

