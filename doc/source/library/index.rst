#################
Library Reference
#################

:Release: |version|
:Date: |today|

Python Spatial Analysis Library
===============================

The Python Spatial Analysis Library consists of several sub-packages each addressing a different area of spatial analysis.  In addition to these sub-packages PySAL includes some general utilities used across all modules.

Documentation
-------------
PySAL documentation is available in two forms: python docstrings and a html webpage at http://pysal.org/

Available sub-packages
----------------------
.. toctree::
    :maxdepth: 2

    cg/index
    esda/index

core
esda
examples
weights

cg
    Basic data structures and tools for Computational Geometry
core
    Basic functions used by several sub-packages
esda
    Tools for Exploratory Spatial Data Analysis
examples
    Example data sets used by several sub-packages for examples and testing
weights
    Tools for creating and manipulating weights

Utilities
---------
fileio
    Tool for file input and output, supports many well known file formats
__version__
    PySAL version string
