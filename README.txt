
=================================================
Python Spatial Analysis Library
=================================================

.. Contents::


What is PySAL
--------------

PySAL is an open source cross-platform library of spatial analysis functions
written in Python. It is intended to support the development of high level
applications for spatial analysis.

It is important to underscore what PySAL is, and is not, designed to do. First
and foremost, PySAL is a library in the fullest sense of the word. Developers
looking for a suite of spatial analytical methods that they can incorporate
into application development should feel at home using PySAL. Spatial analysts
who may be carrying out research projects requiring customized scripting,
extensive simulation analysis, or those seeking to advance the state of the art
in spatial analysis should also find PySAL to be a useful foundation for their
work.

End users looking for a user friendly graphical user interface for spatial
analysis should not turn to PySAL directly. Instead, we would direct them to
projects like STARS and the GeoDaX suite of software products which wrap PySAL
functionality in GUIs. At the same time, we expect that with developments such
as the Python based plug-in architectures for QGIS, GRASS, and the toolbox
extensions for ArcGIS, that end user access to PySAL functionality will be
widening in the near future.


PySAL structure
---------------

Currently PySAL consists of the following files and directories:

  LICENSE.txt
    PySAL license.

  INSTALL.txt
    PySAL prerequisites, installation, testing, and troubleshooting.

  THANKS.txt
    PySAL developers and contributors. Please keep it up to date!!

  README.txt
    PySAL structure (this document).

  setup.py
    Script for building and installing PySAL.

  MANIFEST.in
    Additions to distutils-generated PySAL tar-balls.

  CHANGELOG.txt
    Changes since the last release

  pysal/
    Contains PySAL __init__.py and the directories of PySAL modules.

  doc/
    Contains PySAL documentation using the Sphinx framework.

PySAL modules
+++++++++++++

    * pysal.core — Core Data Structures and IO
    * pysal.cg — Computational Geometry
    * pysal.esda — Exploratory Spatial Data Analysis
    * pysal.examples — Data Sets
    * pysal.inequality — Spatial Inequality Analysis
    * pysal.region — Spatially constrained clustering
    * pysal.spatial_dynamics — Spatial Dynamics
    * pysal.spreg — Regression and diagnostics
    * pysal.weights — Spatial Weights
    * pysal.FileIO — PySAL FileIO: Module for reading and writing various file
                     types in a Pythonic way


Website
-------

All things PySAL can be found here
    http://pysal.org/


Mailing Lists
-------------

Please see the developer's list here
    http://groups.google.com/group/pysal-dev

Help for users is here
    http://groups.google.com/group/openspace-list

Bug reports
-----------

To search for or report bugs, please see
    http://code.google.com/p/pysal/issues/list


License information
-------------------

See the file "LICENSE.txt" for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
