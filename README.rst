Python Spatial Analysis Library
=================================

.. image:: https://travis-ci.org/pysal/pysal.svg
   :target: https://travis-ci.org/pysal

.. image:: https://coveralls.io/repos/pysal/pysal/badge.svg?branch=master
   :target: https://coveralls.io/r/pysal/pysal?branch=master

.. image:: https://badges.gitter.im/pysal/pysal.svg
   :target: https://gitter.im/pysal/pysal

PySAL_ is an open source cross-platform library of spatial analysis functions
written in Python. It is intended to support the development of high level
applications for spatial analysis.

.. image:: https://farm2.staticflickr.com/1699/23937788493_1b9d147b9f_z.jpg
        :width: 25%
        :scale: 25%
        :target: http://nbviewer.ipython.org/urls/gist.githubusercontent.com/darribas/657e0568df7a63362762/raw/pysal_lisa_maps.ipynb
        :alt: LISA Maps of US County Homicide Rates

*Above: Local Indicators of Spatial Association for Homicide Rates in US
Counties 1990.*



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
projects like  the GeoDaX_ suite of software products which wrap PySAL
functionality in GUIs. At the same time, we expect that with developments such
as the Python based plug-in architectures for QGIS, GRASS, and the toolbox
extensions for ArcGIS, that end user access to PySAL functionality will be
widening in the near future.

.. _PySAL : https://github.com/pysal/pysal/
.. _GeoDaX : https://geodacenter.asu.edu/software


.. |build| image:: https://travis-ci.org/pysal/pysal.png
   :scale: 100%
   :align: middle
   :target: https://travis-ci.org/pysal/pysal
.. |cover| image:: https://coveralls.io/repos/pysal/pysal/badge.svg?branch=master
   :scale: 50%
   :align: top
   :target: https://coveralls.io/r/pysal/pysal?branch=master
.. |docs| image:: https://readthedocs.org/projects/pysal/badge/?verison=latest
   :scale: 50%
   :align: top
   :target: http://pysal.readthedocs.org/en/latest/ 
.. |talk| image:: https://badges.gitter.im/Join%20Chat.svg
   :scale: 50%
   :align: top
   :target: https://gitter.im/pysal/pysal?



PySAL modules
=============

* pysal.cg  Computational geometry
* pysal.contrib  Contributed modules
* pysal.core  Core data structures and IO
* pysal.esda  Exploratory spatial data analysis
* pysal.examples  Data sets
* pysal.inequality  Spatial inequality analysis
* pysal.network  Spatial analysis on networks
* pysal.region  Spatially constrained clustering
* pysal.spatial_dynamics  Spatial dynamics
* pysal.spreg  Spatial econometrics and diagnostics
* pysal.weights  Spatial weights


Installation
============

PySAL can be installed using pip:

.. code-block:: bash
   
    $ pip install pysal

PySAL is also available through 
`Anaconda <https://www.continuum.io/downloads>`__ and `Enthought Canopy <https://www.enthought.com/products/canopy/>`__.

Documentation
=============

For help on using PySAL, check out the following resources:

* `User Guide <http://pysal.readthedocs.org/en/latest/users/index.html>`_
* `Tutorials and Short Courses <https://github.com/pysal/notebooks/blob/master/courses.md>`_
* `Notebooks <https://github.com/pysal/notebooks>`_
* `User List <http://groups.google.com/group/openspace-list>`_



Development
===========

PySAL development is hosted on github_.

.. _github : https://github.com/pysal/pysal

Discussions of development occurs on the
`developer list <http://groups.google.com/group/pysal-dev>`_
as well as gitter_.

.. _gitter : https://gitter.im/pysal/pysal?

Getting Involved
================

If you are interested in contributing to PySAL please see our 
`development guidelines <http://pysal.readthedocs.org/en/latest/developers/index.html>`_.


Bug reports
===========
To search for or report bugs, please see PySAL's issues_.

.. _issues :  http://github.com/pysal/pysal/issues

License information
===================

See the file "LICENSE.txt" for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
