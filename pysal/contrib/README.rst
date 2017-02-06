:mod:`pysal.contrib` -- Contributed Modules 
===========================================

**Intro**

The PySAL Contrib library contains user contributions that enhance PySAL, but
are not fit for inclusion in the general library. The primary reason a
contribution would not be allowed in the general library is external
dependencies. PySAL has a strict no dependency policy (aside from Numpy/Scipy).
This helps ensure the library is easy to install and maintain.

However, this policy often limits our ability to make use of existing code or
exploit performance enhancements from C-extensions. This contrib module is
designed to alleviate this problem. There are no restrictions on external
dependencies in contrib. 

**Ground Rules**

 1. Contribs must not be used within the general library.
 2. *Explicit imports*: each contrib must be imported manually.
 3. *Documentation*: each contrib must be documented, dependencies especially.

**Contribs**

Currently the following contribs are available:

 1. World To View Transform -- A class for modeling viewing windows, used by Weights Viewer.

    - .. versionadded:: 1.3
    - Path: pysal.contrib.weights_viewer.transforms
    - Requires: None

 2. Weights Viewer -- A Graphical tool for examining spatial weights.

    - .. versionadded:: 1.3
    - Path: pysal.contrib.weights_viewer.weights_viewer
    - Requires: `wxPython`_

 3. Shapely Extension -- Exposes shapely methods as standalone functions

    - .. versionadded:: 1.3
    - Path: pysal.contrib.shapely_ext
    - Requires: `shapely`_

 4. Shared Perimeter Weights -- calculate shared perimeters weights.

    - .. versionadded:: 1.3
    - Path: pysal.contrib.shared_perimeter_weights
    - Requires: `shapely`_

 5. Visualization -- Lightweight visualization layer (`Project page`_).

    - .. versionadded:: 1.5
    - Path: pysal.contrib.viz
    - Requires: `matplotlib`_, `pandas`_
    - Optional:
        * `bokeh` backend on `geoplot`: `bokeh`_
        * Color: `palettable`_
        * Folium bridge on notebook: `folium`_, `geojson`_, `IPython`_

 6. Clusterpy -- Spatially constrained clustering.

    - .. versionadded:: 1.8
    - Path: pysal.contrib.clusterpy
    - Requires: `clusterpy`_

 7. Pandas utilities -- Tools to work with spatial file formats using `pandas`.

    - .. versionadded:: 1.8
    - Path: pysal.contrib.pdutilities
    - Requires: `pandas`_

 8. Spatial Interaction -- Tools for spatial interaction (SpInt) modeling. 

    - .. versionadded:: 1.10
    - Path: pysal.contrib.spint
    - Requires: No extra dependencies

 9. glm -- GLM estimation using iteratively weighted least squares estimation

    - .. versionadded:: 1.12
    - Path: pysal.contrib.glm
    - Requires: No extra dependencies


       


.. _clusterpy: https://pypi.python.org/pypi/clusterPy/0.9.9
.. _matplotlib: http://matplotlib.org/
.. _project page: https://github.com/pysal/pysal/wiki/PySAL-Visualization-Project
.. _shapely: https://pypi.python.org/pypi/Shapely
.. _wxPython: http://www.wxpython.org/
