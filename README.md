Python Spatial Analysis Library
===============================

[![image](https://travis-ci.org/pysal/pysal.svg)](https://travis-ci.org/pysal)

[![image](https://coveralls.io/repos/pysal/pysal/badge.svg?branch=master)](https://coveralls.io/r/pysal/pysal?branch=master)

[![image](https://badges.gitter.im/pysal/pysal.svg)](https://gitter.im/pysal/pysal)

[![image](https://readthedocs.org/projects/pip/badge/?version=latest)](http://pysal.readthedocs.io/en/latest/index.html)

[![LISA Maps of US County Homicide Rates](https://farm2.staticflickr.com/1699/23937788493_1b9d147b9f_z.jpg)](http://nbviewer.ipython.org/urls/gist.githubusercontent.com/darribas/657e0568df7a63362762/raw/pysal_lisa_maps.ipynb)

*Above: Local Indicators of Spatial Association for Homicide Rates in US
Counties 1990.*

PySAL, the Python spatial analysis library, is an open source
cross-platform library for geospatial data science with an emphasis on
geospatial vector data written in Python. It supports the development of
high level applications for spatial analysis, such as

-   detection of spatial clusters, hot-spots, and outliers
-   construction of graphs from spatial data
-   spatial regression and statistical modeling on geographically
    embedded networks
-   spatial econometrics
-   exploratory spatio-temporal data analysis

PySAL Components
================

-   **explore** - modules to conduct exploratory analysis of spatial and spatio-temporal data, including statistical testing on points, networks, and
    polygonal lattices.  Also includes methods for spatial inequality and distributional dynamics.
-   **viz** - visualize patterns in spatial data to detect clusters,
    outliers, and hot-spots.
-   **model** - model spatial relationships in data with a variety of
    linear, generalized-linear, generalized-additive, and nonlinear
    models.
-   **lib** - solve a wide variety of computational geometry problems:
    -   graph construction from polygonal lattices, lines, and points.
    -   construction and interactive editing of spatial weights matrices
        & graphs
    -   computation of alpha shapes, spatial indices, and
        spatial-topological relationships
    -   reading and writing of sparse graph data, as well as pure python
        readers of spatial vector data.

Installation
============

PySAL is available through
[Anaconda](https://www.continuum.io/downloads) (in the defaults or
conda-forge channel) and [Enthought
Canopy](https://www.enthought.com/products/canopy/). We recommend
installing PySAL from conda-forge:

``` {.sourceCode .bash}
conda install pysal
```

PySAL can be installed using pip:

``` {.sourceCode .bash}
pip install pysal
```

As of version 2.0.0 PySAL has shifted to Python 3 only.

Users who need an older stable version of PySAL that is Python 2
compatible can install version 1.14.3 through pip or conda:

``` {.sourceCode .bash}
conda install pysal==1.14.3
```

Documentation
=============

For help on using PySAL, check out the following resources:

-   [User
    Guide](https://pysal.readthedocs.io/en/latest/)
-   [Tutorials and Short
    Courses](https://github.com/pysal/notebooks)

Development
===========

As of version 2.0.0, PySAL is now a collection of affiliated geographic data
science packages. Changes to the code for any of the subpackages should be
directed at the respective [upstream
repositories](http://github.com/pysal/help), and not made here. Infrastructural
changes for the meta-package, like those for tooling, building the package, and
code standards, will be considered.

Development is hosted on [github](https://github.com/pysal/pysal).

Discussions of development as well as help for users occurs on the
[developer list](http://groups.google.com/group/pysal-dev) as well as
[gitter](https://gitter.im/pysal/pysal?).

Getting Involved
================

If you are interested in contributing to PySAL please see our
[development guidelines](https://github.com/pysal/pysal/wiki).

Bug reports
===========

To search for or report bugs, please see PySAL\'s
[issues](http://github.com/pysal/pysal/issues).

License information
===================

See the file \"LICENSE.txt\" for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
