.. documentation master file

PySAL: Python Spatial Analysis Library
======================================
PySAL is an open source
cross-platform library for geospatial data science with an emphasis on
geospatial vector data written in Python. 

.. raw:: html

    <div class="container-fluid">
      <div class="row equal-height">
        <div class="col-sm-1 col-xs-hidden">
        </div>
        <div class="col-md-3 col-xs-12">
            <a href="http://nbviewer.jupyter.org/github/pysal/esda/blob/main/notebooks/Spatial%20Autocorrelation%20for%20Areal%20Unit%20Data.ipynb" class="thumbnail">
                <img src="_static/images/prices.png" class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Housing Prices Berlin</h6>
                </div>
            </a>
        </div>
        <div class="col-sm-3 col-xs-12">
         <a href="http://nbviewer.jupyter.org/github/pysal/giddy/blob/main/notebooks/DirectionalLISA.ipynb" class="thumbnail">
        <img src="_static/images/rose_conditional.png" class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Rose diagram (directional LISAs)</h6>
                </div>
            </a>
        </div>
        <div class="col-sm-3 col-xs-12">
 <a href="http://nbviewer.jupyter.org/github/pysal/splot/blob/main/notebooks/libpysal_non_planar_joins_viz.ipynb" class="thumbnail">
                <img src="_static/images/nonplanar.png"
                class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Visualizing Non Planar Neighbours
                </h6>
                </div>
      </a>
        </div>
        <div class="col-sm-2 col-xs-hidden">
        </div>
      </div>
    </div>

PySAL supports the development of
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
    polygonal lattices.  Also includes methods for spatial inequality,  distributional dynamics, and segregation.
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

        
.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Contents:

   Installation <installation>
   API <api>
   References <references>

Details are available in the `PySAL api <api.html>`_.

For background information see :cite:`pysal2007, rey2014PythonSpatial, anselin2014ModernSpatial, rey2019pysal, fotheringham2017multiscale, fleischmann_2019, cortes2019OpensourceFramework, wolf2019GeosilhouettesGeographical, yu:2019, rey2020VisualAnalytics, Lumnitz2020, saxon2021OpenSoftware, rey_2021a, Gaboardi2021,rey2022PySALEcosystem,  spopt2022, rey2023GeographicData`.

***********
Development
***********

As of version 2.0.0, PySAL is now a collection of affiliated geographic
data science packages. Changes to the code for any of the subpackages
should be directed at the respective `upstream repositories <https://github.com/pysal/help>`_ and not made
here. Infrastructural changes for the meta-package, like those for
tooling, building the package, and code standards, will be considered.


PySAL development is hosted on github_.

.. _github : https://github.com/pysal/PySAL



Discussions of development occurs on the
`developer list <http://groups.google.com/group/pysal-dev>`_
as well as discord_.

.. _discord : https://discord.gg/BxFTEPFFZn

****************
Getting Involved
****************

If you are interested in contributing to PySAL please see our
`development guidelines <https://github.com/pysal/pysal/wiki>`_.


***********
Bug reports
***********

To search for or report bugs, please see PySAL's issues_.

.. _issues :  http://github.com/pysal/pysal/issues


***************
Citing PySAL
***************

If you use PySAL in a scientific publication, we would appreciate citations to the following paper:

  `PySAL: A Python Library of Spatial Analytical Methods <https://doi.org/10.52324/001c.8285>`_, *Rey, S.J. and L. Anselin*, Review of Regional Studies 37, 5-27 2007.

  Bibtex entry::

      @Article{pysal2007,
        author={Rey, Sergio J. and Anselin, Luc},
        title={{PySAL: A Python Library of Spatial Analytical Methods}},
        journal={The Review of Regional Studies},
        year=2007,
        volume={37},
        number={1},
        pages={5-27},
        keywords={Open Source; Software; Spatial}
      }



*******************
License information
*******************

See the file "LICENSE.txt" for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.


.. _PySAL: https://github.com/pysal/pysal
