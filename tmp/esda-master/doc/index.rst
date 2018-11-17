.. documentation master file

ESDA: Exploratory Spatial Data Analysis
=======================================

ESDA is an open-source Python library for the exploratory analysis of spatial data. A subpackage of `PySAL`_ (Python Spatial Analysis Library), it is under active development and includes methods for global and local spatial autocorrelation analysis.

.. raw:: html

    <div class="container-fluid">
      <div class="row equal-height">
        <div class="col-sm-1 col-xs-hidden">
        </div>
        <div class="col-md-3 col-xs-12">
            <a href="http://nbviewer.jupyter.org/github/pysal/esda/blob/master/notebooks/Spatial%20Autocorrelation%20for%20Areal%20Unit%20Data.ipynb" class="thumbnail">
                <img src="_static/images/prices.png" class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Housing Prices Berlin</h6>
                </div>
            </a>
        </div>
        <div class="col-sm-3 col-xs-12">
            <a href="http://nbviewer.jupyter.org/github/pysal/esda/blob/master/notebooks/Spatial%20Autocorrelation%20for%20Areal%20Unit%20Data.ipynb" class="thumbnail">
                <img src="_static/images/joincount.png" class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Join Count Analysis</h6>
                </div>
            </a>
        </div>
        <div class="col-sm-3 col-xs-12">
            <a href="http://nbviewer.jupyter.org/github/pysal/esda/blob/master/notebooks/Spatial%20Autocorrelation%20for%20Areal%20Unit%20Data.ipynb" class="thumbnail">
                <img src="_static/images/clustermap.png"
                class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Hot-Spot and Cold-Spot Analysis
                </h6>
                </div>
            </a>
        </div>
        <div class="col-sm-2 col-xs-hidden">
        </div>
      </div>
    </div>


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Contents:

   Installation <installation>
   API <api>
   References <references>

************
Introduction
************

**esda** implements measures for the exploratory analysis spatial data and is part of the  `PySAL family <https://pysal.org>`_

Details are available in the `esda api <api.html>`_.


***********
Development
***********

esda development is hosted on github_.

.. _github : https://github.com/pysal/esda

Discussions of development occurs on the
`developer list <http://groups.google.com/group/pysal-dev>`_
as well as gitter_.

.. _gitter : https://gitter.im/pysal/pysal?

****************
Getting Involved
****************

If you are interested in contributing to PySAL please see our
`development guidelines <http://pysal.readthedocs.org/en/latest/developers/index.html>`_.


***********
Bug reports
***********

To search for or report bugs, please see esda's issues_.

.. _issues :  http://github.com/pysal/esda/issues


***********
Citing esda
***********

If you use PySAL-esda in a scientific publication, we would appreciate citations to the following paper:

  `PySAL: A Python Library of Spatial Analytical Methods <http://journal.srsa.org/ojs/index.php/RRS/article/view/134/85>`_, *Rey, S.J. and L. Anselin*, Review of Regional Studies 37, 5-27 2007.

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
