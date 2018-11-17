.. giddy documentation master file, created by
   sphinx-quickstart on Wed Jun  6 15:54:22 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GeospatIal Distribution DYnamics (GIDDY)
========================================

Giddy is an open-source python library for the analysis of dynamics of
longitudinal spatial data. Originating from the spatial dynamics module
in `PySAL`_ (Python Spatial Analysis Library), it is under active development
for the inclusion of many newly proposed analytics that consider the
role of space in the evolution of distributions over time and has
several new features including inter- and intra-regional decomposition
of mobility association and local measures of exchange mobility in
addition to space-time LISA and spatial markov methods.


.. raw:: html

    <div class="container-fluid">
      <div class="row equal-height">
        <div class="col-sm-1 col-xs-hidden">
        </div>
        <div class="col-md-3 col-xs-12">
            <a href="http://nbviewer.jupyter.org/github/pysal/giddy/blob/master/notebooks/directional.ipynb" class="thumbnail">
                <img src="_static/images/rose_conditional.png" class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Rose diagram (directional LISAs)</h6>
                </div>
            </a>
        </div>
        <div class="col-sm-3 col-xs-12">
            <a href="http://nbviewer.jupyter.org/github/pysal/giddy/blob/master/notebooks/Markov%20Based%20Methods.ipynb" class="thumbnail">
                <img src="_static/images/spatial_markov_us.png" class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Spatial Markov Chain</h6>
                </div>
            </a>
        </div>
        <div class="col-sm-3 col-xs-12">
            <a href="http://nbviewer.jupyter.org/github/pysal/giddy/blob/master/notebooks/Rank%20based%20Methods.ipynb" class="thumbnail">
                <img src="_static/images/neighboorsetLIMA_US.png"
                class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Neighbor Set Local Indicator of Mobility Association (LIMA)
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


.. _PySAL: https://github.com/pysal/pysal