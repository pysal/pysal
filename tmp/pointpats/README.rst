Point Pattern Analysis in PySAL
===============================
.. image:: https://api.travis-ci.org/pysal/pointpats.svg
   :target: https://travis-ci.org/pysal/pointpats

Statistical analysis of planar point patterns.

This package is part of a `refactoring of PySAL
<https://github.com/pysal/pysal/wiki/PEP-13:-Refactor-PySAL-Using-Submodules>`_.


************
Introduction
************

This `pointpats <https://github.com/pysal/pointpats>`_ package is intended to support the statistical analysis of planar point patterns.

It currently works on cartesian coordinates. Users with data in geographic coordinates need to project their data prior to using this module.

Mimicking parts of the original PySAL api can be done with

``import pointpats.api as ps``

********
Examples
********

- `Basic point pattern structure <https://github.com/pysal/pointpats/tree/master/notebooks/pointpattern.ipynb>`_
- `Centrography and visualization <https://github.com/pysal/pointpats/tree/master/notebooks/centrography.ipynb>`_
- `Marks <https://github.com/pysal/pointpats/tree/master/notebooks/marks.ipynb>`_
- `Simulation of point processes <https://github.com/pysal/pointpats/tree/master/notebooks/process.ipynb>`_
- `Distance based statistics <https://github.com/pysal/pointpats/tree/master/notebooks/distance_statistics.ipynb>`_

************
Installation
************

Install pointpats by running:

::

    $ pip install pointpats

***********
Development
***********

pointpats development is hosted on `github <https://github.com/pysal/pointpats>`_.

As part of the PySAL project, pointpats development follows these `guidelines <http://pysal.readthedocs.io/en/latest/developers/index.html>`_.

***********
Bug reports
***********

To search for or report bugs, please see pointpat's `issues <https://github.com/pysal/pointpats/issues>`_.
