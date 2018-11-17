GeospatIal Distribution DYnamics (giddy) in PySAL
=================================================

[![Build Status](https://travis-ci.org/pysal/giddy.svg?branch=master)](https://travis-ci.org/pysal/giddy)
[![Gitter room](https://badges.gitter.im/pysal/giddy.svg)](https://gitter.im/pysal/giddy)
[![Documentation Status](https://readthedocs.org/projects/giddy/badge/?version=latest)](https://giddy.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/91390088.svg)](https://zenodo.org/badge/latestdoi/91390088)

Giddy is an open-source python library for the analysis of dynamics of
longitudinal spatial data. Originating from the spatial dynamics module
in [PySAL (Python Spatial Analysis Library)](http://pysal.org/), it is under active development
for the inclusion of newly proposed analytics that consider the
role of space in the evolution of distributions over time.

*Below are six choropleth maps of US state per-capita incomes from 1929 to 2004 at a fifteen-year interval.*

![us_qunitile_maps](figs/us_qunitile_maps.png)

Features
--------
- Directional LISA, inference and visualization as rose diagram

[![rose_conditional](figs/rose_conditional.png)](notebooks/directional.ipynb)

*Above shows the rose diagram (directional LISAs) for US states incomes across 1969-2009 conditional on relative incomes in 1969.*

- Spatially explicit Markov methods:
    - Spatial Markov and inference
    - LISA Markov and inference
- Spatial decomposition of exchange mobility measure (rank methods):
    - Global indicator of mobility association (GIMA) and inference
    - Inter- and intra-regional decomposition of mobility association and inference
    - Local indicator of mobility association (LIMA)
        - Neighbor set LIMA and inference
        - Neighborhood set LIMA and inference

[![us_neigborsetLIMA](figs/us_neigborsetLIMA.png)](notebooks/Rank%20based%20Methods.ipynb)

- Income mobility measures

Examples
--------

* [Directional LISA](notebooks/directional.ipynb)
* [Markov based methods](notebooks/Markov%20Based%20Methods.ipynb)
* [Rank based methods](notebooks/Rank%20based%20Methods.ipynb)
* [Mobility measures](notebooks/Mobility%20measures.ipynb)

Installation
------------

Install the stable version released on the [Python Package Index](http://pypi.python.org/pypi/pysal) from the command line:

```
pip install giddy
```

Install the development version on [pysal/giddy](https://github.com/pysal/giddy):

```
pip install https://github.com/pysal/giddy/archive/master.zip
```


#### Requirements

- libpysal
- esda
- mapclassify

Contribute
----------

PySAL-giddy is under active development and contributors are welcome.

If you have any suggestion, feature request, or bug report, please open a new [issue](https://github.com/pysal/giddy/issues) on GitHub. To submit patches, please follow the PySAL development [guidelines](http://pysal.readthedocs.io/en/latest/developers/index.html) and open a [pull request](https://github.com/pysal/giddy). Once your changes get merged, you’ll automatically be added to the [Contributors List](https://github.com/pysal/giddy/graphs/contributors).

Support
-------

If you are having issues, please talk to us in the [gitter room](https://gitter.im/pysal/giddy).

License
-------

The project is licensed under the [BSD license](https://github.com/pysal/giddy/blob/master/LICENSE.txt).

Funding
-------

<img src="figs/nsf_logo.jpg" width="50"> Award #1421935 [New Approaches to Spatial Distribution Dynamics](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1421935)
