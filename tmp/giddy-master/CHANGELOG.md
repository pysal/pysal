Version 2.0.0 (2018-08-26)

This release does not add any new functionality to `giddy`, but
instead features api changes in `giddy` and its
pysal submodule dependencies `mapclassify`, `libpysal`, and `esda`. More
specifically, the `giddy.api` module which was originally designed to
 facilitate a smoother transition from the old metapackage pysal to the
 refactored submodule structure (see [here](http://pysal.org/about.html#migrating-to-pysal-2-0)
 for details) was removed as we are moving away from the
refactoring stage and looking at the future development.

We closed a total of 20 issues (enhancements and bug fixes) through 9 pull requests, since our last release on 2018-07-27.

## Issues Closed
  - remove giddy.api in README.rst (#66)
  -  chore: update for libpysal lower case module name changes (#65)
  - remove api.py (#62)
  - set up travis dual testing against mapclassify and  esda (#63)
  - replace `libpysal.api` imports with new imports in `markov.py` and `d… (#61)
  - Remove api.py and account for changes in (incoming) API of mapclassify, esda, and libpysal (#64)
  - version giddy only in giddy/__ini__.py (#60)
  - remove duplicate makefile for sphinx build (#59)
  - add zenodo doi badge to README (#58)
  - add changelog for the release 1.2.0 (#57)
  - prepare for release 1.2.0 (#56)

## Pull Requests
  - remove giddy.api in README.rst (#66)
  -  chore: update for libpysal lower case module name changes (#65)
  - replace `libpysal.api` imports with new imports in `markov.py` and `d… (#61)
  - Remove api.py and account for changes in (incoming) API of mapclassify, esda, and libpysal (#64)
  - version giddy only in giddy/__ini__.py (#60)
  - remove duplicate makefile for sphinx build (#59)
  - add zenodo doi badge to README (#58)
  - add changelog for the release 1.2.0 (#57)
  - prepare for release 1.2.0 (#56)

The following individuals contributed to this release:

  - Wei Kang
  - Stefanie Lumnitz

v<1.2.0>, 2018-07-27

This release features:
* a more flexible specification for the [spatial Markov chains model](https://github.com/pysal/giddy/blob/master/giddy/markov.py#L169).More specifically, for continuous time series input:
    * both the numbers of classifications for input continuous time series (k) and spatial lags (m) can be specified and allowed to be different
    * user-defined classifications (cutoffs) for input continuous time series and spatial lags are allowed
* new visualization tools:
    * relies on [pysal/splot](https://github.com/pysal/splot) for [visualizing
    Dynamic LISA related statistics](https://github.com/pysal/giddy/blob/master/giddy/directional.py#L322)
* launch of the [documentation website](http://giddy.readthedocs.io/)

GitHub stats for 2018/05/18 - 2018/07/27

These lists are automatically generated, and may be incomplete or contain duplicates.

We closed a total of 34 issues, 16 pull requests and 18 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (16):

* :ghpull:`56`: b'prepare for release 1.2.0'
* :ghpull:`55`: b'set up dual travis tests for libpysal (pip and github)'
* :ghpull:`54`: b'ENH: Allow for more flexible specification of Spatial Markov'
* :ghpull:`52`: b'Update notebooks to rely on geopandas for mapping'
* :ghpull:`51`: b'ENH to docs'
* :ghpull:`50`: b'include /tests in the release and correct for the directional doctests'
* :ghpull:`49`: b'add doc building badge to README'
* :ghpull:`47`: b'Tests and documentation for `rose.plot()` and `rose.plot_vectors()`'
* :ghpull:`48`: b'A tentative version of giddy documentation website with sphinx '
* :ghpull:`46`: b'force utf8 for the install description read'
* :ghpull:`43`: b'implement `rose.plot()` and `rose.plot_vectors()` method using `splot`'
* :ghpull:`44`: b'More on building doc webpages using sphinx'
* :ghpull:`42`: b'Gallery'
* :ghpull:`41`: b'new features for sphinx documentation website'
* :ghpull:`40`: b'typo - email notifications'
* :ghpull:`38`: b'fix for python 3'

Issues (18):

* :ghissue:`56`: b'prepare for release 1.2.0'
* :ghissue:`55`: b'set up dual travis tests for libpysal (pip and github)'
* :ghissue:`54`: b'ENH: Allow for more flexible specification of Spatial Markov'
* :ghissue:`52`: b'Update notebooks to rely on geopandas for mapping'
* :ghissue:`51`: b'ENH to docs'
* :ghissue:`50`: b'include /tests in the release and correct for the directional doctests'
* :ghissue:`49`: b'add doc building badge to README'
* :ghissue:`47`: b'Tests and documentation for `rose.plot()` and `rose.plot_vectors()`'
* :ghissue:`48`: b'A tentative version of giddy documentation website with sphinx '
* :ghissue:`45`: b'encoding issue in  README.rst'
* :ghissue:`46`: b'force utf8 for the install description read'
* :ghissue:`43`: b'implement `rose.plot()` and `rose.plot_vectors()` method using `splot`'
* :ghissue:`44`: b'More on building doc webpages using sphinx'
* :ghissue:`42`: b'Gallery'
* :ghissue:`41`: b'new features for sphinx documentation website'
* :ghissue:`40`: b'typo - email notifications'
* :ghissue:`38`: b'fix for python 3'
* :ghissue:`39`: b'first draft of sphinx gallery'


v<1.1.1>, 2018-05-17

This release is the first tagged release of giddy on Github.
Starting from this release, giddy supports python 3.5 and 3.6 only.
This release also features categorical spatial Markov which enables spatial Markov
(class [Spatial_Markov](https://github.com/pysal/giddy/blob/master/giddy/markov.py#L179))
to be applied to categorical time series such as land use and land cover change,
as well as neighborhood change. Here, the spatial lag (utilizing function
[lag_categorical](https://github.com/pysal/libpysal/blob/master/libpysal/weights/spatial_lag.py#L88)
in [libpysal](https://github.com/pysal/libpysal)) is defined as the most
common category among neighbors.

GitHub stats for 2017/07/21 - 2018/05/17

These lists are automatically generated, and may be incomplete or contain duplicates.


We closed a total of 20 issues, 8 pull requests and 12 regular issues;
this is the full list (generated with the script 
:file:`tools/github_stats.py`):

Pull Requests (8):

* :ghpull:`36`: b'add changelog for the release 1.1.0'
* :ghpull:`35`: b'prepare for release'
* :ghpull:`34`: b'code 2to3'
* :ghpull:`33`: b'chore: update for python 3+ only'
* :ghpull:`28`: b'add inequality to api'
* :ghpull:`29`: b'adding discretized Spatial_Markov'
* :ghpull:`25`: b'[WIP] prepare for full metapackage integration'
* :ghpull:`27`: b'api module for giddy'

Issues (12):

* :ghissue:`36`: b'add changelog for the release 1.1.0'
* :ghissue:`35`: b'prepare for release'
* :ghissue:`34`: b'code 2to3'
* :ghissue:`33`: b'chore: update for python 3+ only'
* :ghissue:`31`: b"How to use the 'development' version"
* :ghissue:`30`: b'KeyError: 1 in spatial_lag.py'
* :ghissue:`26`: b'giddy needs an api.py module'
* :ghissue:`28`: b'add inequality to api'
* :ghissue:`29`: b'adding discretized Spatial_Markov'
* :ghissue:`25`: b'[WIP] prepare for full metapackage integration'
* :ghissue:`27`: b'api module for giddy'
* :ghissue:`12`: b'Publish on PyPi'
