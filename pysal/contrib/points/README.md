## Point Pattern Analysis in PySAL

Statistical analysis of planar point patterns

### Introduction

This PySAL [contrib][] module is intended to support the statistical analysis of planar point patterns.

It currently works on cartesian coordinates. Users with data in geographic coordinates need to project their data prior to using this module.


### Examples

* [Basic point pattern structure](pointpattern.ipynb)
* [Visualization](visualization.ipynb)
* [Marks](marks.ipynb)
* [Centrography](centrography.ipynb)
* [Simulation of point processes](process.ipynb)
* [Distance based statistics](distance_statistics.ipynb)

### Installation

#### Requirements

- PySAL 1.11+
- Pandas 0.17.0+

### Why in Contrib and Not Core?

The initial implementation focuses on statistical functionality while using [pandas][] as an internal data structure. Future enhancements could either use core PySAL data structures, or swap in [GeoPandas][] for pandas. The former strategy would allow this to move into core, while the latter might postpone such a move until the team makes a decision on whether Pandas and/or GeoPandas were to be hard or soft dependencies in PySAL.


### TODO

- Enhance internal data structures


[contrib]: http://pysal.readthedocs.org/en/latest/library/contrib/index.html
[GeoPandas]: http://geopandas.org
[pandas]: http://pandas.pydata.org

