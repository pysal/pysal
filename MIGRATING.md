# Migrating to PySAL 2.0 

<div align='left'>
![https://gitter.im/pysal/pysal](https://badges.gitter.im/pysal/pysal.svg)
</div>

PySAL, the Python spatial analysis library, will be changing its package structure. 

- We are changing the module structure to better reflect *what you do* with the library rather than the *academic disciplines* the components of the library come from. 

- This also makes the library significantly more maintainable for us, since it reduces the bulk of the library and more evenly distributes the load for maintainers. 

- As an added benefit, it lets end users only install components they need, which is helpful for our colleagues in restricted data centers. 

This means that we have split the `pysal` package into many smaller components, each organized around a common set of problems:

* `lib`: core functionality used by other modules to work with spatial data in Python, including:
    - the construction of graphs (also known as spatial weights) from spatial data
    - computational geometry algorithms: 
         + alpha shapes
         + quadtrees, rTrees, & spherical KDTrees 
         + methods to construct graphs from `scipy` Delaunay triangulation objects
         + Pure Python data structures to represent planar geometric objects
    - pure Python readers/writers for graph/spatial weights files and some spatial data types
* `explore`: exploratory spatial data analysis of clusters, hotspots, and spatial outliers, plus spatial statistics on graphs and point patterns. These include
    * Point pattern analysis of centrography and colocation ($F$,$G$,$J$,$K$ statistics in space)
    * Pattern analysis of point patterns snapped to a network ($F$,$G$,$K$ statistics on networks)
    * Analysis of spatial patterning on lattices, including
        * univariate Join count statistics
        * Moran's $I$ statistics, including local & bivariate methods
        * Geary's $C$
        * Getis-Ord $G$, $G^*$, and local $G$ statistics
        * General $\Gamma$ statistics for spatial clustering 
* `model`: explicitly spatial modeling tools including:
    - geographically-weighted regression, a generalized additive spatial model specification for local coefficients
    - spatially-correlated variance-components models, a type of Bayesian hierarchical model providing for group-level random spatial random effects, as well as diagnostics for Bayesian modeling (Geweke, Markov Chain Monte Carlo Standard Error, Potential Scale Reduction Factors)
    - Bayesian spatially-varying coefficient process models (e.g. local random effects models)
    - Maximum Likelihood spatial econometric models, including:
        + mixed regressive-autoregressive (spatial lag) model
        + spatially-correlated error models
        + spatial regimes/spatial mixture models
        + seemingly-unrelated regression models
        + combinations of these various effects
    - econometric specification testing methods, including spatial Lagrange Multiplier, Anselin-Kelejian, Chow, and Jarque-Bera tests.
- `dynamics`: methods to construct & analyze the space-time dynamics and distributions of data
- `viz`: methods to visualize and analyze spatial datasets, specifically the output of exploratory spatial statistics. 

This document provides a brief overview of what each of the new modules are, how they relate to the last **legacy** version of `pysal`, version `1.14.4`, and what you need to do in order to keep your code running the way you expect. 

# Porting your code

## using the standard `pysal` for a stable six-month release cycle

If you want to keep your usual stable `pysal` dependency, it is sufficient to update your imports according to the mappings we provide in the [**Module Lookup**](#module-lookup) section. `pysal` will continue to stick to regular 6-month releases put out by multiple maintainers, with bug-fix releases as needed throughout the year. 

This version will still have nightly regression testing run and will be ensured to work with the latest releases of `numpy` and `scipy`. 
If you don't have an urgent need to reduce your dependency size or the availability of PySAL, continuing to depend on `pysal` directly is the right choice. 

## using the appropriate sub-module for fresher releases & more stable dependencies

If you only use one contained part of `pysal`, are interested in developing another statistical analysis package that only depends on `libpysal`, or simply want to keep your build as lean as possible, you can also install only the sub-modules you require, independently of `pysal`. This is the best option for those of you in restricted analysis environments where every line of code must be vetted by an expert, such as users in restricted data centers conducting academic work. 

All of the sub-packages included in `pysalnext` contain a significant amount of new functionality, as well as interoperability tools for other packages, such as `networkx` and `geopandas`. In addition, most of the old tools from `pysal` are reorganized. In total, there are `12` distinct packages in `pysalnext`, with more being added often. These packages are:
- `libpysal`: the core of the library, containing computational geometry, graph construction, and read/write tools. 
- `esda`: the exploratory spatial data analysis toolkit, containing many statistical functions for characterizing spatial patterns.
- `pointpats`: methods and statistical functions to statistically analyze spatial clustering in point patterns
- `spaghetti`: methods and statistical functions to analyze geographic networks, including the statistical distribution of points on network topologies.
- `giddy`: the geospatial distribution dynamics package, designed to study and characterize how distributions change and relate over space and time. 
- `mapclassify`: a package with map classification algorithms for cartography
- `splot`: a collection of statistical visualizations for the analysis methods included across `pysalnext`
- `gwr`: geographically-weighted regression (both single- and multi-scale)
- `spglm`: methods & functions to fit sparse GLMs
- `spint`: spatial interaction models
- `spreg`: spatial econometric regression
- `spvcm`: spatially-correlated variance components models, plus diagnostics and plotting for Bayesian models fit in `pysalnext`. 

There are four main deprecations that have occurred in `pysalnext`. From legacy `pysal`, these are:

1. `pysal.contrib`: removed due to lack of unittesting and heterogeneous code quality; moved to independent modules where possible. 

2. `pysal.esda.smoothing`: removed due to intractable and subtle numerical bugs in the code caused by porting to Python 3. There is an effort to re-implement this in Python, and will be added when/if this effort finishes. 

3. `pysal.region`: removed because the new version has significantly more dependencies, including `pulp` and `scikit-learn`. Grab this as a standalone package using `pip install region`. 

4. `pysal.meta`: removed.

If you'd like to get code from submodules, they usually have a one-to-one replacement. This will be discussed later in the [**Module Lookup**](#Module-Lookup) section.

# Module Lookup

here is a list of the locations of all of the commonly-used modules in legacy `pysal`, and where they will move in the next release of `pysal`. 

To preview these changes, install the `pysalnext` package using `pip`. If your package works with `pysalnext`, it should work on `pysal` 2.0. 

### Modules in `libpysal`:
- `pysal.cg` will change to `pysal.lib.cg` 
- `pysal.weights` will change to `pysal.lib.weights`, and many weights construction classes now have `from_geodataframe` methods that can graphs directly from `geopandas` `GeoDataFrames`.
- `pysal.open` will change to `pysal.lib.io.open`, and most of `pysal.core` will move to `pysal.lib.io`. *Note: using* `pysal.lib.io.open`*for **anything** but reading/writing spatial weights matrices/graphs is not advised. Please migrate to* `geopandas`* for all spatial data read/write. Further, note that* `WeightsConverter` *has also been deprecated; if you need to convert weights, do so manually using sequential* `open`* and *`write`* statements.*
- `pysal.examples` will change to `pysal.lib.examples`
### Modules in `dynamics`
- `pysal.spatial_dynamics` will change to`pysalnext.dynamics.giddy`
- `pysal.inequality` will change to `pysal.dynamics.inequality`
### Modules from `esda`:
These will mainly move into `pysal.explore.esda`, except for `smoothing` and `mixture_smoothing` (which will be deprecated) and `mapclassify`, which will move to `pysal.viz.mapclassify`. 
### Modules from `network`:
These will move directly into `pysal.explore.spaghetti`
### Modules from `inequality`:
`pysal.inequality` has been published as its own package, `inequality`, and moved to `pysal.dynamics.inequality`. 
### Modules from `spreg`:
`pysal.spreg` has been moved wholesale into `pysal.model`, which now contains many additional kinds of spatial regression models.

# I really don't want to change anything; what can I do?

This is not recommended, and should only be done by those who are planning to deprecate large portions of their code. For a longer change window, feel free to `import pysal._legacy as pysal`. Note that this can be deprecated at any point, and we urge you to not do this if it can be avoided. If you can make this changes, we hope you can instead make our recommended changes. They are small, reasonable, and will greatly enhance how easy it is for us to maintain `pysal`. 

### Please contact us on [gitter](https://gitter.com/pysal/pysal) if there are any remaining concerns or questions. 