# Python Spatial Analysis Library

![travis (.org)](https://img.shields.io/travis/pysal/pysal)
![pypi - python version](https://img.shields.io/pypi/pyversions/pysal)
![pypi](https://img.shields.io/pypi/v/pysal)
![conda (channel only)](https://img.shields.io/conda/vn/conda-forge/pysal) 
![gitter](https://img.shields.io/gitter/room/pysal/pysal)
[![doi](https://zenodo.org/badge/8295380.svg)](https://zenodo.org/badge/latestdoi/8295380)

<p align="center">
<img src="figs/pysal_logo.png" width="300" height="300" />
</p>

pysal, the python spatial analysis library, is an open source cross-platform library for geospatial data science with an emphasis on geospatial vector data written in python. it supports the development of high level applications for spatial analysis, such as

-   detection of spatial clusters, hot-spots, and outliers
-   construction of graphs from spatial data
-   spatial regression and statistical modeling on geographically embedded networks
-   spatial econometrics
-   exploratory spatio-temporal data analysis

## pysal components

pysal is a family of packages for spatial data science and is divided into four major components:

### lib

solve a wide variety of computational geometry problems including graph construction from polygonal lattices, lines, and points, construction and interactive editing of spatial weights matrices & graphs - computation of alpha shapes, spatial indices, and spatial-topological relationships, and reading and writing of sparse graph data, as well as pure python readers of spatial vector data. unike other pysal modules, these functions are exposed together as a single package.

-   [libpysal](https://pysal.org/libpysal) : `libpysal` provides foundational algorithms and data structures that support the rest of the library. this currently includes the following modules: input/output (`io`), which provides readers and writers for common geospatial file formats; weights (`weights`), which provides the main class to store spatial weights matrices, as well as several utilities to manipulate and operate on them; computational geometry (`cg`), with several algorithms, such as voronoi tessellations or alpha shapes that efficiently process geometric shapes; and an additional module with example data sets (`examples`).

### explore

the `explore` layer includes modules to conduct exploratory analysis of spatial and spatio-temporal data. at a high level, packages in `explore` are focused on enabling the user to better understand patterns in the data and suggest new interesting questions rather than answer existing ones. they include methods to characterize the structure of spatial distributions (either on networks, in continuous space, or on polygonal lattices). in addition, this domain offers methods to examine the *dynamics* of these distributions, such as how their composition or spatial extent changes over time.

-   [esda](https://esda.readthedocs.io/en/latest/) : `esda` implements methods for the analysis of both global (map-wide) and local (focal) spatial autocorrelation, for both continuous and binary data. in addition, the package increasingly offers cutting-edge statistics about boundary strength and measures of aggregation error in statistical analyses

-   [giddy](https://giddy.readthedocs.io/en/latest/) : `giddy` is an extension of `esda` to spatio-temporal data. the package hosts state-of-the-art methods that explicitly consider the role of space in the dynamics of distributions over time

-   [inequality](https://inequality.readthedocs.io/en/latest/) : `inequality` provides indices for measuring inequality over space and time. these comprise classic measures such as the theil *t* information index and the gini index in mean deviation form; but also spatially-explicit measures that incorporate the location and spatial configuration of observations in the calculation of inequality measures.

-   [pointpats](https://pointpats.readthedocs.io/en/latest/) : `pointpats` supports the statistical analysis of point data, including methods to characterize the spatial structure of an observed point pattern: a collection of locations where some phenomena of interest have been recorded. this includes measures of centrography which provide overall geometric summaries of the point pattern, including central tendency, dispersion, intensity, and extent.

-   [segregation](https://github.com/pysal/segregation) : `segregation` package calculates over 40 different segregation indices and provides a suite of additional features for measurement, visualization, and hypothesis testing that together represent the state-of-the-art in quantitative segregation analysis.

-   [spaghetti](https://pysal.org/spaghetti) : `spaghetti` supports the the spatial analysis of graphs, networks, topology, and inference. it includes functionality for the statistical testing of clusters on networks, a robust all-to-all dijkstra shortest path algorithm with multiprocessing functionality, and high-performance geometric and spatial computations using `geopandas` that are necessary for high-resolution interpolation along networks, and the ability to connect near-network observations onto the network

### model

in contrast to `explore`, the `model` layer focuses on confirmatory analysis. in particular, its packages focus on the estimation of spatial relationships in data with a variety of linear, generalized-linear, generalized-additive, nonlinear, multi-level, and local regression models.

-   [mgwr](https://mgwr.readthedocs.io/en/latest/) : `mgwr` provides scalable algorithms for estimation, inference, and prediction using single- and multi-scale geographically-weighted regression models in a variety of generalized linear model frameworks, as well model diagnostics tools

-   [spglm](https://github.com/pysal/spglm) : `spglm` implements a set of generalized linear regression techniques, including gaussian, poisson, and logistic regression, that allow for sparse matrix operations in their computation and estimation to lower memory overhead and decreased computation time.

-   [spint](https://github.com/pysal/spint) : `spint` provides a collection of tools to study spatial interaction processes and analyze spatial interaction data. it includes functionality to facilitate the calibration and interpretation of a family of gravity-type spatial interaction models, including those with *production* constraints, *attraction* constraints, or a combination of the two.

-   [spreg](https://spreg.readthedocs.io/) : `spreg` supports the estimation of classic and spatial econometric models. currently it contains methods for estimating standard ordinary least squares (ols), two stage least squares (2sls) and seemingly unrelated regressions (sur), in addition to various tests of homokestadicity, normality, spatial randomness, and different types of spatial autocorrelation. it also includes a suite of tests for spatial dependence in models with binary dependent variables.

-   [spvcm](https://github.com/pysal/spvcm) : `spvcm` provides a general framework for estimating spatially-correlated variance components models. this class of models allows for spatial dependence in the variance components, so that nearby groups may affect one another. it also also provides a general-purpose framework for estimating models using gibbs sampling in python, accelerated by the `numba` package.

-   [tobler](http://pysal.org/tobler/) : `tobler` provides functionality for for areal interpolation and dasymetric mapping. its name is an homage to the legendary geographer waldo tobler a pioneer of dozens of spatial analytical methods. `tobler` includes functionality for interpolating data using area-weighted approaches, regression model-based approaches that leverage remotely-sensed raster data as auxiliary information, and hybrid approaches.

-   [access](http://github.com/pysal/access) : `access` aims to make it easy for analysis to calculate measures of spatial accessibility. this work has traditionally had two challenges: [1] to calculate accurate travel time matrices at scale and [2] to derive measures of access using the travel times and supply and demand locations. `access` implements classic spatial access models, allowing easy comparison of methodologies and assumptions.

### viz

the `viz` layer provides functionality to support the creation of geovisualisations and visual representations of outputs from a variety of spatial analyses. visualization plays a central role in modern spatial/geographic data science. current packages provide classification methods for choropleth mapping and a common api for linking pysal outputs to visualization tool-kits in the python ecosystem.

-   [legendgram](https://github.com/pysal/legendgram) : `legendgram` is a small package that provides "legendgrams" legends that visualize the distribution of observations by color in a given map. these distributional visualizations for map classification schemes assist in analytical cartography and spatial data visualization

-   [mapclassify](https://pysal.org/mapclassify) : `mapclassify` provides functionality for choropleth map classification. currently, fifteen different classification schemes are available, including a highly-optimized implementation of fisher-jenks optimal classification. each scheme inherits a common structure that ensures computations are scalable and supports applications in streaming contexts.

-   [splot](https://splot.readthedocs.io/en/latest/) : `splot` provides statistical visualizations for spatial analysis. it methods for visualizing global and local spatial autocorrelation (through moran scatterplots and cluster maps), temporal analysis of cluster dynamics (through heatmaps and rose diagrams), and multivariate choropleth mapping (through value-by-alpha maps. a high level api supports the creation of publication-ready visualizations

# installation

pysal is available through [anaconda](https://www.continuum.io/downloads) (in the defaults or conda-forge channel) we recommend installing pysal from conda-forge:

``` {.sourcecode .bash}
conda config --add channels conda-forge
conda install pysal
```

pysal can also be installed using pip:

``` {.sourcecode .bash}
pip install pysal
```

as of version 2.0.0 pysal has shifted to python 3 only.

users who need an older stable version of pysal that is python 2 compatible can install version 1.14.3 through pip or conda:

``` {.sourcecode .bash}
conda install pysal==1.14.3
```

# documentation

for help on using pysal, check out the following resources:

-   [user guide](https://pysal.org/pysal)
-   [tutorials and short courses](https://github.com/pysal/notebooks)

# development

as of version 2.0.0, pysal is now a collection of affiliated geographic data science packages. changes to the code for any of the subpackages should be directed at the respective [upstream repositories](http://github.com/pysal/help), and not made here. infrastructural changes for the meta-package, like those for tooling, building the package, and code standards, will be considered.

development is hosted on [github](https://github.com/pysal/pysal).

discussions of development as well as help for users occurs on the [developer list](http://groups.google.com/group/pysal-dev) as well as [gitter](https://gitter.im/pysal/pysal?).

# getting involved

if you are interested in contributing to pysal please see our [development guidelines](https://github.com/pysal/pysal/wiki).

# bug reports

to search for or report bugs, please see pysal\'s [issues](http://github.com/pysal/pysal/issues).

# build instructions

to build the meta-package pysal see [tools/readme.md](tools/readme.md).

# license information

see the file \"license.txt\" for information on the history of this software, terms & conditions for usage, and a disclaimer of all warranties.
