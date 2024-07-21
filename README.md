# Python Spatial Analysis Library

[![Continuous Integration](https://github.com/pysal/pysal/actions/workflows/testing.yml/badge.svg)](https://github.com/pysal/pysal/actions/workflows/testing.yml)
[![PyPI version](https://badge.fury.io/py/pysal.svg)](https://badge.fury.io/py/pysal)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pysal/badges/version.svg)](https://anaconda.org/conda-forge/pysal)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-7289da?style=flat&logo=discord&logoColor=cccccc&link=https://discord.gg/BxFTEPFFZn)](https://discord.gg/BxFTEPFFZn)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/8295380.svg)](https://zenodo.org/badge/latestdoi/8295380)

<p align="center">
<img src="https://user-images.githubusercontent.com/8590583/89052459-bad41a00-d323-11ea-9be2-beb7d0d1b7ea.png" width="300" height="300" />
</p>

PySAL, the Python spatial analysis library, is an open source cross-platform library for geospatial data science with an emphasis on geospatial vector data written in Python. It supports the development of high level applications for spatial analysis, such as

-   detection of spatial clusters, hot-spots, and outliers
-   construction of graphs from spatial data
-   spatial regression and statistical modeling on geographically embedded networks
-   spatial econometrics
-   exploratory spatio-temporal data analysis

## PySAL Components

PySAL is a family of packages for spatial data science and is divided into four major components:

### Lib

solve a wide variety of computational geometry problems including graph construction from polygonal lattices, lines, and points, construction and interactive editing of spatial weights matrices & graphs - computation of alpha shapes, spatial indices, and spatial-topological relationships, and reading and writing of sparse graph data, as well as pure python readers of spatial vector data. Unike other PySAL modules, these functions are exposed together as a single package.

-   [libpysal](https://pysal.org/libpysal) : `libpysal` provides foundational algorithms and data structures that support the rest of the library. This currently includes the following modules: input/output (`io`), which provides readers and writers for common geospatial file formats; weights (`weights`), which provides the main class to store spatial weights matrices, as well as several utilities to manipulate and operate on them; computational geometry (`cg`), with several algorithms, such as Voronoi tessellations or alpha shapes that efficiently process geometric shapes; and an additional module with example data sets (`examples`).

### Explore

The `explore` layer includes modules to conduct exploratory analysis of spatial and spatio-temporal data. At a high level, packages in `explore` are focused on enabling the user to better understand patterns in the data and suggest new interesting questions rather than answer existing ones. They include methods to characterize the structure of spatial distributions (either on networks, in continuous space, or on polygonal lattices). In addition, this domain offers methods to examine the *dynamics* of these distributions, such as how their composition or spatial extent changes over time.

-   [esda](https://pysal.org/esda/) : `esda` implements methods for the analysis of both global (map-wide) and local (focal) spatial autocorrelation, for both continuous and binary data. In addition, the package increasingly offers cutting-edge statistics about boundary strength and measures of aggregation error in statistical analyses

-   [giddy](https://pysal.org/giddy/) : `giddy` is an extension of `esda` to spatio-temporal data. The package hosts state-of-the-art methods that explicitly consider the role of space in the dynamics of distributions over time

-   [inequality](https://pysal.org/inequality/) : `inequality` provides indices for measuring inequality over space and time. These comprise classic measures such as the Theil *T* information index and the Gini index in mean deviation form; but also spatially-explicit measures that incorporate the location and spatial configuration of observations in the calculation of inequality measures.

-   [momepy](https://docs.momepy.org) : `momepy` is a library for quantitative analysis of urban form - urban morphometrics. It aims to provide a wide range of tools for a systematic and exhaustive analysis of urban form. It can work with a wide range of elements, while focused on building footprints and street networks. momepy stands for Morphological Measuring in Python.

-   [pointpats](https://pysal.org/pointpats/) : `pointpats` supports the statistical analysis of point data, including methods to characterize the spatial structure of an observed point pattern: a collection of locations where some phenomena of interest have been recorded. This includes measures of centrography which provide overall geometric summaries of the point pattern, including central tendency, dispersion, intensity, and extent.

-   [segregation](https://pysal.org/segregation/) : `segregation` package calculates over 40 different segregation indices and provides a suite of additional features for measurement, visualization, and hypothesis testing that together represent the state-of-the-art in quantitative segregation analysis.

-   [spaghetti](https://pysal.org/spaghetti) : `spaghetti` supports the the spatial analysis of graphs, networks, topology, and inference. It includes functionality for the statistical testing of clusters on networks, a robust all-to-all Dijkstra shortest path algorithm with multiprocessing functionality, and high-performance geometric and spatial computations using `geopandas` that are necessary for high-resolution interpolation along networks, and the ability to connect near-network observations onto the network

### Model

In contrast to `explore`, the `model` layer focuses on confirmatory analysis. In particular, its packages focus on the estimation of spatial relationships in data with a variety of linear, generalized-linear, generalized-additive, nonlinear, multi-level, and local regression models.

-   [mgwr](https://mgwr.readthedocs.io/en/latest/) : `mgwr` provides scalable algorithms for estimation, inference, and prediction using single- and multi-scale geographically-weighted regression models in a variety of generalized linear model frameworks, as well model diagnostics tools

-   [spglm](https://pysal.org/spglm/) : `spglm` implements a set of generalized linear regression techniques, including Gaussian, Poisson, and Logistic regression, that allow for sparse matrix operations in their computation and estimation to lower memory overhead and decreased computation time.

-   [spint](https://github.com/pysal/spint) : `spint` provides a collection of tools to study spatial interaction processes and analyze spatial interaction data. It includes functionality to facilitate the calibration and interpretation of a family of gravity-type spatial interaction models, including those with *production* constraints, *attraction* constraints, or a combination of the two.

-   [spreg](https://pysal.org/spreg/) : `spreg` supports the estimation of classic and spatial econometric models. Currently it contains methods for estimating standard Ordinary Least Squares (OLS), Two Stage Least Squares (2SLS) and Seemingly Unrelated Regressions (SUR), in addition to various tests of homokestadicity, normality, spatial randomness, and different types of spatial autocorrelation. It also includes a suite of tests for spatial dependence in models with binary dependent variables.

-   [spvcm](https://github.com/pysal/spvcm) : `spvcm` provides a general
    framework for estimating spatially-correlated variance components
    models. This class of models allows for spatial dependence in the variance
    components, so that nearby groups may affect one another. It also also
    provides a general-purpose framework for estimating models using Gibbs
    sampling in Python, accelerated by the `numba` package. 

	> ⚠️ **Warning:**
	> spvcm has been archived and is planned for deprecation and removal in pysal 25.01.



-   [tobler](http://pysal.org/tobler/) : `tobler` provides functionality for for areal interpolation and dasymetric mapping. Its name is an homage to the legendary geographer Waldo Tobler a pioneer of dozens of spatial analytical methods. `tobler` includes functionality for interpolating data using area-weighted approaches, regression model-based approaches that leverage remotely-sensed raster data as auxiliary information, and hybrid approaches.

-   [access](https://pysal.org/access/) : `access` aims to make it easy for analysis to calculate measures of spatial accessibility. This work has traditionally had two challenges: [1] to calculate accurate travel time matrices at scale and [2] to derive measures of access using the travel times and supply and demand locations. `access` implements classic spatial access models, allowing easy comparison of methodologies and assumptions.

-   [spopt](https://pysal.org/spopt/): `spopt`  is an open-source Python library for solving optimization problems with spatial data. Originating
    from the original `region` module in PySAL, it is under active development for the inclusion of newly proposed models and methods for regionalization, facility location, and transportation-oriented solutions.

### Viz

The `viz` layer provides functionality to support the creation of geovisualisations and visual representations of outputs from a variety of spatial analyses. Visualization plays a central role in modern spatial/geographic data science. Current packages provide classification methods for choropleth mapping and a common API for linking PySAL outputs to visualization tool-kits in the Python ecosystem.

-   [legendgram](https://github.com/pysal/legendgram) : `legendgram` is a small package that provides "legendgrams" legends that visualize the distribution of observations by color in a given map. These distributional visualizations for map classification schemes assist in analytical cartography and spatial data visualization

-   [mapclassify](https://pysal.org/mapclassify) : `mapclassify` provides functionality for Choropleth map classification. Currently, fifteen different classification schemes are available, including a highly-optimized implementation of Fisher-Jenks optimal classification. Each scheme inherits a common structure that ensures computations are scalable and supports applications in streaming contexts.

-   [splot](https://splot.readthedocs.io/en/latest/) : `splot` provides statistical visualizations for spatial analysis. It methods for visualizing global and local spatial autocorrelation (through Moran scatterplots and cluster maps), temporal analysis of cluster dynamics (through heatmaps and rose diagrams), and multivariate choropleth mapping (through value-by-alpha maps. A high level API supports the creation of publication-ready visualizations

# Installation

PySAL is available through [Anaconda](https://www.continuum.io/downloads) (in the defaults or conda-forge channel) We recommend installing PySAL from conda-forge:

``` {.sourceCode .bash}
conda config --add channels conda-forge
conda install pysal
```

PySAL can also be installed using pip:

``` {.sourceCode .bash}
pip install pysal
```

As of version 2.0.0 PySAL has shifted to Python 3 only.

Users who need an older stable version of PySAL that is Python 2 compatible can install version 1.14.3 through pip or conda:

``` {.sourceCode .bash}
conda install pysal==1.14.3
```

# Documentation

For help on using PySAL, check out the following resources:

-   [Project Home](https://pysal.org)
-   [Users](http://pysal.org/docs/users)
-   [Developers](http://pysal.org/docs/devs/)

# Development

As of version 2.0.0, PySAL is now a collection of affiliated geographic data science packages. Changes to the code for any of the subpackages should be directed at the respective [upstream repositories](http://github.com/pysal/help), and not made here. Infrastructural changes for the meta-package, like those for tooling, building the package, and code standards, will be considered.

Development is hosted on [github](https://github.com/pysal/pysal).

Discussions of development as well as help for users occurs on the [developer list](http://groups.google.com/group/pysal-dev) as well as in [PySAL's Discord channel](https://discord.gg/BxFTEPFFZn).

# Getting Involved

If you are interested in contributing to PySAL please see our [development guidelines](https://github.com/pysal/pysal/wiki).

# Bug reports

To search for or report bugs, please see PySAL\'s [issues](http://github.com/pysal/pysal/issues).

# Build Instructions

To build the meta-package pysal see [tools/README.md](tools/README.md).

# License information

See the file \"LICENSE.txt\" for information on the history of this software, terms & conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.
