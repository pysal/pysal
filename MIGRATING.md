# Instructions on Migrating to PySAL 2.0. 

PySAL, the Python spatial analysis library, has recently changed its package structure. 
We've changed the module structure to better reflect *what you do* with the library rather than the *academic disciplines* the components of the library come from. 
This means that we have split the `pysal` package into many smaller components, each organized around a common set of problems:
* `lib`: core functionality to work with spatial data in Python, including:
    - the construction of graphs (also known as spatial weights) from spatial data
    - computational geometry algorithms: 
         + alpha shapes
         + quadtrees, rTrees, & spherical KDTrees 
         + methods to construct graphs from scipy's Delaunay triangulation objects
         + Pure python data structures to represent planar geometric objects
    - pure python readers/writers for graph/spatial weights files and some spatial data types
* `explore`: exploratory spatial data analysis of clusters, hotspots, and spatial outliers, plus spatial statistics on graphs and point patterns. **flesh this out**
* `model`: explicitly spatial modelling tools including:
    - geographically-weighted regression, a generalized additive spatial model specification
    - spatially-correlated variance-components models, a type of spatial multilevel model with correlated components, as well as diagnostics for Bayesian modelling (Geweke, Markov Chain Monte Carlo Standard Error, Potential Scale Reduction Factors)
    - spatially-varying coefficient process models (e.g. local random effects models)
    - spatial econometric models, including:
        + mixed regressive-autoregressive (spatial lag) model
        + spatially-correlated error models
        + spatial regimes/spatial mixture models
        + seemingly-unrelated regression models
        + combinations of these various effects
    - econometric specification testing methods, including spatial Lagrange Multiplier, Anselin-Kelejian, Chow, and Jarque-Bera tests.
- `dynamics`: methods to construct & analyze the space-time dynamics of distributions of data **flesh this out**
- `viz`: methods to visualize and analyze spatial datasets **flesh this out**

This document provides a brief overview of what each of the new modules are, how they relate to the last **legacy** version of PySAL, version `1.14.4`, and what you need to do in order to keep your code running the way you expect. 

# Porting your code

## using the appropriate submodule for fresher releases & more stable dependencies

## using the standard `pysal` for a stable six-month release cycle while bringing a whole lotta friends

# Function Lookup

here is a list of the locations of all of the commonly-used modules in legacy `pysal`, and where they can be found in `pysalnext`

# I really don't want to change anything; what can I do?

This is not recommended, and should only be done by those who are planning to deprecate large portions of their code. For a longer change window, feel free to `import pysal._legacy as pysal`. Note that this can be deprecated at any point, and we urge you to not do this if it can be avoided. 

If you can make this change, we hope you can instead make our recommended changes. 