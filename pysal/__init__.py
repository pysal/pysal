"""
PySAL: Python Spatial Analysis Library
======================================

A federation of packages for spatial data science.


Layers and Subpackages
----------------------
PySAL is organized into four layers (lib, explore, model, and viz), each of which contains subpackages for a particular type of spatial data analysis.


Use of any of these layers requires an explicit import. For example,
``from pysal.explore import esda``

lib: core algorithms, weights, and spatial data structures
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  cg                  -- Computational Geometry
  examples            -- Example data sets
  io                  -- Input/Output
  weights             -- Spatial Weights


explore
+++++++

  esda                -- Global and local spatial autocorrelation
  giddy               -- Spatial distribution dynamics
  inequality          -- Spatial inequality measures
  pointpats           -- Planar point pattern analysis
  segregation         -- Segregation analytics
  spaghetti           -- Spatial analysis on networks


model
+++++

  access              -- Measures of spatial accessibility
  mgwr                -- Multi-scale geographically weighted regression
  spint               -- Spatial interaction modeling
  spglm               -- Spatial general linear modeling
  spvcm               -- Spatial variance component models
  spreg               -- Spatial econometrics
  tobler              -- Spatial areal interpolation models


viz
++++

  mapclassify         -- Classification schemes for choropleth maps
  splot               -- Geovisualization for pysal

"""
from .base import memberships, federation_hierarchy, versions
__version__ = '2.3.0'
