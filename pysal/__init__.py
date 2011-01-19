"""
Python Spatial Analysis Library
===============================


Documentation
-------------
PySAL documentation is available in two forms: python docstrings and a html webpage at http://pysal.org/

Available sub-packages
----------------------

cg
    Basic data structures and tools for Computational Geometry
core
    Basic functions used by several sub-packages
esda
    Tools for Exploratory Spatial Data Analysis
examples
    Example data sets used by several sub-packages for examples and testing
region
    Regionalization algorithms and spatially constrained clustering
spatial_dynamics
    Space-time exploratory methods and clustering
weights
    Tools for creating and manipulating weights

Utilities
---------
`fileio`_
    Tool for file input and output, supports many well known file formats
"""
import cg
import core

from version import version

# toplevel imports to be explicit
from esda.moran import Moran, Moran_BV, Moran_BV_matrix, Moran_Local
from esda.geary import Geary
from esda.join_counts import Join_Counts
from esda.mapclassify import quantile,binC,bin,bin1d,Equal_Interval,Percentiles
from esda.mapclassify import Box_Plot,Quantiles,Std_Mean,Maximum_Breaks
from esda.mapclassify import Natural_Breaks, Fisher_Jenks, Jenks_Caspall
from esda.mapclassify import Jenks_Caspall_Sampled,Jenks_Caspall_Forced
from esda.mapclassify import User_Defined,Max_P_Classifier,gadf
from esda.mapclassify import K_classifiers
from inequality.theil import Theil,TheilD,TheilDSim
from region.maxp import Maxp,Maxp_LISA
from spatial_dynamics import Markov, Spatial_Markov, LISA_Markov, SpatialTau, Theta
from spatial_dynamics import ergodic
from spatial_dynamics import directional
from weights import W,lat2W,regime_weights,comb,full,shimbel,order,higher_order,remap_ids
from weights.Distance import knnW, Kernel, DistanceBand
from weights.Contiguity import buildContiguity
from weights.spatial_lag import lag_spatial
from weights.user import *
from weights.Wsets import w_union, w_intersection, w_difference
from weights.Wsets import w_symmetric_difference, w_subset

# Load the IOHandlers
import core.IOHandlers
# Assign pysal.open to dispatcher
open = core.FileIO.FileIO

#__all__=[]
#import esda,weights
#__all__+=esda.__all__
#__all__+=weights.__all__
