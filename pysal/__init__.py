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
econometrics
    Spatial econometrics
esda
    Tools for Exploratory Spatial Data Analysis
examples
    Example data sets used by several sub-packages for examples and testing
region
    Regionalization algorithms and spatially constrained clustering
weights
    Tools for creating and manipulating weights

Utilities
---------
`fileio`_
    Tool for file input and output, supports many well known file formats
"""
import cg
import core
import pysal.core.FileIO # Load IO metaclass
import pysal.core._FileIO # Load IO inheritors

from common import *

#Assign pysal.open to dispatcher

open = pysal.core.FileIO.FileIO

# toplevel imports to be explicit

from esda.geary import Geary
from esda.join_counts import Join_Counts
from esda.mapclassify import quantile,binC,bin,bin1d,Equal_Interval
from esda.mapclassify import Box_Plot,Quantiles,Std_Mean,Maximum_Breaks
from esda.mapclassify import Natural_Breaks, Fisher_Jenks, Jenks_Caspall
from esda.mapclassify import Jenks_Caspall_Sampled,Jenks_Caspall_Forced
from esda.mapclassify import User_Defined,Max_P
from esda.moran import Moran, Moran_BV, Moran_BV_matrix, Moran_Local
from econometrics import Jarque_Bera,Ols
from inequality.theil import Theil,TheilD,TheilDSim
from markov.markov import Markov
from markov.ergodic import steady_state
from mobility.rank import Theta,SpatialTau
from region.maxp import Maxp,Maxp_LISA
from weights.weights import W,lat2gal,regime_weights,comb
from weights.DistanceWeights import InverseDistance,NearestNeighbors,DistanceBand,Kernel
from weights.spatial_lag import lag, lag_array
from weights.util import *
