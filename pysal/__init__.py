"""
Python Spatial Analysis Library
===============================


Documentation
-------------
PySAL documentation is available in two forms: python docstrings and an html \
        webpage at http://pysal.org/

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
network
    Spatial analysis on networks
region
    Regionalization algorithms and spatially constrained clustering
spatial_dynamics
    Space-time exploratory methods and clustering
spreg
    Spatial regression and econometrics
weights
    Tools for creating and manipulating weights
contrib
    Package for interfacing with third-party libraries

Utilities
---------
`fileio`_
    Tool for file input and output, supports many well known file formats
"""
import pysal.cg
import pysal.core

from pysal.version import version
#from pysal.version import stable_release_date
#import urllib2, json
#import config
#import datetime
#import os, sys

# toplevel imports to be explicit
from pysal.esda.moran import Moran, Moran_BV, Moran_BV_matrix, Moran_Local, Moran_Local_BV
from pysal.esda.geary import Geary
from pysal.esda.join_counts import Join_Counts
from pysal.esda.gamma import Gamma
from pysal.esda.getisord import G, G_Local
from pysal.esda.mapclassify import quantile, binC, bin, bin1d, Equal_Interval, \
    Percentiles
from pysal.esda.mapclassify import Box_Plot, Quantiles, Std_Mean, Maximum_Breaks
from pysal.esda.mapclassify import Natural_Breaks, Fisher_Jenks, Jenks_Caspall
from pysal.esda.mapclassify import Jenks_Caspall_Sampled, Jenks_Caspall_Forced
from pysal.esda.mapclassify import User_Defined, Max_P_Classifier, gadf
from pysal.esda.mapclassify import K_classifiers
from pysal.inequality.theil import Theil, TheilD, TheilDSim
from pysal.region.maxp import Maxp, Maxp_LISA
from pysal.spatial_dynamics import Markov, Spatial_Markov, LISA_Markov, \
    SpatialTau, Theta, Tau
from pysal.spatial_dynamics import ergodic
from pysal.spatial_dynamics import directional
from pysal.weights import W, lat2W, block_weights, comb, full, shimbel, \
    order, higher_order, higher_order_sp, remap_ids, hexLat2W, WSP, regime_weights
from pysal.weights.Distance import knnW, Kernel, DistanceBand
from pysal.weights.Contiguity import buildContiguity
from pysal.weights.spatial_lag import lag_spatial
from pysal.weights.Wsets import w_union, w_intersection, w_difference
from pysal.weights.Wsets import w_symmetric_difference, w_subset
from pysal.weights.user import queen_from_shapefile, rook_from_shapefile, \
    knnW_from_array, knnW_from_shapefile, threshold_binaryW_from_array,\
    threshold_binaryW_from_shapefile, threshold_continuousW_from_array,\
    threshold_continuousW_from_shapefile, kernelW, kernelW_from_shapefile,\
    adaptive_kernelW, adaptive_kernelW_from_shapefile,\
    min_threshold_dist_from_shapefile, build_lattice_shapefile
from pysal.core.util.weight_converter import weight_convert
import pysal.spreg
import pysal.examples
from pysal.network.network import Network, NetworkG, NetworkK, NetworkF

try:
    import pandas
    from pysal.contrib import pdutilities as pdio
except ImportError:
    print('Pandas adapters not loaded')

# Load the IOHandlers
from pysal.core import IOHandlers
# Assign pysal.open to dispatcher
open = pysal.core.FileIO.FileIO

#__all__=[]
#import esda,weights
#__all__+=esda.__all__
#__all__+=weights.__all__

# Constants
MISSINGVALUE = None  # used by fileIO to flag missing values.

# Load stale and other possible messages at import

"""
base_path = os.path.split(pysal.__file__)[0]
config_path = os.path.join(base_path, 'config.py')

def query_yes_no(question):
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])
    while True:
        sys.stdout.write(question)
        choice = raw_input().lower()
        if choice in yes:
            turn_off_check()
            break
        elif choice in no:
            break
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'.\n")

def turn_off_check():
    if os.path.isfile(config_path):
        f = open(config_path, 'w')
        f.write("check_stable=False")
        f.close()
        pass
    else:
        print('Cannot find config.py. Please set value manually.')

def check_version():
    today = datetime.date.today()
    delta = datetime.timedelta(days=180)
    diff = (today - stable_release_date).days
    releases = int(diff)/180
    if today - delta > stable_release_date:
	    print("Your version of PySAL is %d days old.") % diff 
	    print("There have likely been %d new release(s).") % releases 
	    print("Suppress this by setting check_stable=False in config.py.")  
	    #query_yes_no("Disable this check? [Y/n]")
    else:
        pass

def check_remote_version():
    print("Checking web for last stable release....")
    try:
        url = 'http://pypi.python.org/pypi/pysal/json'
        request = urllib2.urlopen(url)
        data = json.load(request)
        newest = data['info']['version']
        late = 'The most recent stable release is %s.' %newest
        print(late)
    except:
        print("Machine is offline. I am unable to check for the latest version of PySAL")

if config.check_stable:
    check_version()
else:
    pass
"""
