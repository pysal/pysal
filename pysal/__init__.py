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
weights
    Tools for creating and manipulating weights

Utilities
---------
`fileio`_
    Tool for file input and output, supports many well known file formats
"""
import cg
import core
import econometrics
import esda
import inequality
import markov
import region
import weights

import pysal.core.FileIO # Load IO metaclass
import pysal.core._FileIO # Load IO inheritors

#Assign pysal.open to dispatcher

open = pysal.core.FileIO.FileIO


