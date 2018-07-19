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
examples
    Example data sets for testing and documentation
io
    Basic functions used by several sub-packages
weights
    Tools for creating and manipulating weights
"""
from . import cg
from . import io
from . import weights
from . import common
from . import examples

from .io import IOHandlers

if common.pandas is not None:
    from .io import geotable

# Assign pysal.open to dispatcher
open = io.FileIO.FileIO

from .version import version
MISSINGVALUE = common.MISSINGVALUE
