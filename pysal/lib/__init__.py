__version__ = "4.2.1"


# __version__ has to be define in the first line

"""
pysal.lib: Python Spatial Analysis Library (core)
================================================


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


import importlib
from typing import List

# Compat re-exports from libpysal
_SUBMODULES = {"cg", "io", "weights", "examples"}

__all__ = sorted(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"libpysal.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + list(_SUBMODULES))
