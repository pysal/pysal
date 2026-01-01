"""
pysal.lib: Python Spatial Analysis Library (core)
================================================


Documentation
-------------
PySAL documentation is available in two forms: python docstrings and an html
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

import lazy_loader as lazy

__version__ = "4.2.1"

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "libpysal": ["cg", "io", "weights", "examples"],
    },
)



