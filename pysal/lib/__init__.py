__version__ = "4.2.1"

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

_SUBMODULES = {"cg", "io", "weights", "examples"}


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib
        module = importlib.import_module(f"libpysal.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES))

