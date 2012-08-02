import os
import pysal

__all__ = ['get_path']

def get_path(example_name):
    base = os.path.split(pysal.__file__)[0]
    return os.path.join(base,'examples',example_name)
