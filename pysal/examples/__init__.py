import os
import pysal

__all__ = ['get_path']

def get_path(example_name):
    base = os.path.split(pysal.__file__)[0]
    pth = os.path.join(base,'examples',example_name)
    if os.path.isfile(pth):
        return os.path.join(base,'examples',example_name)
    else:
        print "Warning example not found: %s"%example_name
        print "PySAL examples are available at http://pysal.org"
        return None
