import os
import pysal

__all__ = ['get_path']

dl = "http://code.google.com/p/pysal/downloads/list"
def get_path(example_name):
    base = os.path.split(pysal.__file__)[0]
    pth = os.path.join(base,'examples',example_name)
    if os.path.isfile(pth):
        print "Warning example not found: %s"%example_name
        print "PySAL examples are available at %s" % dl
    return os.path.join(base,'examples',example_name)
