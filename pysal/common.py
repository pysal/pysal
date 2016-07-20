
# external imports

try:
    import numpy as np
    import numpy.linalg as la
except:
    print('numpy 1.3 is required')
    raise
try:
    import scipy as sp
    import scipy.stats as stats
    from cg.kdtree import KDTree
    from scipy.spatial.distance import pdist, cdist
except:
    print('scipy 0.7+ is required')
    raise

RTOL = .00001
ATOL = 1e-7

import copy
import math
import random
import sys
import time
import unittest
from warnings import warn as Warn
from functools import wraps
from pysal.core.FileIO import FileIO as popen
try:
    from patsy import PatsyError
except ImportError:
    PatsyError = Exception

#################
# Compatibility #
#################

def iteritems(d, **kwargs):
    """
    Implements six compatibility library's iteritems function

    appropriately calls either d.iteritems() or d.items() depending on the
    version of python being used. 
    """
    if sys.version_info.major < 3:
        return d.iteritems(**kwargs)
    else:
        return iter(d.items(**kwargs))

######################
# Decorators/Utils   #
######################

def simport(modname):
    """
    Safely import a module without raising an error. 

    Parameters
    -----------
    modname : str
              module name needed to import
    
    Returns
    --------
    tuple of (True, Module) or (False, None) depending on whether the import
    succeeded.

    Notes
    ------
    Wrapping this function around an iterative context or a with context would
    allow the module to be used without necessarily attaching it permanently in
    the global namespace:
    
    >>> for t,mod in simport('pandas'):
            if t:
                mod.DataFrame()
            else:
                #do alternative behavior here
            del mod #or don't del, your call

    instead of:

    >>> t, mod = simport('pandas')
    >>> if t:
            mod.DataFrame()
        else:
            #do alternative behavior here

    The first idiom makes it work kind of a like a with statement.
    """
    try:
        exec('import {}'.format(modname))
        return True, eval(modname)
    except:
        return False, None

def requires(*args, **kwargs):
    """
    Decorator to wrap functions with extra dependencies:

    Arguments
    ---------
    args : list
            list of strings containing module to import
    verbose : bool
                boolean describing whether to print a warning message on import
                failure
    Returns
    -------
    Original function is all arg in args are importable, otherwise returns a
    function that passes. 
    """
    v = kwargs.pop('verbose', True)
    wanted = copy.deepcopy(args)
    def inner(function):
        available = [simport(arg)[0] for arg in args]
        if all(available):
            return function
        else:
            def passer(*args,**kwargs):
                if v:
                    missing = [arg for i, arg in enumerate(wanted) if not available[i]]
                    print('missing dependencies: {d}'.format(d=missing))
                    print('not running {}'.format(function.__name__))
                else:
                    pass
            return passer 
    return inner
