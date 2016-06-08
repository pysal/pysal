
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
# Decorators         #
######################

def intercept_filepath(f):
    """
    Intercept the first argument of a function if it looks like a string path

    Arguments
    ----------
    f       :   callable
                function to be wrapped

    Returns
    --------
    a function similar to f, but one that attempts to open its first argument as
    a file if it is a string.
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        iargs = iter(args)
        first = next(iargs)
        rest = list(iargs)
        if isinstance(first, str):
            try:
                first = popen(first)
            except IOError:
                pass
        return f(first, *rest, **kwargs)
    return wrapped

def coerce_input_types(f, *ptypes, **kwtypes):
    """
    Coerce input types of a function to given types, either by keyword or by
    position. 

    Arguments
    ---------
    f       : callable
              function to be wrapped
    ptypes  : types or type constructors
              types corresponding in position to the arguments needed to be
              cast.
    kwtypes : types or type constructors
              types corresponding in key to the arguments needed to be cast.
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        args = list(args)
        for i, typ in enumerate(ptypes):
            if typ is np.array:
                typ = np.asarray
            elif typ is None:
                continue
            args[i] = typ(args[i])
        for name, typ in kwtypes.items():
            if typ is np.array:
                typ = np.asarray
            kwargs[name] = typ(kwargs[name])
        return f(*args, **kwargs)

def intercept_formula(f):
    """
    Intercept the first argument of a function if it looks like a string patsy
    formula

    Arguments
    ----------
    f       :   callable
                function to be wrapped

    Returns
    --------
    a function similar to f, but one that first determines whether a `df`
    keyword argument was provided and, if so, attempts to read the first
    argument as if it were a formula.
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        df = kwargs.pop('df', None)
        if df is not None:
            iargs = iter(args)
            first = next(iargs)
            rest = list(args)
            try:
                from patsy import dmatrices
                y, X = dmatrices(first, df)
            except ImportError:
                raise ImportError("No module named 'patsy'. Required to use"
                                  " formula interface")
            except PatsyError:
                Warn('Formula not parsed successfully')
            return f(y, X, *rest, **kwargs)
        elif df is None:
            return f(*args, **kwargs)
    return wrapped

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

def requires(f):
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
    @wraps(f)
    def inner(*args, **kwargs):
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
