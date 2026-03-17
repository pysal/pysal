import copy
import importlib

try:
    from patsy import PatsyError
except ImportError:
    PatsyError = Exception

RTOL = 0.00001
ATOL = 1e-7
MISSINGVALUE = None

######################
# Decorators/Utils   #
######################

# import numba.jit OR create mimic decorator and set existence flag
try:
    from numba import jit

    HAS_JIT = True
except ImportError:

    def jit(function=None, **kwargs):  # noqa: ARG001
        """Mimic numba.jit() with synthetic wrapper"""
        if function is not None:

            def wrapped(*original_args, **original_kw):
                """Case 1 - structure of a standard decorator
                i.e., jit(function)(*args, **kwargs)
                """
                return function(*original_args, **original_kw)

            return wrapped
        else:

            def partial_inner(func):
                """Case 2 - returns Case 1
                i.e., jit()(function)(*args, **kwargs)
                """
                return jit(func)

            return partial_inner

    HAS_JIT = False


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

        for t,mod in simport('pandas'):
            if t:
                mod.DataFrame()
            else:
                #do alternative behavior here
            del mod #or don't del, your call

    instead of:

        t, mod = simport('pandas')
        if t:
            mod.DataFrame()
        else:
            #do alternative behavior here

    The first idiom makes it work kind of a like a with statement.
    """
    try:
        mod = importlib.import_module(modname)
        return True, mod
    except (ModuleNotFoundError, ImportError):
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
    v = kwargs.pop("verbose", True)
    wanted = copy.deepcopy(args)

    def inner(function):
        available = [simport(arg)[0] for arg in args]
        if all(available):
            return function
        else:

            def passer(*args, **kwargs):  # noqa: ARG001
                if v:
                    missing = [arg for i, arg in enumerate(wanted) if not available[i]]
                    print(f"missing dependencies: {missing}")
                    print(f"not running {function.__name__}")

            return passer

    return inner
