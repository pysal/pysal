import copy

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

if __name__ == '__main__':
    
    @requires('pandas')
    def test():
        import pandas
        print('ASDF')

    @requires('thisisnotarealmodule')
    def test2():
        print('you shouldnt see this')

    test()
    test2()

