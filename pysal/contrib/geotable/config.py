import ops

def set_preference(pkg):
    """
    Set which provider of geospatial operations should be preferred. 

    Arguments
    ---------
    pkg     :   string
                string specifying whether to prefer computing properties using
                shapely, or to use attribute access on the polygons themselves.
    Returns
    --------
    None, sets configuration option directly
    """
    if pkg.lower() == 'shapely':
        ops.atomic.__dict__.update({k:v for k,v in
                                    ops.atomic._s.__dict__.items() 
                                    if not k.startswith('_')})
        ops.atomic._preferred = ops.atomic._s
    elif pkg.lower().startswith('att'):
        ops.atomic.__dict__.update({k:v for k,v in
                                    ops.atomic._a.__dict__.items() 
                                    if not k.startswith('_')})
        ops.atomic._preferred = ops.atomic._a
    else:
        raise Exception('Provider not recognized.')

def get_provider(fn=None):
    """
    Discover the provider of a given function

    Arguments
    ---------
    fn      :   Callable or None
                either a function whose provider needs to be discovered, or None

    Returns
    -------
    the module where fn is defined, or the current preferred provider. 

    Example
    -------

    >>> from pysal.contrib.geotable import ops
    >>> from pysal.contrib.config import get_provider, set_config
    >>> get_provider(ops.atomic.area)
        <module 'pysal.contrib.geotable.ops._accessors' from 
         pysal/contrib/geotable/ops/_accessors.pyc'>
    >>> set_preference('shapely')
    >>> get_provider(ops.atomic.area)
        <module 'pysal.contrib.geotable.ops._shapely' from
        'pysal/contrib/geotable/ops/_shapely.pyc'>
    """
    if fn is None:
        return ops.atomic._preferred
    try:
        fprovenance = fn.func.func_name
    except AttributeError:
        fprovenance = fn.func_name
    if fprovenance == 'get_attr':
        return ops._accessors
    else:
        return ops._shapely
