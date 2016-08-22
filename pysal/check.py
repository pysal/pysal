from functools import partial
from scipy import stats as stats
from scipy import spatial as spatial
def safe_import(modname, submods=[], silent=False):
    """
    checks whether a module is importable or not and returns the module or submodule if so

    Arguments
    =========
    modname     :   str
                    name of module to import
    abbrev      :   str
                    abbreviated name of module to import
    submods     :   list of str
                    names of sub-objects in modname to import
    Returns
    =======
    module or list of functions to import.
    """
    try:
        if submods == []:
            exec('import {mod}'.format(mod=modname))
            m = eval(modname)
        elif submods != []:
            if isinstance(submods, str):
                exec('from {mod} import {submod}'.format(mod=modname, submod=submods))
                m = eval(submods)
            elif isinstance(submods, list):
                funcs = ', '.join(submods)
                exec('from {mod} import '.format(mod=modname) + funcs)
                m = [eval(term) for term in submods]
        return m
    except ImportError as E:
        if silent:
            return None
        else:
            raise E

safe_pandas = partial(safe_import, 'pandas')
safe_pandas.__doc__ = "pandas-specific version of safe_import"
safe_geopandas = partial(safe_import, 'geopandas')
safe_geopandas.__doc__ = "geopandas version of safe_import"
safe_mpl = partial(safe_import, 'matplotlib')
safe_mpl.__doc__ = "matplotlib version of safe_import"
safe_seaborn = partial(safe_import, 'seaborn')
safe_seaborn.__doc__ = "seaborn version of safe_import"
safe_cartopy = partial(safe_import, 'cartopy')
safe_cartopy.__doc__ = "cartopy version of safe_import"
safe_shapely = partial(safe_import, 'shapely')
safe_shapely.__doc__ = "shapely version of safe_import"
safe_statsmodels = partial(safe_import, 'statsmodels')
safe_statsmodels.__doc__ = "statsmodels version of safe_import"

nonecheck = partial(safe_import, 'thisisnotarealmoduleandprobablywillneverbe')
