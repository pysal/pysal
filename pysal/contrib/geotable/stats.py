from ...esda import gamma as _gamma, moran as _moran, geary as _geary 
from ...esda.smoothing import assuncao_rate as _assuncao_rate

import itertools as _it

# I would like to define it like this, so that you could make a call like:
# Geary(df, 'HOVAL', 'INC', w=W), but this only works in Python3. So, I have to
# use a workaround
#def _statistic(df, *cols, stat=None, w=None, inplace=True, 
def _univariate_handler(df, cols, stat=None, w=None, inplace=True,
                        pvalue = 'sim', outvals = None, swapname='', **kwargs):
    ### Preprocess
    if not inplace:
        new_df = df.copy()
        return _univariate_handler(new_df, cols, stat=stat, w=w, pvalue='sim', 
                                   inplace=True, outvals=outvals,
                                   swapname=swapname, **kwargs)
    if w is None:
        if ('W' in df._metadata):
            w = df.W
        else:
            raise Exception('Weights not provided and no weights attached to frame! '
                            'Please provide a weight or attach a weight to the'
                            'dataframe ')
    
    ### Prep indexes
    if outvals is None:
        outvals = []
    outvals.append('_statistic')
    if pvalue.lower() in ['all', 'both', '*']: 
        raise NotImplementedError("If you want more than one type of PValue,add"
                                  " the targeted pvalue type to outvals. For example:"
                                  " Geary(df, cols=['HOVAL'], w=w, outvals=['p_z_sim', "
                                  "'p_rand']")
    # this is nontrivial, since we
    # can't know which p_value types are on the object without computing it.
    # This is because we don't flag them with @properties, so they're just
    # arbitrarily assigned post-facto. One solution might be to post-process the
    # objects, determine which pvalue types are available, and then grab them
    # all if needed. 

    outvals.append('p_'+pvalue.lower())
    if isinstance(cols, str):
        cols = [cols]

    ### Make closure around weights & apply columnwise
    def column_stat(column):
        return stat(column.values, w=w, **kwargs)
    stat_objs = df[cols].apply(column_stat)

    ### Assign into dataframe
    for col in cols:
        stat_obj = stat_objs[col]
        y = kwargs.get('y')
        if y is not None:
            col += '-' + y.name
        outcols = ['_'.join((col, val)) for val in outvals]
        for colname, attname in zip(outcols, outvals):
            df[colname] = stat_obj.__getattribute__(attname)
    if swapname is not '':
        df.columns = [_swap_ending(col, swapname) if col.endswith('_statistic') else col
                      for col in df.columns]
    return df

def _bivariate_handler(df, x, y=None, w=None, inplace=True, pvalue = 'sim', 
                       outvals = None, **kwargs):
    real_swapname = kwargs.pop('swapname', '')
    if isinstance(y, str):
        y = [y]
    if not inplace:
        new = df.copy()
        return _bivariate_handler(new, x, y=y, w=w, inplace=True,
                                  swapname=real_swapname, 
                                  pvalue=pvalue, outvals=outvals, **kwargs)
    if y is None:
        y = x
    for xi,yi in _it.product(x,y):
        if xi == yi:
            continue
        df = _univariate_handler(df, cols=xi, w=w, y=df[yi], inplace=inplace, 
                                 pvalue=pvalue, outvals=outvals, swapname='', **kwargs)
    if real_swapname is not '':
        df.columns = [_swap_ending(col, real_swapname) 
                      if col.endswith('_statistic')
                      else col for col in df.columns]
    return df

def Gamma(df, cols, w=None, inplace=True, pvalue = 'sim', outvals = None, **stat_kwds):
    return _univariate_handler(df, cols, w=w, inplace=inplace, pvalue=pvalue,
            outvals=outvals, stat=_gamma.Gamma, swapname='gamma', **stat_kwds)


def Geary(df, cols, w=None, inplace=True, pvalue='sim', outvals=None, **stat_kwds):
    return _univariate_handler(df, cols, w=w, inplace=inplace, pvalue=pvalue,
                      outvals=outvals, stat=_geary.Geary, swapname='geary', **stat_kwds)


def Moran(df, cols, w=None, inplace=True, pvalue='sim', outvals=None, **stat_kwds):
    return _univariate_handler(df, cols, w=w, inplace=inplace, pvalue=pvalue,
            outvals=outvals, stat=_moran.Moran, swapname='moran', **stat_kwds)


def Moran_Local(df, cols, w=None, inplace=True, 
                pvalue = 'sim', outvals = None, **stat_kwds):
    return _univariate_handler(df, cols, w=w, inplace=inplace, pvalue=pvalue,
            outvals=outvals, stat=_moran.Moran_Local, swapname='lmo', **stat_kwds)


def Moran_BV(df, x, y=None, w=None, inplace=True, 
             pvalue = 'sim', outvals = None, **stat_kwds):
    # everything in x to everything in y, unless y is none. 
    # if y is none, everything in x to everything else in x
    return _bivariate_handler(df, x, y=y, w=w, inplace=inplace, 
                              pvalue = pvalue, outvals = outvals, 
                              swapname='mobv', stat=_moran.Moran_BV,**stat_kwds)


def Moran_Local_BV(df, x, y=None, w=None, inplace=True, 
                   pvalue = 'sim', outvals = None, **stat_kwds):
    # everything in x to everything in y, unless y is none. 
    # if y is none, everything in x to everything else in x
    return _bivariate_handler(df, x, y=y, w=w, inplace=inplace, 
                              pvalue = pvalue, outvals = outvals, 
                              swapname='lmobv', stat=_moran.Moran_Local_BV,**stat_kwds)


def Moran_Rate(df, events, populations, w=None, inplace=True, 
               pvalue='sim', outvals=None, **stat_kwds):
    if isinstance(populations, string):
        populations = [populations] * len(events)
    if len(events) != len(populations):
        raise ValueError('There is not a one-to-one matching between events and '
                          'populations!\nEvents: {}\n\nPopulations:'
                          ' {}'.format(events, populations))
    adjusted = kwargs.pop('adjusted', True)
    if isinstance(adjusted, bool):
        adjusted = [adjusted] * len(events)
    rates = [assuncao_rate(df[e],df[pop]) if adj
             else df[e].astype(float) / df[pop] 
             for e,pop,adj in zip(events, populations, adjusted)]
    return _univariate_handler(df, rates, w=w, inplace=inplace, 
                               pvalue = pvalue, outvals = outvals, 
                               swapname='morate', stat=_moran.Moran_Rate,**stat_kwds)


def Moran_Local_Rate(df, events, populations, w=None, 
                     inplace=True, pvalue='sim', outvals=None, **stat_kwds):
    if isinstance(populations, string):
        populations = [populations] * len(events)
    if len(events) != len(populations):
        raise ValueError('There is not a one-to-one matching between events and '
                          'populations!\nEvents: {}\n\nPopulations:'
                          ' {}'.format(events, populations))
    adjusted = kwargs.pop('adjusted', True)
    if isinstance(adjusted, bool):
        adjusted = [adjusted] * len(events)
    rates = [_assuncao_rate(df[e],df[pop]) if adj
             else df[e].astype(float) / df[pop] 
             for e,pop,adj in zip(events, populations, adjusted)]
    return _univariate_handler(df, X, w=w, inplace=inplace, 
                               pvalue = pvalue, outvals = outvals, 
                               swapname='lmorate', stat=_moran.Moran_Local_Rate,**stat_kwds)

def _swap_ending(s, ending, delim='_'):
    """
    Replace the ending of a string, delimited into an arbitrary 
    number of chunks by `delim`, with the ending provided

    Parameters
    ----------
    s       :   string
                string to replace endings
    ending  :   string
                string used to replace ending of `s`
    delim   :   string
                string that splits s into one or more parts
    
    Returns
    -------
    new string where the final chunk of `s`, delimited by `delim`, is replaced
    with `ending`. 
    """
    parts = [x for x in s.split(delim)[:-1] if x != '']
    parts.append(ending)
    return delim.join(parts)
 
##############
# DOCSTRINGS #
##############

_univ_doc_template =\
""" 
Function to compute a {n} statistic on a dataframe

Arguments
---------
df          :   pandas.DataFrame
                a pandas dataframe with a geometry column
cols        :   string or list of string
                name or list of names of columns to use to compute the statistic
w           :   pysal weights object
                a weights object aligned with the dataframe. If not provided, this
                is searched for in the dataframe's metadata
inplace     :   bool
                a boolean denoting whether to operate on the dataframe inplace or to
                return a series contaning the results of the computation. If
                operating inplace, the derived columns will be named 'column_{nl}'
pvalue      :   string
                a string denoting which pvalue should be returned. Refer to the
                the {n} statistic's documentation for available p-values
outvals     :   list of strings
                list of arbitrary attributes to return as columns from the {n} statistic
**stat_kwds :   keyword arguments
                options to pass to the underlying statistic. For this, see the
                documentation for the {n} statistic.

Returns
--------
If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
returns a copy of the dataframe with the relevant columns attached.

See Also
---------
For further documentation, refer to the {n} class in pysal.esda
"""

_bv_doc_template =\
""" 
Function to compute a {n} statistic on a dataframe

Arguments
---------
df          :   pandas.DataFrame
                a pandas dataframe with a geometry column
X           :   list of strings
                column name or list of column names to use as X values to compute
                the bivariate statistic. If no Y is provided, pairwise comparisons
                among these variates are used instead. 
Y           :   list of strings
                column name or list of column names to use as Y values to compute
                the bivariate statistic. if no Y is provided, pariwise comparisons
                among the X variates are used instead. 
w           :   pysal weights object
                a weights object aligned with the dataframe. If not provided, this
                is searched for in the dataframe's metadata
inplace     :   bool
                a boolean denoting whether to operate on the dataframe inplace or to
                return a series contaning the results of the computation. If
                operating inplace, the derived columns will be named 'column_{nl}'
pvalue      :   string
                a string denoting which pvalue should be returned. Refer to the
                the {n} statistic's documentation for available p-values
outvals     :   list of strings
                list of arbitrary attributes to return as columns from the {n} statistic
**stat_kwds :   keyword arguments
                options to pass to the underlying statistic. For this, see the
                documentation for the {n} statistic.


Returns
--------
If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
returns a copy of the dataframe with the relevant columns attached.

See Also
---------
For further documentation, refer to the {n} class in pysal.esda
"""

_rate_doc_template =\
""" 
Function to compute a {n} statistic on a dataframe

Arguments
---------
df          :   pandas.DataFrame
                a pandas dataframe with a geometry column
events      :   string or  list of strings
                one or more names where events are stored
populations :   string or list of strings
                one or more names where the populations corresponding to the
                events are stored. If one population column is provided, it is
                used for all event columns. If more than one population column
                is provided but there is not a population for every event
                column, an exception will be raised.
w           :   pysal weights object
                a weights object aligned with the dataframe. If not provided, this
                is searched for in the dataframe's metadata
inplace     :   bool
                a boolean denoting whether to operate on the dataframe inplace or to
                return a series contaning the results of the computation. If
                operating inplace, the derived columns will be named 'column_{nl}'
pvalue      :   string
                a string denoting which pvalue should be returned. Refer to the
                the {n} statistic's documentation for available p-values
outvals     :   list of strings
                list of arbitrary attributes to return as columns from the {n} statistic
**stat_kwds :   keyword arguments
                options to pass to the underlying statistic. For this, see the
                documentation for the {n} statistic.

Returns
--------
If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
returns a copy of the dataframe with the relevant columns attached.

See Also
---------
For further documentation, refer to the {n} class in pysal.esda
"""
Gamma.__doc__ = _univ_doc_template.format(n='Gamma', nl='gamma')
Geary.__doc__ = _univ_doc_template.format(n='Geary', nl='geary')
Moran.__doc__ = _univ_doc_template.format(n='Moran', nl='moran')
Moran_Local.__doc__ = _bv_doc_template.format(n='Moran_Local', nl='lmo')
Moran_BV.__doc__ = _bv_doc_template.format(n='Moran_BV', nl='bv')
Moran_Local_BV.__doc__ = _bv_doc_template.format(n='Moran_Local_BV', nl='lmbv')
Moran_Rate.__doc__ = _rate_doc_template.format(n='Moran_Rate', nl='morate')
Moran_Local_Rate.__doc__ = _rate_doc_template.format(n='Moran_Local_Rate', nl='lmorate')