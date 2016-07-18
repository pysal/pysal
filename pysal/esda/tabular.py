#from ...common import requires as _requires

import itertools as _it
from pysal.weights import W

# I would like to define it like this, so that you could make a call like:
# Geary(df, 'HOVAL', 'INC', w=W), but this only works in Python3. So, I have to
# use a workaround
#def _statistic(df, *cols, stat=None, w=None, inplace=True, 
def _univariate_handler(df, cols, stat=None, w=None, inplace=True,
                        pvalue = 'sim', outvals = None, swapname='', **kwargs):
    """
    Compute a univariate descriptive statistic `stat` over columns `cols` in
    `df`.

    Parameters
    ----------
    df          : pandas.DataFrame
                  the dataframe containing columns to compute the descriptive
                  statistics
    cols        : string or list of strings
                  one or more names of columns in `df` to use to compute
                  exploratory descriptive statistics. 
    stat        : callable
                  a function that takes data as a first argument and any number
                  of configuration keyword arguments and returns an object
                  encapsulating the exploratory statistic results 
    w           : pysal.weights.W
                  the spatial weights object corresponding to the dataframe
    inplace     : bool
                  a flag denoting whether to add the statistic to the dataframe
                  in memory, or to construct a copy of the dataframe and append
                  the results to the copy
    pvalue      : string
                  the name of the pvalue on the results object wanted
    outvals     : list of strings
                  names of attributes of the dataframe to attempt to flatten
                  into a column
    swapname    : string
                  suffix to replace generic identifier with. Each caller of this
                  function should set this to a unique column suffix
    **kwargs    : optional keyword arguments
                  options that are passed directly to the statistic
    """
    ### Preprocess
    if not inplace:
        new_df = df.copy()
        _univariate_handler(new_df, cols, stat=stat, w=w, pvalue=pvalue, 
                                   inplace=True, outvals=outvals,
                                   swapname=swapname, **kwargs)
        return new_df
    if w is None:
        for name in df._metadata:
            this_obj = df.__dict__.get(name)
            if isinstance(this_obj, W):
                w = this_obj
    if w is None:
        raise Exception('Weights not provided and no weights attached to frame!'
                            ' Please provide a weight or attach a weight to the'
                            ' dataframe')
    ### Prep indexes
    if outvals is None:
        outvals = []
    outvals.insert(0,'_statistic')
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
    
    if pvalue is not '':
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

def _bivariate_handler(df, x, y=None, w=None, inplace=True, pvalue='sim', 
                       outvals=None, **kwargs):
    """
    Compute a descriptive bivariate statistic over two sets of columns, `x` and
    `y`, contained in `df`. 

    Parameters
    ----------
    df          : pandas.DataFrame
                  dataframe in which columns `x` and `y` are contained
    x           : string or list of strings
                  one or more column names to use as variates in the bivariate
                  statistics
    y           : string or list of strings
                  one or more column names to use as variates in the bivariate
                  statistics
    w           : pysal.weights.W
                  spatial weights object corresponding to the dataframe `df`
    inplace     : bool
                  a flag denoting whether to add the statistic to the dataframe
                  in memory, or to construct a copy of the dataframe and append
                  the results to the copy
    pvalue      : string
                  the name of the pvalue on the results object wanted
    outvals     : list of strings
                  names of attributes of the dataframe to attempt to flatten
                  into a column
    swapname    : string
                  suffix to replace generic identifier with. Each caller of this
                  function should set this to a unique column suffix
    **kwargs    : optional keyword arguments
                  options that are passed directly to the statistic
    """
    real_swapname = kwargs.pop('swapname', '')
    if isinstance(y, str):
        y = [y]
    if isinstance(x, str):
        x = [x]
    if not inplace:
        new_df = df.copy()
        _bivariate_handler(new_df, x, y=y, w=w, inplace=True,
                           swapname=real_swapname, 
                           pvalue=pvalue, outvals=outvals, **kwargs)
        return new_df
    if y is None:
        y = x
    for xi,yi in _it.product(x,y):
        if xi == yi:
            continue
        _univariate_handler(df, cols=xi, w=w, y=df[yi], inplace=True, 
                            pvalue=pvalue, outvals=outvals, swapname='', **kwargs)
    if real_swapname is not '':
        df.columns = [_swap_ending(col, real_swapname) 
                      if col.endswith('_statistic')
                      else col for col in df.columns]

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
                list of arbitrary attributes to return as columns from the 
                {n} statistic
**stat_kws  :   keyword arguments
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
                list of arbitrary attributes to return as columns from the 
                {n} statistic
**stat_kws  :   keyword arguments
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
                list of arbitrary attributes to return as columns from the 
                {n} statistic
**stat_kws  :   keyword arguments
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
