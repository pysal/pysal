from scipy import sparse as sp
import numpy as np
from collections import defaultdict
from functools import partial
from itertools import count


def spcategorical(n_cat_ids):
    '''
    Returns a dummy matrix given an array of categorical variables.
    Parameters
    ----------
    n_cat_ids    : array
                   A 1d vector of the categorical labels for n observations; it
                   will be faster to use integers rather than strings for
                   labels.

    Returns
    --------
    dummy        : array
                   A sparse matrix of dummy (indicator/binary) variables for the
                   categorical data.  

    '''
    if np.squeeze(n_cat_ids).ndim == 1:
        cat_set = np.unique(n_cat_ids)
        n = len(n_cat_ids)
        if n_cat_ids.dtype.type is not np.int_:
            mapper = defaultdict(partial(next, count()))
            [mapper[each] for each in cat_set]
            n_cat_ids = [mapper[each] for each in n_cat_ids]
        indptr = np.arange(n+1, dtype=int) 
        return sp.csr_matrix((np.ones(n), n_cat_ids, indptr))
    else:
        raise IndexError("The index %s is not understood" % col)

#older and very slow
"""
def spcategorical(data):
    '''
    Returns a dummy matrix given an array of categorical variables.
    Parameters
    ----------
    data : array
        A 1d vector of the categorical variable.

    Returns
    --------
    dummy_matrix
        A sparse matrix of dummy (indicator/binary) float variables for the
        categorical data.  

    '''
    if np.squeeze(data).ndim == 1:
        tmp_arr = np.unique(data)
        tmp_dummy = sp.csr_matrix((0, len(data)))
        for each in tmp_arr[:, None]:
            row = sp.csr_matrix((each == data).astype(float))
            tmp_dummy = sp.vstack([tmp_dummy, row])
        tmp_dummy = tmp_dummy.T
        return tmp_dummy
    else:
        raise IndexError("The index %s is not understood" % col)

"""

#old and slow
"""
def spcategorical(n_cat_ids):
    '''
    Returns a dummy matrix given an array of categorical variables.
    Parameters
    ----------
    n_cat_ids    : array
                   A 1d vector of the categorical labels for n observations.

    Returns
    --------
    dummy        : array
                   A sparse matrix of dummy (indicator/binary) variables for the
                   categorical data.  

    '''
    if np.squeeze(n_cat_ids).ndim == 1:
        cat_set = np.unique(n_cat_ids)
        n = len(n_cat_ids)
        index = [np.where(cat_set == id)[0].tolist()[0] for id in n_cat_ids]
        indptr = np.arange(n+1, dtype=int) 
        return sp.csr_matrix((np.ones(n), index, indptr))
    else:
        raise IndexError("The index %s is not understood" % col)
"""

