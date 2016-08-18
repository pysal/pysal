"""
Function to efficiently compute sparse categorical vairables for regression
deisgn matrices.
"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

from scipy import sparse as sp
import numpy as np
from collections import defaultdict
from functools import partial
from itertools import count

def spcategorical(index):
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
    if np.squeeze(index).ndim == 1:
        id_set = np.unique(index)
        n = len(index)
        if index.dtype.type is not np.int_:
            mapper = defaultdict(partial(next, count()))
            [mapper[each] for each in id_set]
            index = [mapper[each] for each in index]
        indptr = np.arange(n+1, dtype=int) 
        return sp.csr_matrix((np.ones(n), index, indptr))
    else:
        raise IndexError("The index %s is not understood" % index)

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

