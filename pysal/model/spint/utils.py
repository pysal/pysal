"""
Useful functions for analyzing spatial interaction data.
"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

from scipy import sparse as sp
import numpy as np
from collections import defaultdict
from functools import partial
from itertools import count


def CPC(model):
    """
    Common part of commuters based on Sorensen index
    Lenormand et al. 2012
    """
    y = model.y
    try:
        yhat = model.yhat.resahpe((-1, 1))
    except BaseException:
        yhat = model.mu((-1, 1))
    N = model.n
    YYhat = np.hstack([y, yhat])
    NCC = np.sum(np.min(YYhat, axis=1))
    NCY = np.sum(Y)
    NCYhat = np.sum(yhat)
    return (N * NCC) / (NCY + NCYhat)


def sorensen(model):
    """
    Sorensen similarity index

    For use on spatial interaction models; N = sample size
    rather than N = number of locations and normalized by N instead of N**2
    """
    try:
        y = model.y.reshape((-1, 1))
    except BaseException:
        y = model.f.reshape((-1, 1))
    try:
        yhat = model.yhat.reshape((-1, 1))
    except BaseException:
        yhat = model.mu.reshape((-1, 1))
    N = model.n
    YYhat = np.hstack([y, yhat])
    num = 2.0 * np.min(YYhat, axis=1)
    den = yhat + y
    return (1.0 / N) * (np.sum(num.reshape((-1, 1)) / den.reshape((-1, 1))))


def srmse(model):
    """
    Standardized root mean square error
    """
    n = float(model.n)
    try:
        y = model.y.reshape((-1, 1)).astype(float)
    except BaseException:
        y = model.f.reshape((-1, 1)).astype(float)
    try:
        yhat = model.yhat.reshape((-1, 1)).astype(float)
    except BaseException:
        yhat = model.mu.reshape((-1, 1)).astype(float)
    srmse = ((np.sum((y - yhat)**2) / n)**.5) / (np.sum(y) / n)
    return srmse


def spcategorical(index):
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
    if np.squeeze(index).ndim == 1:
        id_set = np.unique(index)
        n = len(index)
        # if index.dtype.type is not np.int_:
        mapper = defaultdict(partial(next, count()))
        [mapper[each] for each in id_set]
        index = [mapper[each] for each in index]
        indptr = np.arange(n + 1, dtype=int)
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
