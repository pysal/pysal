import numpy as np
import numpy.linalg as la
import scipy.sparse as SP
import scipy
from scipy.sparse import linalg as SPla

def spdot(a, b, array_out=True):
    """
    Matrix multiplication function to deal with sparse and dense objects

    Parameters
    ----------

    a           : array
                  first multiplication factor. Can either be sparse or dense.
    b           : array
                  second multiplication factor. Can either be sparse or dense.
    array_out   : boolean
                  If True (default) the output object is always a np.array

    Returns
    -------

    ab : array
         product of a times b. Sparse if a and b are sparse. Dense otherwise.
    """
    if type(a).__name__ == 'ndarray' and type(b).__name__ == 'ndarray':
        ab = np.dot(a, b)
    elif type(a).__name__ == 'csr_matrix' or type(b).__name__ == 'csr_matrix' \
            or type(a).__name__ == 'csc_matrix' or type(b).__name__ == 'csc_matrix':
        ab = a * b
        if array_out:
            if type(ab).__name__ == 'csc_matrix' or type(ab).__name__ == 'csr_matrix':
                ab = ab.toarray()
    else:
        raise Exception("Invalid format for 'spdot' argument: %s and %s" % (
            type(a).__name__, type(b).__name__))
    return ab


def spmultiply(a, b, array_out=True):
    """
    Element-wise multiplication function to deal with sparse and dense
    objects. Both objects must be of the same type.

    Parameters
    ----------

    a           : array
                  first multiplication factor. Can either be sparse or dense.
    b           : array
                  second multiplication factor. Can either be sparse or dense.
                  integer.
    array_out   : boolean
                  If True (default) the output object is always a np.array

    Returns
    -------

    ab : array
         elementwise multiplied object. Sparse if a is sparse. Dense otherwise.
    """
    if type(a).__name__ == 'ndarray' and type(b).__name__ == 'ndarray':
        ab = a * b
    elif (type(a).__name__ == 'csr_matrix' or type(a).__name__ == 'csc_matrix') \
            and (type(b).__name__ == 'csr_matrix' or type(b).__name__ == 'csc_matrix'):
        ab = a.multiply(b)
        if array_out:
            if type(ab).__name__ == 'csc_matrix' or type(ab).__name__ == 'csr_matrix':
                ab = ab.toarray()
    else:
        raise Exception("Invalid format for 'spmultiply' argument: %s and %s" % (
            type(a).__name__, type(b).__name__))
    return ab


def sphstack(a, b, array_out=False):
    """
    Horizontal stacking of vectors (or matrices) to deal with sparse and dense objects

    Parameters
    ----------

    a           : array or sparse matrix
                  First object.
    b           : array or sparse matrix
                  Object to be stacked next to a
    array_out   : boolean
                  If True the output object is a np.array; if False (default)
                  the output object is an np.array if both inputs are
                  arrays or CSR matrix if at least one input is a CSR matrix

    Returns
    -------

    ab          : array or sparse matrix
                  Horizontally stacked objects
    """
    if type(a).__name__ == 'ndarray' and type(b).__name__ == 'ndarray':
        ab = np.hstack((a, b))
    elif type(a).__name__ == 'csr_matrix' or type(b).__name__ == 'csr_matrix':
        ab = SP.hstack((a, b), format='csr')
        if array_out:
            if type(ab).__name__ == 'csr_matrix':
                ab = ab.toarray()
    else:
        raise Exception("Invalid format for 'sphstack' argument: %s and %s" % (
            type(a).__name__, type(b).__name__))
    return ab


def spbroadcast(a, b, array_out=False):
    """
    Element-wise multiplication of a matrix and vector to deal with sparse 
    and dense objects

    Parameters
    ----------

    a           : array or sparse matrix
                  Object with one or more columns.
    b           : array
                  Object with only one column
    array_out   : boolean
                  If True the output object is a np.array; if False (default)
                  the output object is an np.array if both inputs are
                  arrays or CSR matrix if at least one input is a CSR matrix

    Returns
    -------

    ab          : array or sparse matrix
                  Element-wise multiplication of a and b
    """
    if type(a).__name__ == 'ndarray' and type(b).__name__ == 'ndarray':
        ab = a * b
    elif type(a).__name__ == 'csr_matrix':
        b_mod = SP.lil_matrix((b.shape[0], b.shape[0]))
        b_mod.setdiag(b)
        ab = (a.T * b_mod).T
        if array_out:
            if type(ab).__name__ == 'csr_matrix':
                ab = ab.toarray()
    else:
        raise Exception("Invalid format for 'spbroadcast' argument: %s and %s" % (
            type(a).__name__, type(b).__name__))
    return ab


def spmin(a):
    """
    Minimum value in a matrix or vector to deal with sparse and dense objects

    Parameters
    ----------

    a           : array or sparse matrix
                  Object with one or more columns.

    Returns
    -------

    min a       : int or float
                  minimum value in a
    """
    return a.min()

def spmax(a):
    """
    Maximum value in a matrix or vector to deal with sparse and dense objects

    Parameters
    ----------

    a           : array or sparse matrix
                  Object with one or more columns.

    Returns
    -------

    max a       : int or float
                  maximum value in a
    """
    return a.max()

def splogdet(a):
    """
    Compute the log determinant of a large matrix. 

    Parameters
    ----------

    a       :   array or sparse matrix
                Object with one or more columns

    Returns
    -------

    log determinant of a    :   int or float
                                logged determinant of a
    """
    if SP.issparse(a):
        LU = SPla.splu(a)
        det = np.sum(np.log(np.abs(LU.U.diagonal())))
    else:
        sgn, ldet = la.slogdet(a)
        det = sgn * ldet
    return det

def spfill_diagonal(a, val):
    """
    Fill the diagonal of a sparse or dense matrix

    Parameters
    ----------

    a       :   array or sparse matrix
                Object with one or more columns
    val     :   int or float
                value with which to fill the diagonal of a

    Returns
    -------

    a with val on each element of the diagonal
    """
    if SP.issparse(a):
        a.setdiag(val)
    else:
        np.fill_diagonal(a, val)
    return a

def spinv(a):
    """
    Compute the inverse of a sparse or dense matrix

    Parameters
    ----------

    a       :   array or sparse matrix
                Object with one or more columns
    
    Returns
    -------

    ai, the inverse of a
    """
    if SP.issparse(a):
        ai = SPla.inv(a)
    else:
        ai = la.inv(a)
    return ai

def spisfinite(a):
    """
    Determine whether an array has nan or inf values

    Parameters
    ----------
    a   :   array or sparse matrix
            Object with one or more columns

    Returns
    -------
    bool denoting whether or not the array contains any NaN or inf
    """
    return np.isfinite(a.sum())

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()

