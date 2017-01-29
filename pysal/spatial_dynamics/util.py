"""
Utilities for the spatial dynamics module.
"""

__all__ = ['shuffle_matrix', 'get_lower']

import numpy as np

def shuffle_matrix(X, ids):
    """
    Random permutation of rows and columns of a matrix

    Parameters
    ----------
    X   : array
          (k, k), array to be permutated.
    ids : array
          range (k, ).

    Returns
    -------
    X   : array
          (k, k) with rows and columns randomly shuffled.

    Examples
    --------
    >>> X=np.arange(16)
    >>> X.shape=(4,4)
    >>> np.random.seed(10)
    >>> shuffle_matrix(X,range(4))
    array([[10,  8, 11,  9],
           [ 2,  0,  3,  1],
           [14, 12, 15, 13],
           [ 6,  4,  7,  5]])

    """
    np.random.shuffle(ids)
    return X[ids, :][:, ids]


def get_lower(matrix):
    """
    Flattens the lower part of an n x n matrix into an n*(n-1)/2 x 1 vector.

    Parameters
    ----------
    matrix  : array
              (n, n) numpy array, a distance matrix.

    Returns
    -------
    lowvec  : array
              numpy array, the lower half of the distance matrix flattened into
              a vector of length n*(n-1)/2.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> test = np.array([[0,1,2,3],[1,0,1,2],[2,1,0,1],[4,2,1,0]])
    >>> lower = get_lower(test)
    >>> lower
    array([[1],
           [2],
           [1],
           [4],
           [2],
           [1]])

    """
    n = matrix.shape[0]
    lowerlist = []
    for i in range(n):
        for j in range(n):
            if i > j:
                lowerlist.append(matrix[i, j])
    veclen = n * (n - 1) / 2
    lowvec = np.reshape(np.array(lowerlist), (int(veclen), 1))
    return lowvec

