import numpy as np
import pysal as ps
import scipy.sparse as SPARSE


def symmetrize(w):
    """Generate symmetric matrix that has same eigenvalues as an asymmetric row
    standardized matrix w

    Parameters
    ----------
    w: weights object that has been row standardized

    Returns
    -------
    a sparse symmetric matrix with same eigenvalues as w

    """
    current = w.transform
    w.transform = 'B'
    d = w.sparse.sum(axis=1)  # row sum
    d.shape = (w.n,)
    d = np.sqrt(d)
    Di12 = SPARSE.spdiags(1. / d, [0], w.n, w.n)
    D12 = SPARSE.spdiags(d, [0], w.n, w.n)
    w.transform = 'r'
    return D12 * w.sparse * Di12
