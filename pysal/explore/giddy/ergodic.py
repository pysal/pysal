"""
Summary measures for ergodic Markov chains
"""
__author__ = "Sergio J. Rey <sjsrey@gmail.com>"

__all__ = ['steady_state', 'fmpt', 'var_fmpt']

import numpy as np
import numpy.linalg as la


def steady_state(P):
    """
    Calculates the steady state probability vector for a regular Markov
    transition matrix P.

    Parameters
    ----------
    P        : array
               (k, k), an ergodic Markov transition probability matrix.

    Returns
    -------
             : array
               (k, ), steady state distribution.

    Examples
    --------
    Taken from :cite:`Kemeny1967`. Land of Oz example where the states are
    Rain, Nice and Snow, so there is 25 percent chance that if it
    rained in Oz today, it will snow tomorrow, while if it snowed today in
    Oz there is a 50 percent chance of snow again tomorrow and a 25
    percent chance of a nice day (nice, like when the witch with the monkeys
    is melting).

    >>> import numpy as np
    >>> from pysal.explore.giddy.ergodic import steady_state
    >>> p=np.array([[.5, .25, .25],[.5,0,.5],[.25,.25,.5]])
    >>> steady_state(p)
    array([0.4, 0.2, 0.4])

    Thus, the long run distribution for Oz is to have 40 percent of the
    days classified as Rain, 20 percent as Nice, and 40 percent as Snow
    (states are mutually exclusive).

    """

    v, d = la.eig(np.transpose(P))
    d = np.array(d)

    # for a regular P maximum eigenvalue will be 1
    mv = max(v)
    # find its position
    i = v.tolist().index(mv)

    row = abs(d[:, i])

    # normalize eigenvector corresponding to the eigenvalue 1
    return row / sum(row)


def fmpt(P):
    """
    Calculates the matrix of first mean passage times for an ergodic transition
    probability matrix.

    Parameters
    ----------
    P    : array
           (k, k), an ergodic Markov transition probability matrix.

    Returns
    -------
    M    : array
           (k, k), elements are the expected value for the number of intervals
           required for a chain starting in state i to first enter state j.
           If i=j then this is the recurrence time.

    Examples
    --------
    >>> import numpy as np
    >>> from pysal.explore.giddy.ergodic import fmpt
    >>> p=np.array([[.5, .25, .25],[.5,0,.5],[.25,.25,.5]])
    >>> fm=fmpt(p)
    >>> fm
    array([[2.5       , 4.        , 3.33333333],
           [2.66666667, 5.        , 2.66666667],
           [3.33333333, 4.        , 2.5       ]])

    Thus, if it is raining today in Oz we can expect a nice day to come
    along in another 4 days, on average, and snow to hit in 3.33 days. We can
    expect another rainy day in 2.5 days. If it is nice today in Oz, we would
    experience a change in the weather (either rain or snow) in 2.67 days from
    today. (That wicked witch can only die once so I reckon that is the
    ultimate absorbing state).

    Notes
    -----
    Uses formulation (and examples on p. 218) in :cite:`Kemeny1967`.

    """

    P = np.matrix(P)
    k = P.shape[0]
    A = np.zeros_like(P)
    ss = steady_state(P).reshape(k, 1)
    for i in range(k):
        A[:, i] = ss
    A = A.transpose()
    I = np.identity(k)
    Z = la.inv(I - P + A)
    E = np.ones_like(Z)
    A_diag = np.diag(A)
    A_diag = A_diag + (A_diag == 0)
    D = np.diag(1. / A_diag)
    Zdg = np.diag(np.diag(Z))
    M = (I - Z + E * Zdg) * D
    return np.array(M)


def var_fmpt(P):
    """
    Variances of first mean passage times for an ergodic transition
    probability matrix.

    Parameters
    ----------
    P      : array
             (k, k), an ergodic Markov transition probability matrix.

    Returns
    -------
           : array
             (k, k), elements are the variances for the number of intervals
             required for a chain starting in state i to first enter state j.

    Examples
    --------
    >>> import numpy as np
    >>> from pysal.explore.giddy.ergodic import var_fmpt
    >>> p=np.array([[.5, .25, .25],[.5,0,.5],[.25,.25,.5]])
    >>> vfm=var_fmpt(p)
    >>> vfm
    array([[ 5.58333333, 12.        ,  6.88888889],
           [ 6.22222222, 12.        ,  6.22222222],
           [ 6.88888889, 12.        ,  5.58333333]])

    Notes
    -----
    Uses formulation (and examples on p. 83) in :cite:`Kemeny1967`.


    """

    P = np.matrix(P)
    A = P ** 1000
    n, k = A.shape
    I = np.identity(k)
    Z = la.inv(I - P + A)
    E = np.ones_like(Z)
    D = np.diag(1. / np.diag(A))
    Zdg = np.diag(np.diag(Z))
    M = (I - Z + E * Zdg) * D
    ZM = Z * M
    ZMdg = np.diag(np.diag(ZM))
    W = M * (2 * Zdg * D - I) + 2 * (ZM - E * ZMdg)
    return np.array(W - np.multiply(M, M))
