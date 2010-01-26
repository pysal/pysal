import numpy as num
import numpy.linalg as la


def steady_state(P):
    """Returns the steady state probability vector for a regular Markov
    transition matrix P"""

    v,d=la.eig(num.transpose(P))

    # for a regular P maximum eigenvalue will be 1
    mv=max(v)
    # find its position
    i=v.tolist().index(mv)

    # normalize eigenvector corresponding to the eigenvalue 1
    return d[:,i]/sum(d[:,i])
