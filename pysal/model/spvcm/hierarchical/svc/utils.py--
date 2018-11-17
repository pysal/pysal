import numpy as np

def explode(X):
    """
    Convert an (n,p) design matrix into an (n, n*p) local design matrix
    """
    n,p = X.shape
    Xs = X
    X = np.zeros((n, n*p))
    for i,row in enumerate(Xs):
        X[i, i*p:i*p+p] = row
    return X

def nexp(phi, pwds):
    """
    use the nexgative exponential distance weighting function on a matrix of
    pairwise distances to generate a spatial-dependence matrix for a svcp 
    """
    H = np.exp(- pwds / phi)
    return H 
