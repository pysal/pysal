import numpy as np


def shuffle_matrix(X,ids):
    """
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
    return X[ids,:][:,ids]


if __name__ == '__main__':

    import doctest
    doctest.testmod()

