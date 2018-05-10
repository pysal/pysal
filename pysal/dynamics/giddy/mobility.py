"""
Income mobility measures. 
"""

__author__ = "Wei Kang <weikang9009@gmail.com>, Sergio J. Rey <sjsrey@gmail.com>"

__all__ = ["markov_mobility"]

import numpy as np
import numpy.linalg as la

def markov_mobility(p, measure="P",ini=None):
    """
    Markov-based mobility indice.

    Parameters
    ----------
    p       : array
              (k, k), Markov transition probability matrix.
    measure : string
              If measure = "P", 
              :math:`M_{P} = \frac{m-\sum_{i=1}^m P_{ii}}{m-1}`;
              if measure = "D", 
              :math:`M_{D} = 1 - |\det(P)|`,
              where :math:`\det(P)` is the determinant of :math:`P`;
              if measure = "L2", 
              :math:`M_{L2} = 1  - |\lambda_2|`,
              where :math:`\lambda_2` is the second largest eigenvalue of
              :math:`P`;
              if measure = "B1", 
              :math:`M_{B1} = \frac{m-m \sum_{i=1}^m \pi_i P_{ii}}{m-1}`, 
              where :math:`\pi` is the initial income distribution;
              if measure == "B2", 
              :math:`M_{Bartholomew2} = \frac{1}{m-1} \sum_{i=1}^m \sum_{
              j=1}^m \pi_i P_{ij} |i-j|`,
              where :math:`\pi` is the initial income distribution.
    ini     : array
              (k,), initial distribution. Need to be specified if
              measure = "B1" or "B2". If not,
              the initial distribution would be treated as a uniform
              distribution.
              
    Returns
    -------
    mobi    : float
              Mobility value.
              
    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> import giddy.markov 
    >>> import mapclassify.api as mc
    >>> f = libpysal.open(libpysal.examples.get_path("usjoin.csv"))
    >>> pci = np.array([f.by_col[str(y)] for y in range(1929,2010)])
    >>> q5 = np.array([mc.Quantiles(y).yb for y in pci]).transpose()
    >>> m = giddy.markov.Markov(q5)
    >>> m.p
    matrix([[ 0.91011236,  0.0886392 ,  0.00124844,  0.        ,  0.        ],
            [ 0.09972299,  0.78531856,  0.11080332,  0.00415512,  0.        ],
            [ 0.        ,  0.10125   ,  0.78875   ,  0.1075    ,  0.0025    ],
            [ 0.        ,  0.00417827,  0.11977716,  0.79805014,  0.07799443],
            [ 0.        ,  0.        ,  0.00125156,  0.07133917,  0.92740926]])
    >>> markov_mobility(m.p, measure="P")
    0.19758992000997844
    """

    p = np.array(p)
    k = p.shape[1]
    if measure == "P":
        t = np.trace(p)
        mobi = (k - t) / (k - 1)
    elif measure == "D":
        mobi = 1 - abs(la.det(p))
    elif measure == "L2":
        w, v = la.eig(p)
        eigen_value_abs = abs(w)
        mobi = 1 - np.sort(eigen_value_abs)[-2]
    elif measure == "B1":
        if ini is None:
            ini = 1.0/k * np.ones(k)
        mobi = (k - k * np.sum(ini * np.diag(p))) / (k - 1)
    elif measure == "B2":
        mobi = 0
        if ini is None:
            ini = 1.0 / k * np.ones(k)
        for i in range(k):
            for j in range(k):
                mobi = mobi + ini[i] * p[i, j] * abs(i - j)
        mobi = mobi / (k - 1)

    return mobi