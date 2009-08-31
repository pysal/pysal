******************************************
Spatial Weights
******************************************

Introduction
============

PySAL supports the creation, manipulation and analysis of spatial weights
matrices of three different types:

 * Contiguity Based Weights
 * Distance Based Weights
 * General Weights

 

Contiguity Based Weights
========================

To illustrate the general weights object, we start with a simple contiguity
matrix constructed for a 5 by 5 lattice::

    serge@jessica:~/Research/p/PySAL/src/google/pysal/pysal$ ipython
    Python 2.5.2 (r252:60911, Jul 22 2009, 15:35:03) 
    Type "copyright", "credits" or "license" for more information.

    IPython 0.9.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object'. ?object also works, ?? prints more.

    In [1]: import pysal

    In [2]: w=pysal.weights.lat2gal(5,5)

    In [3]: w.n
    Out[3]: 25

    In [4]: w.pct_nonzero
    Out[4]: 0.128

    In [5]: w.weights[0]
    Out[5]: [1, 1]

    In [6]: w.neighbors[0]
    Out[6]: [5, 1]


Distance Based Weights
======================

General distance weights
------------------------

Weights based on distance between spatial objects can be constructed. Here we
create a simple 5 by 5 grid of observations and generate inverse distance
weights::

    In [7]: import numpy

    In [8]: x,y=numpy.indices((5,5))

    In [9]: x.shape=(25,1)

    In [10]: y.shape=(25,1)

    In [11]: data=numpy.hstack([x,y])

    In [12]: wid=pysal.weights.InverseDistance(data)

    In [13]: wid_ns=pysal.weights.InverseDistance(data,row_standardize=False)

    In [14]: wid.weights[0][0:3]
    Out[14]: [0.0, 0.21689522769159933, 0.054223806922899832]

    In [15]: wid_ns.weights[0][0:3]
    Out[15]: [0.0, 1.0, 0.25]

    In [16]: wid.pct_nonzero
    Out[16]: 0.95999999999999996

    In [17]: 

Nearest neighbor weights
------------------------

Simple contiguity matrices based on nearest neighbor relations can also be
defined. In the following example we create three different weights objects
for k-nearest neighbors with k from 2 to 4::

        >>> c=5
        >>> nf=num.indices((c,c)).flatten()
        >>> data=num.array(zip(nf[0:c*c],nf[c*c::]))
        >>> wid=pysal.weights.InverseDistance(data)
        >>> wid_ns=pysal.weights.InverseDistance(data,row_standardize=False)
        >>> wid.weights[0][0:3]
        [0.0, 0.21689522769159933, 0.054223806922899832]
        >>> wid_ns.weights[0][0:3]
        [0.0, 1.0, 0.25]
        >>> from pysal.weights import NearestNeighbors as nn
        >>> nnw=[ nn(data,k) for k in range(2,5)]
        >>> for k,nnwk in enumerate(nnw):
              print k+2,nnwk.neighbors[0]
        2 [1, 5]
        3 [1, 5, 6]
        4 [1, 5, 6, 2]
