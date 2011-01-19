"""
:mod:`weights` --- Spatial Weights
==================================

"""
from weights import *
from util import *
from Distance import *
from Contiguity import *
from user import *
from spatial_lag import *
from Wsets import *

if __name__ == "__main__":

    import doctest
    doctest.testmod(verbose=True)
