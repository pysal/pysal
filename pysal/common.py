
# external imports

try:
    import numpy as np
    import numpy.linalg as la
except:
    print 'numpy 1.3 is required'
    raise

try:
    import scipy as sp
    import scipy.stats as stats
    from scipy.spatial import KDTree
    from scipy.spatial.distance import pdist,cdist
except:
    print 'scipy 0.7+ is required'
    raise

class ROD(dict):
    """
    Read Only Dictionary
    """
    def __setitem__(self,*args):
        raise TypeError, "'Read Only Dictionary (ROD)' object does not support item assignment"



import copy
import math
import random
import sys
import time
import unittest


