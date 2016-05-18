
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
    from cg.kdtree import KDTree
    from scipy.spatial.distance import pdist, cdist
except:
    print 'scipy 0.7+ is required'
    raise

RTOL = .00001

import copy
import math
import random
import sys
import time
import unittest

def iteritems(d, **kwargs):
    """
    Implements six compatibility library's iteritems function

    appropriately calls either d.iteritems() or d.items() depending on the
    version of python being used. 
    """
    if sys.version_info.major < 3:
        return d.iteritems(**kwargs)
    else:
        return iter(d.items(**kwargs))
