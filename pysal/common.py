
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


class ROD(dict):
    """
    Read Only Dictionary
    """
    def __setitem__(self, *args):
        raise TypeError, "'Read Only Dictionary (ROD)' object does not support item assignment"

    def __copy__(self):
        """
        Example:
        >>> import copy
        >>> d = ROD({'key':[1,2,3]})
        >>> d2 = copy.copy(d)
        >>> d2 == d
        True
        >>> d2 is d
        False
        >>> d2['key'] is d['key']
        True
        """
        return ROD(self.copy())
    def __deepcopy__(self,memo):
        """
        Example:
        >>> import copy
        >>> d = ROD({'key':[1,2,3]})
        >>> d2 = copy.deepcopy(d)
        >>> d2 == d
        True
        >>> d2 is d
        False
        >>> d2['key'] is d['key']
        False
        """
        import copy
        return ROD(copy.deepcopy(self.copy()))

import copy
import math
import random
import sys
import time
import unittest
