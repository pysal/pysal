
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


import copy
import math
import random
import sys
import time
import unittest
