
# external imports

try:
    import numpy as np
    import numpy.linalg as la
except:
    print 'numpy 0.3+ is required'

try:
    import scipy as sp
    import scipy.stats as stats
    from scipy.spatial import distance_matrix
    from scipy.spatial import KDTree
except:
    print 'scipy 0.7+ is required'


import copy
import math
import random
import sys
import time
import unittest

