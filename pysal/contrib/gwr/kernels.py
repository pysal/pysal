# GWR kernel function specifications

__author__ = "Taylor Oshan tayoshan@gmail.com"

#from pysal.weights.Distance import Kernel
import scipy
from scipy.spatial.kdtree import KDTree
import numpy as np

#adaptive specifications should be parameterized with nn-1 to match original gwr
#implementation. That is, pysal counts self neighbors with knn automatically.

def fix_gauss(coords, bw, points=None):
    w = _Kernel(coords, function='gwr_gaussian', bandwidth=bw,
            truncate=False, points=points)
    return w.kernel

def adapt_gauss(coords, nn, points=None):
    w = _Kernel(coords, fixed=False, k=nn-1, function='gwr_gaussian',
            truncate=False, points=points)
    return w.kernel

def fix_bisquare(coords, bw, points=None):
    w = _Kernel(coords, function='bisquare', bandwidth=bw, points=points)
    return w.kernel

def adapt_bisquare(coords, nn, points=None):
    w = _Kernel(coords, fixed=False, k=nn-1, function='bisquare', points=points)
    return w.kernel

def fix_exp(coords, bw, points=None):
    w = _Kernel(coords, function='exponential', bandwidth=bw,
            truncate=False, points=points)
    return w.kernel

def adapt_exp(coords, nn, points=None):
    w = _Kernel(coords, fixed=False, k=nn-1, function='exponential',
            truncate=False, points=points)
    return w.kernel

from scipy.spatial.distance import cdist

#Customized Kernel class user for GWR because the default PySAL kernel class
#favors memory optimization over speed optimizations and GWR often needs the 
#speed optimization since it is not always assume points far awar are truncated
#to zero
class _Kernel(object):
    """

    """
    def __init__(self, data, bandwidth=None, fixed=True, k=None,
                 function='triangular', eps=1.0000001, ids=None, truncate=True,
                 points=None): #Added truncate flag
        if issubclass(type(data), scipy.spatial.KDTree):
            self.data = data.data
            data = self.data
        else:
            self.data = data
        if k is not None:
            self.k = int(k) + 1
        else:
            self.k = k
        if points is None:
            self.dmat = cdist(self.data, self.data)
        else:
            self.points = points
            self.dmat = cdist(self.points, self.data)
        self.function = function.lower()
        self.fixed = fixed
        self.eps = eps
        self.trunc = truncate
        if bandwidth:
            try:
                bandwidth = np.array(bandwidth)
                bandwidth.shape = (len(bandwidth), 1)
            except:
                bandwidth = np.ones((len(data), 1), 'float') * bandwidth
            self.bandwidth = bandwidth
        else:
            self._set_bw()
        self.kernel = self._kernel_funcs(self.dmat/self.bandwidth)

        if self.trunc:
            mask = np.repeat(self.bandwidth, len(self.data), axis=1)
            self.kernel[(self.dmat >= mask)] = 0

    def _set_bw(self):
        if self.k is not None:
            dmat = np.sort(self.dmat)[:,:self.k]
        else:
            dmat = self.dmat
        if self.fixed:
            # use max knn distance as bandwidth
            bandwidth = dmat.max() * self.eps
            n = len(self.data)
            self.bandwidth = np.ones((n, 1), 'float') * bandwidth
        else:
            # use local max knn distance
            self.bandwidth = dmat.max(axis=1) * self.eps
            self.bandwidth.shape = (self.bandwidth.size, 1)


    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zi.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs ** 2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs ** 2) ** 2
        elif self.function == 'gaussian':
            c = np.pi * 2
            c = c ** (-0.5)
            return c * np.exp(-(zs ** 2) / 2.)
        elif self.function == 'gwr_gaussian':
            return np.exp(-0.5*(zs)**2)
        elif self.function == 'bisquare':
            return (1-(zs)**2)**2
        elif self.function =='exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)
