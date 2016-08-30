# GWR kernel function specifications

__author__ = "Taylor Oshan tayoshan@gmail.com" 

#from pysal.weights.Distance import Kernel 
import scipy
from scipy.spatial.kdtree import KDTree
import numpy as np

#adaptive specifications should be parameterized with nn-1 to match original gwr
#implementation. That is, pysal counts self neighbors with knn automatically.

def fix_gauss(points, bw):
    w = _Kernel(points, function='gwr_gaussian', bandwidth=bw,
            truncate=False)
    return w.kernel

def adapt_gauss(points, nn):
    w = _Kernel(points, fixed=False, k=nn-1, function='gwr_gaussian',
            truncate=False)
    return w.kernel

def fix_bisquare(points, bw):
    w = _Kernel(points, function='bisquare', bandwidth=bw)
    return w.kernel

def adapt_bisquare(points, nn):
    w = _Kernel(points, fixed=False, k=nn-1, function='bisquare')
    return w.kernel

def fix_exp(points, bw):
    w = _Kernel(points, function='exponential', bandwidth=bw,
            truncate=False)
    return w.kernel

def adapt_exp(points, nn):
    w = _Kernel(points, fixed=False, k=nn-1, function='exponential',
            truncate=False)
    return w.kernel

from scipy.spatial.distance import cdist

class _Kernel(object):
    """

    """
    def __init__(self, data, bandwidth=None, fixed=True, k=None,
                 function='triangular', eps=1.0000001, ids=None, truncate=True): #Added truncate flag
        if issubclass(type(data), scipy.spatial.KDTree):
            self.kdt = data
            self.data = self.kdt.data
            data = self.data
        else:
            self.data = data
            self.kdt = KDTree(self.data)
        if k is not None:
            self.k = int(k) + 1
        else:
            self.k = k
        self.dmat = cdist(self.data, self.data)
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
            mask = np.repeat(self.bandwidth, len(self.bandwidth), axis=1)
            kernel_mask = self._kernel_funcs(1.0/mask)
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
