# GWR kernel function specifications
import sys
sys.path.append('/Users/toshan/projects/GIS596/pysal/pysal/weights/')
from Distance import Kernel
#from pysal.weights.Distance import Kernel 


#adaptive specifications should be parameterized with nn-1 to match original gwr
#implementation. That is, pysal counts self neighbors with knn automatically.

def fix_gauss(points, bw):
    w = Kernel(points, function='gwr_gaussian', bandwidth=bw, diagonal=True,
            truncate=False)
    return w.full()[0]

def adapt_gauss(points, nn):
    w = Kernel(points, fixed=False, k=nn-1, function='gwr_gaussian', diagonal=True,
            truncate=False)
    return w.full()[0]

def fix_bisquare(points, bw):
    w = Kernel(points, function='bisquare', bandwidth=bw, diagonal=True)
    return w.full()[0]

def adapt_bisquare(points, nn):
    w = Kernel(points, fixed=False, k=nn-1, function='bisquare', diagonal=True)
    return w.full()[0]

def fix_exp(points, bw):
    w = Kernel(points, function='exponential', bandwidth=bw, diagonal=True,
            truncate=False)
    return w.full()[0]

def adapt_exp(points, nn):
    w = Kernel(points, fixed=False, k=nn-1, function='exponential', diagonal=True,
            truncate=False)
    return w.full()[0]
