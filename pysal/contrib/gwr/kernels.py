# GWR kernel function specifications

from pysal.weights.Distance import Kernel, knnW

#adaptive specifications should be parameterized with nn+1 to match original gwr
#implementation. That is, pysal counts self neighbors with knn.

def fix_gauss(points, bw):
    w = Kernel(points, function='gwr_gauss', bandwidth=bw, diagonal=True,
            truncate=False)
    return w.full()[0]

def adapt_gauss(points, nn):
    w = Kernel(points, fixed=False, k=nn, function='gwr_gauss', diagonal=True,
            truncate=False)
    return w.full()[0]

def fix_bisqaure(points, bw):
    w = Kernel(points, function='bisquare', bandwidth=bw, diagonal=True)
    return w.full()[0]

def adapt_bisquare(points, nn):
    w = Kernel(points, fixed=false, k=nn, function='bisquare', diagonal=True)
    return w.full()[0]

def fix_exp(points, bw):
    w = Kernel(points, function='exponential', bandwidth=bw, diagonal=True,
            truncate=False)
    return w.full()[0]

def adapt_exp(points, nn):
    w = Kernel(points, fixed=False, k=nn, function='exponential', diagonal=True,
            truncate=False)
    return w.full()[0]
