"""
cython -a fast_geary.pyx
gcc -shared -fPIC -O2 -Wall -fno-strict-aliasing -o fast_geary.so fast_geary.c -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/
"""


cimport cython
from numpy import zeros

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_geary_c(double[:] y, id_order, neighbor_offsets, weights, int n,
                    double den):
    cdef int i
    cdef int j
    cdef int ii
    cdef int neighbor_j
    cdef double wij
    cdef int n_ids = len(id_order)
    cdef int nobs = y.shape[0]
    cdef double ys = 0.0
    cdef double[:] y2 = zeros(nobs)
    for ii in range(nobs):
        y2[ii] = y[ii]**2

    for i in range(n_ids):
        i0 = id_order[i]
        neighbors = neighbor_offsets[i0]
        wijs = weights[i0]
        for j in range(len(wijs)):
            wij = wijs[j]
            neighbor_j = neighbors[j]
            ys = ys + wij * (y2[i] - 2 * y[i] * y[neighbor_j] + y2[neighbor_j])
    a = (n - 1) * ys
    return a / den
