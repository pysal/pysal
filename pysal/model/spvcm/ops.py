import scipy.sparse as spar
import scipy.sparse.linalg as spla
import theano.tensor as tt
import theano.sparse as ts
import theano as th
from theano.gof import Apply
from theano import Op
import numpy as np

import theano as th

# define this as if it's in terms of rho and W, and give the derivatives in
# terms of rho, since that's what the graph is expecting
class Sparse_LapDet(Op):
    """Sparse Matrix Determinant of a Laplacian Matrix using Sparse LU
    Decomposition"""
    def __init__(self, W):
        self.W = spar.csc_matrix(W)
        self.I = spar.identity(W.shape[0]).tocsc()
        self.Id = self.I.toarray()
        self.Wd = self.W.toarray()

    def make_node(self, rho):
        rho = tt.as_tensor(rho)
        ld = tt.scalar(dtype=rho.dtype)
        return Apply(self, [rho], [ld]) 
    
    def perform(self, node, inputs, outputs):
        (rho,) = inputs
        (z, ) = outputs
        rW = rho * self.W
        A = self.I - rW
        Ud = spla.splu(A).U.diagonal()
        ld = np.asarray(np.sum(np.log(np.abs(Ud))))
        z[0] = ld

    def grad(self, inputs, g_outputs):
        (rho, ) = inputs
        (gz,) = g_outputs
        A = self.Id - tt.mul(rho, self.Wd)
        dinv = tt.nlinalg.matrix_inverse(A).T
        out = tt.mul(dinv, - self.Wd)
        return [tt.as_tensor(tt.sum(tt.mul(out, gz)), ndim=1)]

# define this as if it's in terms of rho and W, and give the derivatives in
# terms of rho, since that's what the graph is expecting
class Sparse_AGrad_LapDet(Op):
    """Sparse Matrix Determinant of a Laplacian Matrix using Sparse LU
    Decomposition"""
    def __init__(self, W):
        self.W = spar.csc_matrix(W)
        self.WW = W.dot(W)
        self.WWW = self.WW.dot(W)
        self.I = spar.identity(W.shape[0]).tocsc()
        self.Id = self.I.toarray()
        self.Wd = self.W.toarray()

    def make_node(self, rho):
        rho = tt.as_tensor(rho)
        ld = tt.scalar(dtype=rho.dtype)
        return Apply(self, [rho], [ld]) 
    
    def perform(self, node, inputs, outputs):
        (rho,) = inputs
        (z, ) = outputs
        rW = rho * self.W
        A = self.I - rW
        Ud = spla.splu(A).U.diagonal()
        ld = np.asarray(np.sum(np.log(np.abs(Ud))))
        z[0] = ld

    def grad(self, inputs, g_outputs):
        (rho, ) = inputs
        (gz,) = g_outputs
        A = self.Id - tt.mul(rho, self.Wd)
        dinv = self.I + ts.mul_s_d(self.W, rho) 
        dinv +=ts.mul_s_d(self.WW, rho**2)
        dinv +=ts.mul_s_d(self.WWW, rho**3)
        out = tt.mul(dinv, - self.Wd)
        return [tt.as_tensor(tt.sum(tt.mul(out, gz)), ndim=1)]

class Dense_LULogDet(Op):
    """Log Determinant of a matrix by sparse LU decomposition,
       from dense inputs. Use when casting has no significant overhead."""
    def make_node(self, A):
        A = tt.as_tensor(A)
        ld = tt.scalar(dtype=A.dtype)
        return Apply(self, [A], [ld]) 
    
    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (z,) = outputs
        As = spar.csc_matrix(A)
        Ud = spla.splu(As).U.diagonal()
        ld = np.sum(np.log(np.abs(Ud)))
        z[0] = ld

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [A] = inputs
        dinv = tt.nlinalg.matrix_inverse(A).T
        dout = tt.dot(gz, dinv)
        return [dout]

class Dense_LogDet(Op):
    """Log Determinant of a matrix using numpy.linalg.slogdet.
       Use as a reference implementation"""
    def make_node(self, A):
        A = tt.as_tensor(A)
        ld = tt.scalar(dtype=A.dtype)
        return Apply(self, [A], [ld]) 
    
    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (z,) = outputs
        sgn, ld = np.linalg.slogdet(A)
        if sgn not in [-1,0,1]:
            raise Exception('Loss of precision in log determinant')
        ld *= sgn
        z[0] = ld

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [A] = inputs
        dinv = tt.nlinalg.matrix_inverse(A).T
        dout = tt.dot(gz, dinv)
        return [dout]
