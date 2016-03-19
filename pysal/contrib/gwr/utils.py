import numpy as np
import numpy.linalg as la
from scipy import sparse as SP
from scipy.sparse import linalg as SPla


class RegressionPropsY(object):

    """
    Helper class that adds common regression properties to any regression
    class that inherits it.  It takes no parameters.  See BaseOLS for example
    usage.

    Parameters
    ----------

    Attributes
    ----------
    mean_y  : float
              Mean of the dependent variable
    std_y   : float
              Standard deviation of the dependent variable

    """

    @property
    def mean_y(self):
        try:
            return self._cache['mean_y']
        except AttributeError:
            self._cache = {}
            self._cache['mean_y'] = np.mean(self.y)
        except KeyError:
            self._cache['mean_y'] = np.mean(self.y)
        return self._cache['mean_y']
    
    @mean_y.setter
    def mean_y(self, val):
        try:
            self._cache['mean_y'] = val
        except AttributeError:
            self._cache = {}
            self._cache['mean_y'] = val
        except KeyError:
            self._cache['mean_y'] = val

    @property
    def std_y(self):
        try:
            return self._cache['std_y']
        except AttributeError:
            self._cache = {}
            self._cache['std_y'] = np.std(self.y, ddof=1)
        except KeyError:
            self._cache['std_y'] = np.std(self.y, ddof=1)
        return self._cache['std_y']
    
    @std_y.setter
    def std_y(self, val):
        try:
            self._cache['std_y'] = val
        except AttributeError:
            self._cache = {}
            self._cache['std_y'] = val
        except KeyError:
            self._cache['std_y'] = val


class RegressionPropsVM(object):

    """
    Helper class that adds common regression properties to any regression
    class that inherits it.  It takes no parameters.  See BaseOLS for example
    usage.

    Parameters
    ----------

    Attributes
    ----------
    utu     : float
              Sum of the squared residuals
    sig2n    : float
              Sigma squared with n in the denominator
    sig2n_k : float
              Sigma squared with n-k in the denominator
    vm      : array
              Variance-covariance matrix (kxk)

    """

    @property
    def utu(self):
        try:
            return self._cache['utu']
        except AttributeError:
            self._cache = {}
            self._cache['utu'] = np.sum(self.u ** 2)
        except KeyError:
            self._cache['utu'] = np.sum(self.u ** 2)
        return self._cache['utu']

    @utu.setter
    def utu(self, val):
        try:
            self._cache['utu'] = val
        except AttributeError:
            self._cache = {}
            self._cache['utu'] = val
        except KeyError:
            self._cache['utu'] = val

    @property
    def sig2n(self):
        try:
            return self._cache['sig2n']
        except AttributeError:
            self._cache = {}
            self._cache['sig2n'] = self.utu / self.n
        except KeyError:
            self._cache['sig2n'] = self.utu / self.n
        return self._cache['sig2n']

    @sig2n.setter
    def sig2n(self, val):
        try:
            self._cache['sig2n'] = val
        except AttributeError:
            self._cache = {}
            self._cache['sig2n'] = val
        except KeyError:
            self._cache['sig2n'] = val

    @property
    def sig2n_k(self):
        try:
            return self._cache['sig2n_k']
        except AttributeError:
            self._cache = {}
            self._cache['sig2n_k'] = self.utu / (self.n - self.k)
        except KeyError:
            self._cache['sig2n_k'] = self.utu / (self.n - self.k)
        return self._cache['sig2n_k']
    
    @sig2n_k.setter
    def sig2n_k(self, val):
        try:
            self._cache['sig2n_k'] = val
        except AttributeError:
            self._cache = {}
            self._cache['sig2n_k'] = val
        except KeyError:
            self._cache['sig2n_k'] = val

    @property
    def vm(self):
        try:
            return self._cache['vm']
        except AttributeError:
            self._cache = {}
            self._cache['vm'] = np.dot(self.sig2, self.xtxi)
        except KeyError:
            self._cache['vm'] = np.dot(self.sig2, self.xtxi)
        return self._cache['vm']

    @vm.setter
    def vm(self, val):
        try:
            self._cache['vm'] = val
        except AttributeError:
            self._cache = {}
            self._cache['vm'] = val
        except KeyError:
            self._cache['vm'] = val


def sphstack(a, b, array_out=False):
    """
    Horizontal stacking of vectors (or matrices) to deal with sparse and dense objects

    Parameters
    ----------

    a           : array or sparse matrix
                  First object.
    b           : array or sparse matrix
                  Object to be stacked next to a
    array_out   : boolean
                  If True the output object is a np.array; if False (default)
                  the output object is an np.array if both inputs are
                  arrays or CSR matrix if at least one input is a CSR matrix

    Returns
    -------

    ab          : array or sparse matrix
                  Horizontally stacked objects
    """
    if type(a).__name__ == 'ndarray' and type(b).__name__ == 'ndarray':
        ab = np.hstack((a, b))
    elif type(a).__name__ == 'csr_matrix' or type(b).__name__ == 'csr_matrix':
        ab = SP.hstack((a, b), format='csr')
        if array_out:
            if type(ab).__name__ == 'csr_matrix':
                ab = ab.toarray()
    else:
        raise Exception, "Invalid format for 'sphstack' argument: %s and %s" % (
            type(a).__name__, type(b).__name__)
    return ab
