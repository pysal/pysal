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

def set_Bandwidth(data, method=1, band=0.0, criterion=0.0, maxVal=0.0, minVal=0.0, interval=0.0):
    """
    Set bandwidth using specified method and parameters
        
    Arguments:
        data: dictionary, including (x,y) coordinates involved in the weight evaluation (including point i) 
        band: float, bandwidth for single bandwidth
        criterion: integer, criteria for bandwidth selection
                   0: AICc
                   1: AIC
                   2: BIC/MDL
                   3: CV
        method: integer, method to use
                0: Gloden section search
                1: Single bandwith 
                2: Interval Search
        maxVal: max value for method 0 or 2
        minVal: min value for method 0 or 2
        interval: float, for Interval Search        
            
    Return:
        band: float, bandwidth       
    """
    if method == 1:
        return band
    if method == 0:
        return band_Golden(data, criterion, maxVal, minVal)
    if method == 2:
        return band_Interval(data, criterion, maxVal, minVal, interval)
    
    
def band_Golden(data, criterion, maxVal, minVal):
    """
    Set bandwidth using golden section search
    
    Methods: p212-213, section 9.6.4, Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        data: dictionary, including (x,y) coordinates involved in the weight evaluation (including point i) 
        criterion: integer, criteria for bandwidth selection
                   0: AICc
                   1: AIC
                   2: BIC/MDL
                   3: CV
        maxVal: max value for method 0 or 2
        minVal: min value for method 0 or 2       
            
    Return:
        band: float, bandwidth
    """
    a, b, c
    b = (1-0.618) * abs(c-a)
    
    get_criteria[criterion]()

def band_Interval(data, criterion, maxVal, minVal, interval):
    """
    Set bandwidth using interval search
    
    Methods: p61, (2.34), Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        data: dictionary, including (x,y) coordinates involved in the weight evaluation (including point i) 
        criterion: integer, criteria for bandwidth selection
                   0: AICc
                   1: AIC
                   2: BIC/MDL
                   3: CV
        maxVal: max value for method 0 or 2
        minVal: min value for method 0 or 2
        interval: float, for Interval Search        
            
    Return:
        band: float, bandwidth
    """  
    get_criteria[criterion]()
