"""
Classes for statistics for testing hypotheses of spatial autocorrelation amongst
vectors. 
"""

_author_ = "Taylor Oshan tayoshan@gmail.com, Levi Wolf levi.john.wolf@gmail.com"

import numpy as np
import scipy.stats as stats
from pysal.weights.Distance import DistanceBand

PERMUTATIONS = 99

class VecMoran:
    """Moran's I Global Autocorrelation Statistic For Vectors
    
    Parameters
    ----------
    y               : array
                      variable measured across n origin-destination vectors
    w               : W
                      spatial weights instance
    focus           : string
                      denotes whether to calculate the statistic with a focus on
                      spatial proximity between origins or destinations; default
                      is 'origin' but option include:

                      'origin' | 'destination'
    
    rand            : string
                      denote which randomization technqiue to use for
                      significance testing; default is 'A' but options are:

                      'A': transate entire vector 
                      'B': shuffle points and redraw vectors

    permutations    : int
                      number of random permutations for calculation of
                      pseudo-p_values
    two_tailed      : boolean
                      If True (default) analytical p-values for Moran are two
                      tailed, otherwise if False, they are one-tailed.
    Attributes
    ----------
    y               : array
                      original variable
    w               : W obejct
                      original w object
    n               : integer
                      number of vectors
    o               : array
                      n x 2; 2D coordinates of vector origins
    d               : array
                      n x 2: 2D coordinates of vector destinations
    alpha           : scalar
                      distance decay parameter harvested from W object
    binary          : boolean
                      True is all entries in W > 0 are set to 1; False if if they
                      are inverse distance weighted; default is False; attribute is
                      harvested from W object
    build_sp        : boolean 
                      True if W object is build using sparse distance matrix and
                      False if it is built using a dense distance matrix; attribute
                      is harvested from W object
    threshold       : scalar
                      all values larger than threshold are set 0 in W object;
                      attribute is harvested from W object
    silent          : boolean
                      True if island warnings are silent and False if they are not;
                      default is False; attribute is harvested from W object
    focus           : string
                      denotes whether to calculate the statistic with a focus on
                      spatial proximity between origins or destinations; default
                      is 'origin' but option include:

                      'origin' | 'destination'
    
    rand            : string
                      denote which randomization technqiue to use for
                      significance testing; default is 'A' but options are:

                      'A': transate entire vector 
                      'B': shuffle points and redraw vectors

    permutations    : int
                      number of permutations
    I               : float
                      value of vector-based Moran's I
    EI              : float
                      expected value under randomization assumption
    VI_rand         : float
                      variance of I under randomization assumption
    seI_rand        : float
                      standard deviation of I under randomization assumption
    z_rand          : float
                      z-value of I under randomization assumption
    p_rand          : float
                      p-value of I under randomization assumption
    two_tailed      : boolean
                      If True p_norm and p_rand are two-tailed, otherwise they
                      are one-tailed.
    sim             : array
                      (if permutations>0)
                      vector of I values for permuted samples
    p_sim           : array
                      (if permutations>0)
                      p-value based on permutations (one-tailed)
                      null: spatial randomness
                      alternative: the observed I is extreme if
                      it is either extremely greater or extremely lower
                      than the values obtained based on permutations
    EI_sim          : float
                      (if permutations>0)
                      average value of I from permutations
    VI_sim          : float
                      (if permutations>0)
                      variance of I from permutations
    seI_sim         : float
                      (if permutations>0)
                      standard deviation of I under permutations.
    z_sim           : float
                      (if permutations>0)
                      standardized I based on permutations
    p_z_sim         : float
                      (if permutations>0)
                      p-value based on standard normal approximation from
                      permutations

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> from pysal.weight import DistanceBand
    >>> from pysal.contrib.spint.vec_SA import VecMoran
    >>> vecs = np.array([[1, 55, 60, 100, 500], 
    >>>                 [2, 60, 55, 105, 501], 
    >>>                 [3, 500, 55, 155, 500], 
    >>>                 [4, 505, 60, 160, 500], 
    >>>                 [5, 105, 950, 105, 500], 
    >>>                 [6, 155, 950, 155, 499]])
    >>> origins = vecs[:, 1:3]
    >>> dests = vecs[:, 3:5]
    >>> wo = DistanceBand(origins, threshold=9999, alpha=-1.5, binary=False)
    >>> wd = DistanceBand(dests, threshold=9999, alpha=-1.5, binary=False)
    
    #randomization technique A
    >>> vmo = VecMoran(vecs, wo, focus='origin', rand='A')
    >>> vmd = VecMoran(vecs, wd, focus='destination', rand='A')
    >>> vmo.I
    -0.764603695022
    >>> vmo.p_z_sim
    0.99549579548
    >>>  vmd.I
    0.645944594367
    >>>  vmd.p_z_sim
    0.1494726733677

    #randomization technique B
    >>> vmo = VecMoran(vecs, wo, focus='origin', rand='B')
    >>> vmd = VecMoran(vecs, wd, foucs='destination', rand='B')
    >>> vmo.I
    -0.764603695022
    >>> vmo.p_z_sim
    0.071427063787951814
    >>>  vmd.I
    0.645944594367
    >>>  vmd.p_z_sim
    0.086894261015806051
    
    """

    def __init__(self, y, w, focus='origin', rand='A', permutations=PERMUTATIONS,
            two_tailed=True):
        self.y = y
        self.o = y[:, 1:3]
        self.d = y[:, 3:5]
        self.focus = focus
        self.rand = rand
        self.permutations = permutations
        self.two_tailed = two_tailed
        if isinstance(w, DistanceBand): 
            self.w = w
        else:
            raise TypeError('Spatial weight, W, must be of type DistanceBand')
        try:
            self.threshold = w.threshold
            self.alpha = w.alpha
            self.build_sp = w.build_sp
            self.binary = w.binary
            self.silent = w.silent
        except:
            raise AttributeError('W object missing necessary attributes: '
                'threshold, alpha, binary, build_sp, silent')

        self.__moments()
        self.I = self.__calc(self.z)
        self.z_rand = (self.I - self.EI) / self.seI_rand
  
        if self.z_rand > 0:
            self.p_rand = 1 - stats.norm.cdf(self.z_rand)
        else:
            self.p_rand = stats.norm.cdf(self.z_rand)

        if self.two_tailed:
            self.p_rand *= 2.

        if permutations:
            if self.rand.upper() == 'A':
            	sim = self.__rand_vecs_A(self.focus)
            elif self.rand.upper() == 'B':
                sim = self.__rand_vecs_B(self.focus)
            else:
                raise ValueError("Parameter 'rand' must take a value of either 'A' or 'B'")

            self.sim = sim = np.array(sim)
            above = sim >= self.I
            larger = above.sum()
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.p_sim = (larger + 1.) / (permutations + 1.)
            self.EI_sim = sim.sum() / permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim ** 2
            self.z_sim = (self.I - self.EI_sim) / self.seI_sim
            if self.z_sim > 0:
                self.p_z_sim = 1 - stats.norm.cdf(self.z_sim)
            else:
                self.p_z_sim = stats.norm.cdf(self.z_sim)

    def __moments(self):
        self.n = len(self.y)
        xObar = self.o[:,0].mean()
        yObar = self.o[:,1].mean()
        xDbar = self.d[:,0].mean()
        yDbar = self.d[:,1].mean()
        u = (self.y[:,3] - self.y[:,1]) - (xDbar - xObar)
        v = (self.y[:,4] - self.y[:,2]) - (yDbar - yObar)
        z = np.outer(u, u) + np.outer(v,v)
        self.z = z
        self.uv2ss = np.sum(np.dot(u,u) + np.dot(v,v))
        self.EI = -1. / (self.n - 1)
        n = self.n
        s1 = self.w.s1
        W = self.w.s0
        s2 = self.w.s2

        v_num = n * n * s1 - n * s2 + 3 * W * W
        v_den = (n - 1) * (n + 1) * W * W

        a2 = np.sum(np.dot(u, u))/n
        b2 = np.sum(np.dot(v, v))/n
        m2 = a2 + b2
        a4 = np.sum(np.dot(np.dot(u, u), np.dot(u, u)))/n
        b4 = np.sum(np.dot(np.dot(v, u), np.dot(v, v)))/n
        n1 = a2**2*((n**2 - 3*n + 3)*s1-n*s2 + 3*W**2)
        n2 = a4*((n**2 - n)*s1 - 2*n*s2 + 6*W**2)
        n3 = b2**2*((n**2 - 3*n + 3)*s1 - n*s2 + 3*W**2)
        n4 = b4*((n**2 - n)*s1 - 2*n*s2 + 6*W**2)
        d = (n - 1)*(n - 2)*(n - 3)
        self.VI_rand = 1/(W**2*m2**2) * \
                  ((n1 - n2)/d + (n3 - n4)/d) + \
                  ((a2*b2) - m2**2)/(m2**2*(n - 1)**2)
        self.seI_rand = self.VI_rand ** (1 / 2.)

    def __calc(self, z):
        zl = self._slag(self.w, z)
        inum = np.sum(zl)
        return self.n / self.w.s0 * inum / self.uv2ss
    
    def _newD(self, oldO, oldD, newO):
        oldOX, oldOY = oldO[:,0], oldO[:,1]
        oldDX, oldDY = oldD[:,0], oldD[:,1]
        newOX, newOY = newO[:,0], newO[:,1]
        deltaX = newOX - oldOX
        deltaY = newOY - oldOY
        newDX = oldDX + deltaX
        newDY = oldDY + deltaY
        return np.hstack([newDX.reshape((-1,1)), newDY.reshape((-1,1))])

    def _newO(self, oldO, oldD, newD):
        oldOX, oldOY = oldO[:,0], oldO[:,1]
        oldDX, oldDY = oldD[:,0], oldD[:,1]
        newDX, newDY = newD[:,0], newD[:,1]
        deltaX = newDX - oldDX
        deltaY = newDY - oldDY
        newOX = oldOX + deltaX
        newOY = oldOY + deltaY
        return np.hstack([newOX.reshape((-1,1)), newOY.reshape((-1,1))])
    
    def __rand_vecs_A(self, focus):
        if focus.lower() == 'origin':
            newOs = [np.random.permutation(self.o) for i in xrange(self.permutations)]
            sims = [np.hstack([np.arange(self.n).reshape((-1,1)), newO,
            self._newD(self.o, self.d, newO)]) for newO in newOs]
            Ws = [DistanceBand(newO, threshold=self.threshold, alpha=self.alpha, 
                binary=self.binary, build_sp=self.build_sp, silent=self.silent)
                for newO in newOs]
        elif focus.lower() == 'destination':
            newDs = [np.random.permutation(self.d) for i in xrange(self.permutations)]
            sims = [np.hstack([np.arange(self.n).reshape((-1,1)),
                self._newO(self.o, self.d, newD), newD]) for newD in newDs]
            Ws = [DistanceBand(newD, threshold=self.threshold, alpha=self.alpha, 
                binary=self.binary, build_sp=self.build_sp, silent=self.silent)
                for newD in newDs]
        else:
            raise ValueError("Parameter 'focus' must take value of either 'origin' or 'destination.'")

        VMs = [VecMoran(y, Ws[i], permutations=None) for i, y in enumerate(sims)]
        sim = [VM.__calc(VM.z) for VM in VMs]
        return sim

    def __rand_vecs_B(self, focus):
        if focus.lower() == 'origin':
            sims = [np.hstack([np.arange(self.n).reshape((-1,1)), self.o,
                np.random.permutation(self.d)]) for i in xrange(self.permutations)]
        elif focus.lower() == 'destination':
            sims = [np.hstack([np.arange(self.n).reshape((-1,1)),
                np.random.permutation(self.o), self.d]) for i in xrange(self.permutations)]
        else:
            raise ValueError("Parameter 'focus' must take value of either 'origin' or 'destination.'")
        sims = [VecMoran(y, self.w, permutations=None) for y in sims]
        sim = [VM.__calc(VM.z) for VM in sims]
        return sim
       
    def _slag(self, w, y):
        """
        Dense spatial lag operator for.
        If w is row standardized, returns the average of each observation's neighbors;
        if not, returns the weighted sum of each observation's neighbors.
        Parameters
        ----------
        w                   : W
                              object
        y                   : array
                              numpy array with dimensionality conforming to w (see examples)
        Returns
        -------
        wy                  : array
                              array of numeric values for the spatial lag
        """
        return np.array(w.sparse.todense()) * y

