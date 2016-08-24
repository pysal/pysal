# GWR Bandwidth selection class

#Thinking about removing the search method and just having optimization begin in
#class __init__

#x_glob and offset parameters dont yet do anything; former is for semiparametric
#GWR and later is for offset variable for Poisson model

__author__ = "Taylor Oshan Tayoshan@gmail.com"

from kernels import *
from search import golden_section, equal_interval, flexible_bw
from gwr import GWR
from pysal.contrib.glm.family import Gaussian, Poisson, Binomial
from diagnostics import get_AICc, get_AIC, get_BIC, get_CV
from scipy.spatial.distance import pdist, squareform
from pysal.common import KDTree
import numpy as np

kernels = {1: fix_gauss, 2: adapt_gauss, 3: fix_bisquare, 4:
        adapt_bisquare, 5: fix_exp, 6:adapt_exp}
getDiag = {'AICc': get_AICc,'AIC':get_AIC, 'BIC': get_BIC, 'CV': get_CV}

class Sel_BW(object):
    """
    Select bandwidth for kernel

    Methods: p211 - p213, bandwidth selection
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Parameters
    ----------
    y              : array
                     n*1, dependent variable.
    x_glob         : array
                     n*k1, fixed independent variable.
    x_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    offset         : array
                     n*1, offset variable for Poisson model
    kernel         : string
                     kernel function: 'gaussian', 'bisquare', 'exponetial'
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)

    Attributes
    ----------
    y              : array
                     n*1, dependent variable.
    x_glob         : array
                     n*k1, fixed independent variable.
    x_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family        : string
                    GWR model type: 'Gaussian', 'logistic, 'Poisson''
    kernel        : string
                    type of kernel used and wether fixed or adaptive
    criterion     : string
                    bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
    search        : string
                    bw search method: 'golden', 'interval'
    bw_min        : float
                    min value used in bandwidth search
    bw_max        : float
                    max value used in bandwidth search
    interval      : float
                    interval increment used in interval search
    tol           : float
                    tolerance used to determine convergence
    max_iter      : integer
                    max interations if no convergence to tol
    """
    def __init__(self, coords, y, x_loc, x_glob=None, family=Gaussian(),
            offset=None, kernel='gaussian', fixed=False, fb=False):
        self.coords = coords
        self.y = y
        self.x_loc = x_loc
        if x_glob is not None:
            self.x_glob = x_glob
        else:
            self.x_glob = []
        self.family=family
        self.fixed = fixed
        self.kernel = kernel
        self.fb = fb

    def search(self, search='golden_section', criterion='AICc', bw_min=0.0, 
            bw_max=0.0, interval=0.0, tol=1.0e-6, max_iter=200, init_fb=True,
            tol_fb=1.0e-5, rss_score=False, max_iter_fb=200):
        """
        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        search         : string
                         bw search method: 'golden', 'interval'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        interval       : float
                         interval increment used in interval search
        tol            : float
                         tolerance used to determine convergence
        max_iter       : integer
                         max iterations if no convergence to tol
        
        Returns
        -------
        bw             : scalar
                         optimal bandwidth value
        """     
        self.search = search
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter
        self.init_fb = init_fb
        self.tol_fb = tol_fb
        self.rss_score = rss_score
        self.max_iter_fb = max_iter_fb
        

        if self.fixed:
            if self.kernel == 'gaussian':
                ktype = 1
            elif self.kernel == 'bisquare':
                ktype = 3
            elif self.kernel == 'exponential':
                ktype = 5
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)
        else:
            if self.kernel == 'gaussian':
            	ktype = 2
            elif self.kernel == 'bisquare':
                ktype = 4
            elif self.kernel == 'exponential':
                ktype = 6
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)

        if ktype % 2 == 0:
            int_score = True
        else:
            int_score = False
        self.int_score = int_score
        
        if self.fb:
            self._fbw()
        else:
            self_bw()

        return self.bw[0]

    def _init_section(self, x_glob, x_loc, coords):
        if len(x_glob) > 0:
            n_glob = x_glob.shape[1]
        else:
            n_glob = 0
        if len(x_loc) > 0:
            n_loc = x_loc.shape[1]
        else:
            n_loc = 0
        n_vars = n_glob + n_loc + 1 #intercept
        n = np.array(coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            nn = 40 + 2 * n_vars
            sq_dists = squareform(pdist(coords))
            sort_dists = np.sort(sq_dists, axis=1)
            min_dists = sort_dists[:,nn-1]
            max_dists = sort_dists[:,-1]
            a = np.min(min_dists)/2.0
            c = np.max(max_dists)/2.0
    
    def _bw(self):
        gwr_func = lambda bw: getDiag[criterion](
                GWR(self.coords, self.y, self.x_loc, bw, family=self.family,
                    kernel=self.kernel, fixed=self.fixed).fit())
        if self.search == 'golden_section':
            a,c = self._init_section(self.x_glob, self.x_loc, self.coords)
            delta = 0.38197 #1 - (np.sqrt(5.0)-1.0)/2.0
            self.bw = golden_section(a, c, delta, gwr_func, self.tol, 
                    self.max_iter, self.int_score)
        elif self.search == 'interval':
            self.bw = equal_interval(self.bw_min, self.bw_max, self.interval,
                    gwr_func, self.int_score)
        else:
            raise TypeError('Unsupported computational search method ', search)
   
   def _fbw(self):
        y = self.y
        X = self.x_loc
        family = self.family
        offset = self.offset
        kernel = self.kernel
        fixed = self.fixed
        coords = self.coords
        search = self.search
        criterion = self.criterion
        bw_min = self.bw_min
        bw_max = self.bw_max
        interval = self.interval
        tol = self.tol
        max_iter - self.max_iter
        gwr_func = lambda bw: GWR(coords, y, x_loc, bw, family=family, 
                kernel=kernel, fixed=fixed).fit())
        sel_func = lambda y, X: Sel_BW(coords, y, X, family=family,
                offset=offset, kernel=kernel, fixed=fixed).search(search=search,
                        criterion=criterion, bw_min=bw_min, bw_max=bw_max,
                        interval=interval, tol=tol, max_iter=max_iter)
       self.bw = flexible_bw(self.init_fb, y, X, self.n, self.k, family, self.tol_fb,
               self.max_iter_fb, self.rss_score, gwr_func, sel_fun)
