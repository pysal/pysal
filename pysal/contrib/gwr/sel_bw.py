# GWR Bandwidth selection

# For Model estimation
from kernels import fix_gauss, fix_bisquare, fix_exp, adapt_gauss, adapt_bisquare, adapt_exp
from search import golden_section, interval
from gwr import GWR
from scipy.spatial.distance import cdist
from pysal.common import KDTree
import numpy as np

from diagnostics import get_AICc_GWR#, get_AIC_GWR, get_BIC_GWR, get_CV_GWR, get_AICc_GWGLM, get_AIC_GWGLM, get_BIC_GWGLM

kernels = {1: fix_gauss, 2: adapt_gauss, 3: fix_bisquare, 4:
        adapt_bisquare, 5: fix_exp, 6:adapt_exp}

getDiag = {'AICc': get_AICc_GWR}#,1:get_AIC_GWR, 2:get_BIC_GWR,3: get_CV_GWR} # bandwidth selection criteria

class Sel_BW(object):
    """
    Select bandwidth for kernel
    
    Methods: Fotheringham, Brunsdon and Charlton (2002)
    
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
    link           : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson'' 
    y_off          : array
                     n*1, offset variable for Poisson model
    kernel         : string
                     kernel function: 'gaussian', 'bisquare', 'exponetial'
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    criterion      : string
                     bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
    search         : string
                     bw search method: 'golden', 'interval'
    bw_min         : float
                     min value used in bandwidth search
    bw_max         : float
                     max value used in bandwidth search
    interval       : float
                     interval used in interval search 
    tol            : float
                     tolerance used to determine convergence   
    max_iter       : integer
                     max iterations if no convergence to tol
 
    Attributes                     
    ----------
    link           : string
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

    tol           : float
                    tolerance used to determine convergence
    max_iter      : integer
                    max interations if no convergence to tol
    """
    def __init__(self, coords, y, x_loc, x_glob, family='Gaussian',
            y_off=None, kernel='gaussian', fixed=False, criterion='AICc', search='golden_section', bw_min=0.0, bw_max=0.0, interval=0.0, tol=1.0e-6, max_iter=200):
        self.family=family
        self.fixed = fixed
        self.search = search
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter
        #dists = cdist(coords, coords)
        if fixed:
            if kernel == 'gaussian':
                self.kernel = 'fixed_gauss'
                ktype = 1
            elif kernel == 'bisquare':
                self.kernel = 'fixed_bisquare'
                ktype = 3
            elif kernel == 'exponential':
                self.kernel = 'fixed_exp'
                ktype = 5
            else:    
                print 'Unsupported kernel function ', kernel
        else:
            if kernel == 'gaussian':
            	self.kernel = 'adapt_gauss'
            	ktype = 2
            elif kernel == 'bisquare':
                self.kernel = 'adapt_bisquare'
                ktype = 4
            elif kernel == 'exponential':
                self.kernel = 'adapt_bisquare'
                ktype = 6
            else:
                print 'Unsupported kernel function ', kernel
        
        function = lambda x: getDiag[criterion](
                GWR(coords, y, x_loc, x, family=family,
                    fixed=fixed).fit())
         
        if ktype % 2 == 0: 
            int_score = True
        else:
            int_score = False
        self.int_score = int_score

        if search == 'golden_section':
            a,c = self. _init_section(x_glob, x_loc, coords)
            delta = 0.38197 #1 - (np.sqrt(5.0)-1.0)/2.0
            return golden_section(a, c, delta, function, tol, max_iter,
                    int_score)
        elif search == 'interval':
            return interval(l_bound, u_bound, interval, function,
                    int_score=False)  
        else:
            print 'Unsupported computational search method ', search

    def _init_section(self, x_glob, x_loc, coords):
        if len(x_glob) > 0:
            n_glob = x_glob.shape[1]
        else:
            n_glob = 0
        if len(x_loc) > 0:
            n_loc = x_loc.shape[1]
        else:
            n_loc = 0
        n_vars = n_glob + n_loc
        n = np.array(coords).shape[0]    
            
        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            tree = KDTree(coords)
            nn = 40 + 2 * n_vars
            min_dists = [tree.query(point, nn)[0][nn-1] for point in coords]
            max_dists = [tree.query(point, nn)[0][-1] for point in coords]
            a = np.min(min_dists)/2.0   
            c = np.max(max_dists)/2.0
        
        if a < self.bw_min:
            a = self.bw_min
        if c > self.bw_max and self.bw_max > 0:
            c = self.bw_max
                
        return a, c
        
    

