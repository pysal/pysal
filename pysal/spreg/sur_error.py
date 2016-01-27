"""
Spatial Error SUR estimation
"""

__author__= "Luc Anselin lanselin@gmail.com,    \
             Pedro V. Amaral pedrovma@gmail.com"
            

import numpy as np
import pysal
import numpy.linalg as la
import scipy.stats as stats
import summary_output as SUMMARY
import user_output as USER
from scipy.sparse.linalg import splu as SuperLU
from scipy.optimize import minimize_scalar, minimize
from scipy import sparse as sp

from ml_error import err_c_loglik_sp
from sur_utils import sur_dictxy,sur_corr,sur_dict2mat,\
               sur_crossprod,sur_est,sur_resids,filter_dict,\
               check_k
from sur import BaseSUR
from diagnostics_sur import sur_setp, lam_setp, sur_chow
from regimes import buildR,wald_test

__all__ = ["BaseSURerrorML","SURerrorML"]


class BaseSURerrorML():
    """Base class for SUR Error estimation by Maximum Likelihood
    
       requires: scipy.optimize.minimize_scalar and 
                 scipy.optimize.minimize
                 
    Parameters
    ----------
    bigy       : dictionary with vectors of dependent variable, one for
                 each equation
    bigX       : dictionary with matrices of explanatory variables,
                 one for each equation
    w          : spatial weights object
    epsilon    : convergence criterion for ML iterations
                 default 0.0000001
       
    Attributes
    ----------
    n          : number of observations in each cross-section
    n2         : n/2
    n_eq       : number of equations
    bigy       : dictionary with vectors of dependent variable, one for
                 each equation
    bigX       : dictionary with matrices of explanatory variables,
                 one for each equation
    bigK       : n_eq x 1 array with number of explanatory variables
                 by equation
    bigylag    : spatially lagged dependent variable
    bigXlag    : spatially lagged explanatory variable
    lamols     : spatial autoregressive coefficients from equation by
                 equation ML-Error estimation
    clikerr    : concentrated log-likelihood from equation by equation
                 ML-Error estimation (no constant)
    bSUR0      : SUR estimation for betas without spatial autocorrelation
    llik       : log-likelihood for classic SUR estimation (includes constant)
    lamsur     : spatial autoregressive coefficient in ML SUR Error
    bSUR       : beta coefficients in ML SUR Error
    varb       : variance of beta coefficients in ML SUR Error
    sig        : error variance-covariance matrix in ML SUR Error
    corr       : error correlation matrix
    bigE       : n by n_eq matrix of vectors of residuals for each equation
    cliksurerr : concentrated log-likelihood from ML SUR Error (no constant) 
      
    """    
    def __init__(self,bigy,bigX,w,epsilon=0.0000001):
        # setting up constants
        self.n = w.n
        self.n2 = self.n / 2.0
        self.n_eq = len(bigy.keys())
        WS = w.sparse
        I = sp.identity(self.n)
        # variables
        self.bigy = bigy
        self.bigX = bigX
        # number of variables by equation
        self.bigK = np.zeros((self.n_eq,1),dtype=np.int_)
        for r in range(self.n_eq):
            self.bigK[r] = self.bigX[r].shape[1]
        # spatially lagged variables
        self.bigylag = {}
        for r in range(self.n_eq):
            self.bigylag[r] = WS*self.bigy[r]
        # note: unlike WX as instruments, this includes the constant
        self.bigXlag = {}
        for r in range(self.n_eq):
            self.bigXlag[r] = WS*self.bigX[r]
            
        # spatial parameter starting values
        lam = np.zeros((self.n_eq,1))  # initialize as an array
        fun0 = 0.0
        fun1 = 0.0
        for r in range(self.n_eq):
            res = minimize_scalar(err_c_loglik_sp, 0.0, bounds=(-1.0,1.0),
                        args=(self.n, self.bigy[r], self.bigylag[r], 
                        self.bigX[r], self.bigXlag[r], I, WS),
                        method='bounded', options={'xatol':epsilon})
            lam[r]= res.x
            fun1 += res.fun
        self.lamols = lam
        self.clikerr = -fun1  #negative because use in min
        
        # SUR starting values
        reg0 = BaseSUR(self.bigy,self.bigX,iter=True)
        bigE = reg0.bigE
        self.bSUR0 = reg0.bSUR
        self.llik = reg0.llik  # as is, includes constant
        
        # iteration
        lambdabounds = [ (-1.0,+1.0) for i in range(self.n_eq)]
        while abs(fun0 - fun1) > epsilon:
            fun0 = fun1
            sply = filter_dict(lam,self.bigy,self.bigylag)
            splX = filter_dict(lam,self.bigX,self.bigXlag)
            WbigE = WS * bigE
            splbigE = bigE - WbigE * lam.T
            splXX,splXy = sur_crossprod(splX,sply)
            b1,varb1,sig1 = sur_est(splXX,splXy,splbigE,self.bigK)
            bigE = sur_resids(self.bigy,self.bigX,b1)
            res = minimize(clik,lam,args=(self.n,self.n2,self.n_eq,\
                            bigE,I,WS),method='L-BFGS-B',\
                            bounds=lambdabounds)
            lam = res.x
            lam.resize((self.n_eq,1))
            fun1 = res.fun
        self.lamsur = lam
        self.bSUR = b1
        self.varb = varb1
        self.sig = sig1
        self.corr = sur_corr(self.sig)
        self.bigE = bigE
        self.cliksurerr = -fun1  #negative because use in min, no constant
        
class SURerrorML(BaseSURerrorML):
    """User class for SUR Error estimation by Maximum Likelihood
    
    Parameters
    ----------
    bigy         : dictionary with vectors of dependent variable, one for
                   each equation
    bigX         : dictionary with matrices of explanatory variables,
                   one for each equation
    w            : spatial weights object
    epsilon      : convergence criterion for ML iterations
                   default 0.0000001
    nonspat_diag : boolean; flag for non-spatial diagnostics, default = True
    spat_diag    : boolean; flag for spatial diagnostics, default = False
    vm           : boolean; flag for asymptotic variance for lambda and Sigma,
                   default = False
    name_bigy    : dictionary with name of dependent variable for each equation
                     default = None, but should be specified
                     is done when sur_stackxy is used
    name_bigX    : dictionary with names of explanatory variables for each
                     equation
                     default = None, but should be specified
                     is done when sur_stackxy is used
    name_ds      : string; name for the data set
    name_w       : string; name for the weights file
       
    Attributes
    ----------
    n          : number of observations in each cross-section
    n2         : n/2
    n_eq       : number of equations
    bigy       : dictionary with vectors of dependent variable, one for
                 each equation
    bigX       : dictionary with matrices of explanatory variables,
                 one for each equation
    bigK       : n_eq x 1 array with number of explanatory variables
                 by equation
    bigylag    : spatially lagged dependent variable
    bigXlag    : spatially lagged explanatory variable
    lamols     : spatial autoregressive coefficients from equation by
                 equation ML-Error estimation
    clikerr    : concentrated log-likelihood from equation by equation
                 ML-Error estimation (no constant)
    bSUR0      : SUR estimation for betas without spatial autocorrelation
    llik       : log-likelihood for classic SUR estimation (includes constant)
    lamsur     : spatial autoregressive coefficient in ML SUR Error
    bSUR       : beta coefficients in ML SUR Error
    varb       : variance of beta coefficients in ML SUR Error
    sig        : error variance-covariance matrix in ML SUR Error
    bigE       : n by n_eq matrix of vectors of residuals for each equation
    cliksurerr : concentrated log-likelihood from ML SUR Error (no constant) 
    sur_inf    : inference for regression coefficients, stand. error, t, p
    errllik    : log-likelihood for error model without SUR (with constant)
    surerrllik : log-likelihood for SUR error model (with constant)
    lrtest     : likelihood ratio test for off-diagonal Sigma elements
    likrlambda : likelihood ratio test on spatial autoregressive coefficients
    vm         : asymptotic variance matrix for lambda and Sigma (only for vm=True)
    lamsetp    : inference for lambda, stand. error, t, p (only for vm=True)
    lamtest    : tuple with test for constancy of lambda across equations
                 (test value, degrees of freedom, p-value)
    joinlam    : tuple with test for joint significance of lambda across 
                 equations (test value, degrees of freedom, p-value)
    surchow    : list with tuples for Chow test on regression coefficients
                 each tuple contains test value, degrees of freedom, p-value
    name_bigy  : dictionary with name of dependent variable for each equation
    name_bigX  : dictionary with names of explanatory variables for each
                 equation
    name_ds    : string; name for the data set
    name_w     : string; name for the weights file      


    Examples
    --------

    First import pysal to load the spatial analysis tools.

    >>> import pysal

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open(). 
    This is the DBF associated with the NAT shapefile. Note that pysal.open() 
    also reads data in CSV format.

    >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')

    The specification of the model to be estimated can be provided as lists.
    Each equation should be listed separately. Equation 1 has HR80 as dependent 
    variable, and PS80 and UE80 as exogenous regressors. 
    For equation 2, HR90 is the dependent variable, and PS90 and UE90 the
    exogenous regressors.

    >>> y_var = ['HR80','HR90']
    >>> x_var = [['PS80','UE80'],['PS90','UE90']]
    >>> yend_var = [['RD80'],['RD90']]
    >>> q_var = [['FP79'],['FP89']]

    The SUR method requires data to be provided as dictionaries. PySAL
    provides the tool sur_dictxy to create these dictionaries from the
    list of variables. The line below will create four dictionaries
    containing respectively the dependent variables (bigy), the regressors
    (bigX), the dependent variables' names (bigyvars) and regressors' names
    (bigXvars). All these will be created from th database (db) and lists
    of variables (y_var and x_var) created above.

    >>> bigy,bigX,bigyvars,bigXvars = pysal.spreg.sur_utils.sur_dictxy(db,y_var,x_var)

    To run a spatial error model, we need to specify the spatial weights matrix. 
    To do that, we can open an already existing gal file or create a new one.
    In this example, we will create a new one from NAT.shp and transform it to
    row-standardized.

    >>> w = pysal.queen_from_shapefile(pysal.examples.get_path("NAT.shp"))
    >>> w.transform='r'

    We can now run the regression and then have a summary of the output by typing:
    print(reg.summary)

    Alternatively, we can just check the betas and standard errors, asymptotic t 
    and p-value of the parameters:        

    >>> reg = SURerrorML(bigy,bigX,w=w,name_bigy=bigyvars,name_bigX=bigXvars,name_ds="NAT",name_w="nat_queen")
    >>> reg.bSUR
    {0: array([[ 4.0222855 ],
           [ 0.88489646],
           [ 0.42402853]]), 1: array([[ 3.04923009],
           [ 1.10972634],
           [ 0.47075682]])}

    >>> reg.sur_inf
    {0: array([[  0.36692181,  10.96224141,   0.        ],
           [  0.14129077,   6.26294579,   0.        ],
           [  0.04267954,   9.93517021,   0.        ]]), 1: array([[  0.33139969,   9.20106497,   0.        ],
           [  0.13352591,   8.31094371,   0.        ],
           [  0.04004097,  11.756878  ,   0.        ]])}
           
    """        
    def __init__(self,bigy,bigX,w,nonspat_diag=True,spat_diag=False,vm=False,\
        epsilon=0.0000001,\
        name_bigy=None,name_bigX=None,name_ds=None,name_w=None):
        
        #need checks on match between bigy, bigX dimensions

        # moved init here
        BaseSURerrorML.__init__(self,bigy=bigy,bigX=bigX,w=w,epsilon=epsilon)

        # check on variable names for listing results
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        #initialize names - should be generated by sur_stack
        if name_bigy:
            self.name_bigy = name_bigy
        else: # need to construct y names
            self.name_bigy = {}
            for r in range(self.n_eq):
                yn = 'dep_var_' + str(r)
                self.name_bigy[r] = yn               
        if name_bigX:
            self.name_bigX = name_bigX
        else: # need to construct x names
            self.name_bigX = {}
            for r in range(self.n_eq):
                k = self.bigX[r].shape[1] - 1
                name_x = ['var_' + str(i + 1) + "_" + str(r+1) for i in range(k)]
                ct = 'Constant_' + str(r+1)  # NOTE: constant always included in X
                name_x.insert(0, ct)
                self.name_bigX[r] = name_x

        

                   
        #inference
        self.sur_inf = sur_setp(self.bSUR,self.varb)
        
        # adjust concentrated log lik for constant
        const = -self.n2 * (self.n_eq * (1.0 + np.log(2.0*np.pi)))
        self.errllik = const + self.clikerr
        self.surerrllik = const + self.cliksurerr
                
        # LR test on off-diagonal sigma
        if nonspat_diag:
            M = self.n_eq * (self.n_eq - 1)/2.0
            likrodiag = 2.0 * (self.surerrllik - self.errllik)
            plik1 = stats.chisqprob(likrodiag, M)
            self.lrtest = (likrodiag,int(M),plik1)
        else:
            self.lrtest = None
        
        # LR test on spatial autoregressive coefficients
        if spat_diag:
            liklambda = 2.0 * (self.surerrllik - self.llik)
            plik2 = stats.chisqprob(liklambda, self.n_eq)
            self.likrlambda = (liklambda,self.n_eq,plik2)
        else:
            self.likrlambda = None
        
        # asymptotic variance for spatial coefficient
        if vm:
            self.vm = surerrvm(self.n,self.n_eq,w,self.lamsur,self.sig)
            vlam = self.vm[:self.n_eq,:self.n_eq]
            self.lamsetp = lam_setp(self.lamsur,vlam)
            # test on constancy of lambdas
            R = buildR(kr=1,kf=0,nr=self.n_eq)
            w,p = wald_test(self.lamsur,R,np.zeros((R.shape[0],1)),vlam)
            self.lamtest = (w,R.shape[0],p)
            if spat_diag:  # test on joint significance of lambdas
                Rj = np.identity(self.n_eq)
                wj,pj = wald_test(self.lamsur,Rj,np.zeros((Rj.shape[0],1)),vlam)
                self.joinlam = (wj,Rj.shape[0],pj)
            else:
                self.joinlam = None
        else:
            self.vm = None
            self.lamsetp = None
            self.lamtest = None
            self.joinlam = None

        # test on constancy of regression coefficients across equations
        if check_k(self.bigK):   # only for equal number of variables
            self.surchow = sur_chow(self.n_eq,self.bigK,self.bSUR,self.varb)
        else:
            self.surchow = None          
                    
        # listing of results
        self.title = "SEEMINGLY UNRELATED REGRESSIONS (SUR) - SPATIAL ERROR MODEL"                
        SUMMARY.SUR(reg=self, nonspat_diag=nonspat_diag, spat_diag=spat_diag, lambd=True)



def jacob(lam,n_eq,I,WS):
    """Log-Jacobian for SUR Error model
    
    Parameters
    ----------
    lam      : n_eq by 1 array of spatial autoregressive parameters
    n_eq     : number of equations
    I        : sparse Identity matrix
    WS       : sparse spatial weights matrix
    
    Returns
    -------
    logjac   : the log Jacobian
    
    """
    logjac = 0.0
    for r in range(n_eq):
        lami = lam[r]
        lamWS = WS.multiply(lami)
        B = (I - lamWS).tocsc()
        LU = SuperLU(B)
        jj = np.sum(np.log(np.abs(LU.U.diagonal())))
        logjac += jj
    return logjac

def clik(lam,n,n2,n_eq,bigE,I,WS):
    """ Concentrated (negative) log-likelihood for SUR Error model
    
    Parameters
    ----------
    lam         : n_eq x 1 array of spatial autoregressive parameters
    n           : number of observations in each cross-section
    n2          : n/2
    n_eq        : number of equations
    bigE        : n by n_eq matrix with vectors of residuals for 
                  each equation
    I           : sparse Identity matrix
    WS          : sparse spatial weights matrix
    
    Returns
    -------
    -clik       : negative (for minimize) of the concentrated
                  log-likelihood function
    
    """
    WbigE = WS * bigE
    spfbigE = bigE - WbigE * lam.T
    sig = np.dot(spfbigE.T,spfbigE) / n
    ldet = la.slogdet(sig)[1]
    logjac = jacob(lam,n_eq,I,WS)
    clik = - n2 * ldet + logjac
    return -clik  # negative for minimize

def surerrvm(n,n_eq,w,lam,sig):
    """Asymptotic variance matrix for lambda and Sigma in
       ML SUR Error estimation
       
       Source: Anselin (1988), Chapter 10.
       
    Parameters
    ----------
    n         : scalar, number of cross-sectional observations
    n_eq      : scalar, number of equations
    w         : spatial weights object
    lam       : n_eq by 1 vector with spatial autoregressive coefficients
    sig       : n_eq by n_eq matrix with cross-equation error covariances
    
    Returns
    -------
    vm        : asymptotic variance-covariance matrix for spatial autoregressive
                coefficients and the upper triangular elements of Sigma
                n_eq + n_eq x (n_eq + 1) / 2 coefficients
    
    
    """
    # inverse Sigma
    sigi = la.inv(sig)
    sisi = sigi * sig
    # elements of Psi_lam,lam
    # trace terms
    trDi = np.zeros((n_eq,1))
    trDDi = np.zeros((n_eq,1))
    trDTDi = np.zeros((n_eq,1))
    trDTiDj = np.zeros((n_eq,n_eq))
    WS = w.sparse
    I = sp.identity(n)
    for i in range(n_eq):
        lami = lam[i][0]
        lamWS = WS.multiply(lami)
        B = (I - lamWS)
        bb = B.todense()
        Bi = la.inv(bb)
        D = WS * Bi
        trDi[i] = np.trace(D)
        DD = np.dot(D,D)
        trDDi[i] = np.trace(DD)
        DD = np.dot(D.T,D)
        trDTDi[i] = np.trace(DD)
        for j in range(i+1,n_eq):
            lamj = lam[j][0]
            lamWS = WS.multiply(lamj)
            B = (I - lamWS)
            bb = B.todense()
            Bi = la.inv(bb)
            Dj = WS * Bi
            DD = np.dot(D.T,Dj)
            trDTiDj[i,j] = np.trace(DD)
            trDTiDj[j,i] = trDTiDj[i,j]
    np.fill_diagonal(trDTiDj,trDTDi)   
    
    sisjT = sisi * trDTiDj
    Vll = np.diagflat(trDDi) + sisjT
    
    # elements of Psi_lam_sig
    P = int(n_eq * (n_eq + 1)/2) #force ints to be ints
    tlist = [ (i,j) for i in range(n_eq) for j in range(i,n_eq) ]
    zog = sigi * trDi
    Vlsig = np.zeros((n_eq,P))
    for i in range(n_eq):
        for j in range(n_eq):
            if i > j:
                jj = tlist.index((j,i))
            else:
                jj = tlist.index((i,j))
            Vlsig[i,jj] = zog[i,j]
            
    # top of Psi
    vtop = np.hstack((Vll,Vlsig))
    
    # elements of Psi_sig_sig
    
    Vsig = np.zeros((P,P))
    for ij in range(P):
        i,j = tlist[ij]    
        for hk in range(P):
            h,k = tlist[hk]
            if i == j:
                if h == k:
                    Vsig[ij,hk] = 0.5 * (sigi[i,h]**2)
                else:  # h not equal to k
                    Vsig[ij,hk] = sigi[i,h]*sigi[i,k]
            else:  # i not equal to j
                if h == k:
                    Vsig[ij,hk] = sigi[i,h]*sigi[j,h]
                else:  # h not equal to k
                    Vsig[ij,hk] = sigi[i,h]*sigi[j,k] + sigi[i,k]*sigi[j,h]
    Vsig = n * Vsig 
    
    # bottom of Psi
    vbottom = np.hstack((Vlsig.T,Vsig))
    
    # all of Psi
    vbig = np.vstack((vtop,vbottom))
    
    # inverse of Psi
    vm = la.inv(vbig)
    
    return vm
        
def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    from sur_utils import sur_dictxy,sur_dictZ

    db = pysal.open(pysal.examples.get_path('NAT.dbf'), 'r')
    y_var = ['HR80','HR90']
    x_var = [['PS80','UE80'],['PS90','UE90']]
    w = pysal.queen_from_shapefile(pysal.examples.get_path("NAT.shp"))
    w.transform='r'
    bigy0,bigX0,bigyvars0,bigXvars0 = sur_dictxy(db,y_var,x_var)
    reg0 = SURerrorML(bigy0,bigX0,w,name_bigy=bigyvars0,name_bigX=bigXvars0,\
        name_w="natqueen",name_ds="natregimes",vm=False,nonspat_diag=True,spat_diag=True)
    #print reg0.summary  
