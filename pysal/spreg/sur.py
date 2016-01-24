"""
SUR and 3SLS estimation
"""

__author__= "Luc Anselin lanselin@gmail.com,    \
             Pedro V. Amaral pedrovma@gmail.com"
            

import numpy as np
import numpy.linalg as la
import scipy.stats as stats
import summary_output as SUMMARY
import user_output as USER
from sur_utils import sur_dict2mat,sur_mat2dict,sur_corr,\
                      sur_crossprod,sur_est,sur_resids,check_k
from diagnostics_sur import sur_setp,sur_lrtest,sur_lmtest,surLMe,sur_chow


__all__ = ['SUR','ThreeSLS']


class BaseSUR():
    """ Base class for SUR estimation, both two step as well as iterated
    
        Parameters
        ----------
        
        bigy       : dictionary with vector for dependent variable by equation
        bigX       : dictionary with matrix of explanatory variables by equation
                     (note, already includes constant term)
        iter       : whether or not to use iterated estimation
                     default = False
        maxiter    : maximum iterations; default = 5
        epsilon    : precision criterion to end iterations
                     default = 0.00001
        verbose    : flag to print out iteration number and value of log det(sig)
                     at the beginning and the end of the iteration

                     
        Attributes
        ----------
        
        bigy        : dictionary with y values
        bigX        : dictionary with X values
        bigXX       : dictionary with X_t'X_r cross-products
        bigXy       : dictionary with X_t'y_r cross-products
        n_eq        : number of equations
        n           : number of observations in each cross-section
        bigK        : vector with number of explanatory variables (including constant)
                      for each equation
        bOLS        : dictionary with OLS regression coefficients for each equation
        olsE        : N x n_eq array with OLS residuals for each equation
        bSUR        : dictionary with SUR regression coefficients for each equation
        varb        : variance-covariance matrix
        bigE        : N x n_eq array with SUR residuals for each equation
        sig         : Sigma matrix of inter-equation error covariances
        ldetS1      : log det(Sigma) for SUR model
        resids      : n by n_eq array of residuals
        sig_ols     : Sigma matrix for OLS residuals 
        ldetS0      : log det(Sigma) for null model (OLS by equation, diagonals only)
        niter       : number of iterations (=0 for iter=False)
        corr        : inter-equation SUR error correlation matrix
        llik        : log-likelihood (including the constant pi)
        
        
        Methods
        -------

        sur_ols         : OLS estimation by equation

    """
    def __init__(self,bigy,bigX,iter=False,maxiter=5,epsilon=0.00001,verbose=False):
        # setting up the cross-products
        self.bigy = bigy
        self.bigX = bigX
        self.n_eq = len(bigy.keys())
        self.n = bigy[0].shape[0]
        self.bigK = np.zeros((self.n_eq,1),dtype=np.int_)
        for r in range(self.n_eq):
            self.bigK[r] = self.bigX[r].shape[1]
        self.bigXX,self.bigXy = sur_crossprod(self.bigX,self.bigy)
        # OLS regression by equation, sets up initial residuals
        self.sur_ols() # creates self.bOLS and self.olsE
        # SUR estimation using OLS residuals - two step estimation
        self.bSUR,self.varb,self.sig = sur_est(self.bigXX,self.bigXy,self.olsE,self.bigK)
        resids = sur_resids(self.bigy,self.bigX,self.bSUR)  # matrix of residuals
        # Sigma and log det(Sigma) for null model
        self.sig_ols = self.sig
        sols = np.diag(np.diag(self.sig))
        self.ldetS0 = np.log(np.diag(sols)).sum()
        det0 = self.ldetS0
        # setup for iteration
        det1 = la.slogdet(self.sig)[1]
        self.ldetS1 = det1
        #self.niter = 0
        if iter:    # iterated FGLS aka ML
            n_iter = 0
            while np.abs(det1-det0) > epsilon and n_iter <= maxiter:
                n_iter += 1
                det0 = det1
                self.bSUR,self.varb,self.sig = sur_est(self.bigXX,self.bigXy,\
                          resids,self.bigK)
                resids = sur_resids(self.bigy,self.bigX,self.bSUR)
                det1 = la.slogdet(self.sig)[1]
                if verbose:
                    print (n_iter,det0,det1)
            self.bigE = sur_resids(self.bigy,self.bigX,self.bSUR)
            self.ldetS1 = det1
            self.niter = n_iter
        else:
            self.niter = 1
            self.bigE = resids
        self.corr = sur_corr(self.sig)
        lik = self.n_eq * (1.0 + np.log(2.0*np.pi)) + self.ldetS1
        self.llik = - (self.n / 2.0) * lik
                

    def sur_ols(self):
        '''OLS estimation of SUR equations
    
           Parameters
           ----------
    
           self  : BaseSUR object
    
           Creates
           -------

           self.bOLS    : dictionary with regression coefficients for each equation
           self.olsE    : N x n_eq array with OLS residuals for each equation
                  
        '''
        self.bOLS = {}
        for r in range(self.n_eq):
            self.bOLS[r] = np.dot(la.inv(self.bigXX[(r,r)]),self.bigXy[(r,r)])
        self.olsE = sur_resids(self.bigy,self.bigX,self.bOLS)
        



class SUR(BaseSUR):
    """ User class for SUR estimation, both two step as well as iterated
    
        Parameters
        ----------
        
        bigy       : dictionary with vector for dependent variable by equation
        bigX       : dictionary with matrix of explanatory variables by equation
                     (note, already includes constant term)
        w          : spatial weights object, default = None
        nonspat_diag : boolean; flag for non-spatial diagnostics, default = True
        spat_diag  : boolean; flag for spatial diagnostics, default = False
        iter       : boolean; whether or not to use iterated estimation
                     default = False
        maxiter    : integer; maximum iterations; default = 5
        epsilon    : float; precision criterion to end iterations
                     default = 0.00001
        verbose    : boolean; flag to print out iteration number and value 
                     of log det(sig) at the beginning and the end of the iteration
        name_bigy  : dictionary with name of dependent variable for each equation
                     default = None, but should be specified
                     is done when sur_stackxy is used
        name_bigX  : dictionary with names of explanatory variables for each
                     equation
                     default = None, but should be specified
                     is done when sur_stackxy is used
        name_ds    : string; name for the data set
        name_w     : string; name for the weights file

                     
        Attributes
        ----------
        
        bigy        : dictionary with y values
        bigX        : dictionary with X values
        bigXX       : dictionary with X_t'X_r cross-products
        bigXy       : dictionary with X_t'y_r cross-products
        n_eq        : number of equations
        n           : number of observations in each cross-section
        bigK        : vector with number of explanatory variables (including constant)
                      for each equation
        bOLS        : dictionary with OLS regression coefficients for each equation
        olsE        : N x n_eq array with OLS residuals for each equation
        bSUR        : dictionary with SUR regression coefficients for each equation
        varb        : variance-covariance matrix
        sig         : Sigma matrix of inter-equation error covariances
        ldetS1      : log det(Sigma) for SUR model
        bigE        : n by n_eq array of residuals
        sig_ols     : Sigma matrix for OLS residuals (diagonal)
        ldetS0      : log det(Sigma) for null model (OLS by equation)
        niter       : number of iterations (=0 for iter=False)
        corr        : inter-equation error correlation matrix
        llik        : log-likelihood (including the constant pi)
        sur_inf     : dictionary with standard error, asymptotic t and p-value,
                      one for each equation
        lrtest      : Likelihood Ratio test on off-diagonal elements of sigma
                      (tupel with test,df,p-value)
        lmtest      : Lagrange Multipler test on off-diagonal elements of sigma
                      (tupel with test,df,p-value)
        lmEtest     : Lagrange Multiplier test on error spatial autocorrelation in SUR
        surchow     : list with tuples for Chow test on regression coefficients
                      each tuple contains test value, degrees of freedom, p-value
        name_bigy   : dictionary with name of dependent variable for each equation
        name_bigX   : dictionary with names of explanatory variables for each
                      equation
        name_ds     : string; name for the data set
        name_w      : string; name for the weights file        


        Examples
        --------

        First import pysal to load the spatial analysis tools.

        >>> import pysal

        Open data on NCOVR US County Homicides (3085 areas) using pysal.open(). 
        This is the DBF associated with the NAT shapefile. Note that pysal.open() 
        also reads data in CSV format.

        >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')

        The specification of the model to be estimated can be provided as lists.
        Each equation should be listed separately. In this example, equation 1
        has HR80 as dependent variable and PS80 and UE80 as exogenous regressors.
        For equation 2, HR90 is the dependent variable, and PS90 and UE90 the
        exogenous regressors.

        >>> y_var = ['HR80','HR90']
        >>> x_var = [['PS80','UE80'],['PS90','UE90']]

        Although not required for this method, we can load a weights matrix file
        to allow for spatial diagnostics.

        >>> w = pysal.queen_from_shapefile(pysal.examples.get_path("NAT.shp"))
        >>> w.transform='r'

        The SUR method requires data to be provided as dictionaries. PySAL
        provides the tool sur_dictxy to create these dictionaries from the
        list of variables. The line below will create four dictionaries
        containing respectively the dependent variables (bigy), the regressors
        (bigX), the dependent variables' names (bigyvars) and regressors' names
        (bigXvars). All these will be created from th database (db) and lists
        of variables (y_var and x_var) created above.

        >>> bigy,bigX,bigyvars,bigXvars = pysal.spreg.sur_utils.sur_dictxy(db,y_var,x_var)

        We can now run the regression and then have a summary of the output by typing:
        'print(reg.summary)'  

        >>> reg = SUR(bigy,bigX,w=w,name_bigy=bigyvars,name_bigX=bigXvars,spat_diag=True,name_ds="nat")
        >>> print(reg.summary)
        REGRESSION
        ----------
        SUMMARY OF OUTPUT: SEEMINGLY UNRELATED REGRESSIONS (SUR)
        --------------------------------------------------------
        Data set            :         nat
        Weights matrix      :     unknown
        Number of Equations :           2                Number of Observations:        3085
        Log likelihood (SUR):  -19902.966                Number of Iterations  :           1
        ----------
        <BLANKLINE>
        SUMMARY OF EQUATION 1
        ---------------------
        Dependent Variable  :        HR80                Number of Variables   :           3
        Mean dependent var  :      6.9276                Degrees of Freedom    :        3082
        S.D. dependent var  :      6.8251
        <BLANKLINE>
        ------------------------------------------------------------------------------------
                    Variable     Coefficient       Std.Error     z-Statistic     Probability
        ------------------------------------------------------------------------------------
                  Constant_1       5.1390718       0.2624673      19.5798587       0.0000000
                        PS80       0.6776481       0.1219578       5.5564132       0.0000000
                        UE80       0.2637240       0.0343184       7.6846277       0.0000000
        ------------------------------------------------------------------------------------
        <BLANKLINE>
        SUMMARY OF EQUATION 2
        ---------------------
        Dependent Variable  :        HR90                Number of Variables   :           3
        Mean dependent var  :      6.1829                Degrees of Freedom    :        3082
        S.D. dependent var  :      6.6403
        <BLANKLINE>
        ------------------------------------------------------------------------------------
                    Variable     Coefficient       Std.Error     z-Statistic     Probability
        ------------------------------------------------------------------------------------
                  Constant_2       3.6139403       0.2534996      14.2561949       0.0000000
                        PS90       1.0260715       0.1121662       9.1477755       0.0000000
                        UE90       0.3865499       0.0341996      11.3027760       0.0000000
        ------------------------------------------------------------------------------------
        <BLANKLINE>
        <BLANKLINE>
        REGRESSION DIAGNOSTICS
                                             TEST         DF       VALUE           PROB
                                 LM test on Sigma         1      680.168           0.0000
                                 LR test on Sigma         1      768.385           0.0000
        <BLANKLINE>
        OTHER DIAGNOSTICS - CHOW TEST
                                        VARIABLES         DF       VALUE           PROB
                           Constant_1, Constant_2         1       26.729           0.0000
                                       PS80, PS90         1        8.241           0.0041
                                       UE80, UE90         1        9.384           0.0022
        <BLANKLINE>
        DIAGNOSTICS FOR SPATIAL DEPENDENCE
        TEST                              DF       VALUE           PROB
        Lagrange Multiplier (error)       2        1333.625        0.0000
        <BLANKLINE>
        ERROR CORRELATION MATRIX
          EQUATION 1  EQUATION 2
            1.000000    0.469548
            0.469548    1.000000
        ================================ END OF REPORT =====================================
    """

    def __init__(self,bigy,bigX,w=None,nonspat_diag=True,spat_diag=False,vm=False,\
        iter=False,maxiter=5,epsilon=0.00001,verbose=False,\
        name_bigy=None,name_bigX=None,name_ds=None,name_w=None):
        
        #need checks on match between bigy, bigX dimensions
        # init moved here before name check
        BaseSUR.__init__(self,bigy=bigy,bigX=bigX,iter=iter,\
            maxiter=maxiter,epsilon=epsilon,verbose=verbose)

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
                name_x = ['var_' + str(i + 1) + "_" + str(r) for i in range(k)]
                ct = 'Constant_' + str(r)  # NOTE: constant always included in X
                name_x.insert(0, ct)
                self.name_bigX[r] = name_x
        
        #inference
        self.sur_inf = sur_setp(self.bSUR,self.varb)
        
        if nonspat_diag:
            #LR test on off-diagonal elements of Sigma
            self.lrtest = sur_lrtest(self.n,self.n_eq,self.ldetS0,self.ldetS1)
        
            #LM test on off-diagonal elements of Sigma
            self.lmtest = sur_lmtest(self.n,self.n_eq,self.sig_ols)
        else:
            self.lrtest = None
            self.lmtest = None
        
        #LM test on spatial error autocorrelation
        if spat_diag:
            if not w:
                 raise Exception, "Error: spatial weights needed"
            WS = w.sparse
            self.lmEtest = surLMe(self.n_eq,WS,self.bigE,self.sig)
        else:
            self.lmEtest = None
        
        #LM test on spatial lag autocorrelation
        
        # test on constancy of coefficients across equations
        if check_k(self.bigK):   # only for equal number of variables
            self.surchow = sur_chow(self.n_eq,self.bigK,self.bSUR,self.varb)
        else:
            self.surchow = None
        
        #Listing of the results
        self.title = "SEEMINGLY UNRELATED REGRESSIONS (SUR)"                
        SUMMARY.SUR(reg=self, nonspat_diag=nonspat_diag, spat_diag=spat_diag, surlm=True)
        
class BaseThreeSLS():
    """ Base class for 3SLS estimation, two step
    
        Parameters
        ----------
        
        bigy       : dictionary with vector for dependent variable by equation
        bigX       : dictionary with matrix of explanatory variables by equation
                     (note, already includes constant term)
        bigyend    : dictionary with matrix of endogenous variables by equation
        bigq       : dictionary with matrix of instruments by equation

                     
        Attributes
        ----------
        
        bigy        : dictionary with y values
        bigZ        : dictionary with matrix of exogenous and endogenous variables
                      for each equation
        bigZHZH     : dictionary with matrix of cross products Zhat_r'Zhat_s
        bigZHy      : dictionary with matrix of cross products Zhat_r'y_end_s
        n_eq        : number of equations
        n           : number of observations in each cross-section
        bigK        : vector with number of explanatory variables (including constant,
                      exogenous and endogenous) for each equation
        b2SLS       : dictionary with 2SLS regression coefficients for each equation
        tslsE       : N x n_eq array with OLS residuals for each equation
        b3SLS       : dictionary with 3SLS regression coefficients for each equation
        varb        : variance-covariance matrix
        sig         : Sigma matrix of inter-equation error covariances
        bigE        : n by n_eq array of residuals
        corr        : inter-equation 3SLS error correlation matrix
        
        
        Methods
        -------

        tsls_2sls       : 2SLS estimation by equation
    
    """
    def __init__(self,bigy,bigX,bigyend,bigq):
        # setting up the cross-products
        self.bigy = bigy
        self.n_eq = len(bigy.keys())
        self.n = bigy[0].shape[0]
        # dictionary with exog and endog, Z
        self.bigZ = {}
        for r in range(self.n_eq):
            self.bigZ[r] = np.hstack((bigX[r],bigyend[r]))   
        # number of explanatory variables by equation     
        self.bigK = np.zeros((self.n_eq,1),dtype=np.int_)
        for r in range(self.n_eq):
            self.bigK[r] = self.bigZ[r].shape[1]
        # dictionary with instruments, H
        bigH = {}
        for r in range(self.n_eq):
            bigH[r] = np.hstack((bigX[r],bigq[r]))    
        # dictionary with instrumental variables, X and yend_predicted, Z-hat
        bigZhat = {}
        for r in range(self.n_eq):
            try:
                HHi = la.inv(np.dot(bigH[r].T,bigH[r]))
            except:
                raise Exception, "ERROR: singular cross product matrix, check instruments"
            Hye = np.dot(bigH[r].T,bigyend[r])
            yp = np.dot(bigH[r],np.dot(HHi,Hye))
            bigZhat[r] = np.hstack((bigX[r],yp))
        self.bigZHZH,self.bigZHy = sur_crossprod(bigZhat,self.bigy)
         
        # 2SLS regression by equation, sets up initial residuals
        self.sur_2sls() # creates self.b2SLS and self.tslsE

        self.b3SLS,self.varb,self.sig = sur_est(self.bigZHZH,self.bigZHy,self.tslsE,self.bigK)
        self.bigE = sur_resids(self.bigy,self.bigZ,self.b3SLS)  # matrix of residuals
        
        # inter-equation correlation matrix
        self.corr = sur_corr(self.sig)
                

    def sur_2sls(self):
        '''2SLS estimation of SUR equations
    
           Parameters
           ----------
    
           self  : BaseSUR object
    
           Creates
           -------

           self.b2SLS    : dictionary with regression coefficients for each equation
           self.tslsE    : N x n_eq array with OLS residuals for each equation
                  
        '''
        self.b2SLS = {}
        for r in range(self.n_eq):
            self.b2SLS[r] = np.dot(la.inv(self.bigZHZH[(r,r)]),self.bigZHy[(r,r)])
        self.tslsE = sur_resids(self.bigy,self.bigZ,self.b2SLS)
        

class ThreeSLS(BaseThreeSLS):
    """ User class for 3SLS estimation
    
        Parameters
        ----------
        
        bigy       : dictionary with vector for dependent variable by equation
        bigX       : dictionary with matrix of explanatory variables by equation
                     (note, already includes constant term)
        bigyend    : dictionary with matrix of endogenous variables by equation
        bigq       : dictionary with matrix of instruments by equation
        nonspat_diag : boolean; flag for non-spatial diagnostics, default = True
        name_bigy  : dictionary with name of dependent variable for each equation
                     default = None, but should be specified
                     is done when sur_stackxy is used
        name_bigX  : dictionary with names of explanatory variables for each
                     equation
                     default = None, but should be specified
                     is done when sur_stackxy is used
        name_bigyend : dictionary with names of endogenous variables for each
                       equation
                       default = None, but should be specified
                       is done when sur_stackZ is used
        name_bigq  : dictionary with names of instrumental variables for each
                     equations
                     default = None, but should be specified
                     is done when sur_stackZ is used
        name_ds    : string; name for the data set

                     
        Attributes
        ----------
        
        bigy        : dictionary with y values
        bigZ        : dictionary with matrix of exogenous and endogenous variables
                      for each equation
        bigZHZH     : dictionary with matrix of cross products Zhat_r'Zhat_s
        bigZHy      : dictionary with matrix of cross products Zhat_r'y_end_s
        n_eq        : number of equations
        n           : number of observations in each cross-section
        bigK        : vector with number of explanatory variables (including constant,
                      exogenous and endogenous) for each equation
        b2SLS       : dictionary with 2SLS regression coefficients for each equation
        tslsE       : N x n_eq array with OLS residuals for each equation
        b3SLS       : dictionary with 3SLS regression coefficients for each equation
        varb        : variance-covariance matrix
        sig         : Sigma matrix of inter-equation error covariances
        bigE        : n by n_eq array of residuals
        corr        : inter-equation 3SLS error correlation matrix
        tsls_inf    : dictionary with standard error, asymptotic t and p-value,
                      one for each equation
        surchow     : list with tuples for Chow test on regression coefficients
                      each tuple contains test value, degrees of freedom, p-value
        name_ds    : string; name for the data set
        name_bigy  : dictionary with name of dependent variable for each equation
        name_bigX  : dictionary with names of explanatory variables for each
                     equation
        name_bigyend : dictionary with names of endogenous variables for each
                       equation
        name_bigq  : dictionary with names of instrumental variables for each
                     equations
        

        Examples
        --------

        First import pysal to load the spatial analysis tools.

        >>> import pysal

        Open data on NCOVR US County Homicides (3085 areas) using pysal.open(). 
        This is the DBF associated with the NAT shapefile. Note that pysal.open() 
        also reads data in CSV format.

        >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')

        The specification of the model to be estimated can be provided as lists.
        Each equation should be listed separately. In this example, equation 1
        has HR80 as dependent variable, PS80 and UE80 as exogenous regressors,
        RD80 as endogenous regressor and FP79 as additional instrument.
        For equation 2, HR90 is the dependent variable, PS90 and UE90 the
        exogenous regressors, RD90 as endogenous regressor and FP99 as 
        additional instrument

        >>> y_var = ['HR80','HR90']
        >>> x_var = [['PS80','UE80'],['PS90','UE90']]
        >>> yend_var = [['RD80'],['RD90']]
        >>> q_var = [['FP79'],['FP89']]

        The SUR method requires data to be provided as dictionaries. PySAL
        provides two tools to create these dictionaries from the list of variables: 
        sur_dictxy and sur_dictZ. The tool sur_dictxy can be used to create the
        dictionaries for Y and X, and sur_dictZ for endogenous variables (yend) and
        additional instruments (q).

        >>> bigy,bigX,bigyvars,bigXvars = pysal.spreg.sur_utils.sur_dictxy(db,y_var,x_var)
        >>> bigyend,bigyendvars = pysal.spreg.sur_utils.sur_dictZ(db,yend_var)
        >>> bigq,bigqvars = pysal.spreg.sur_utils.sur_dictZ(db,q_var)

        We can now run the regression and then have a summary of the output by typing:
        print(reg.summary)

        Alternatively, we can just check the betas and standard errors, asymptotic t 
        and p-value of the parameters:        

        >>> reg = ThreeSLS(bigy,bigX,bigyend,bigq,name_bigy=bigyvars,name_bigX=bigXvars,name_bigyend=bigyendvars,name_bigq=bigqvars,name_ds="NAT")
        >>> reg.b3SLS
        {0: array([[ 6.92426353],
               [ 1.42921826],
               [ 0.00049435],
               [ 3.5829275 ]]), 1: array([[ 7.62385875],
               [ 1.65031181],
               [-0.21682974],
               [ 3.91250428]])}

        >>> reg.tsls_inf
        {0: array([[  0.23220853,  29.81916157,   0.        ],
               [  0.10373417,  13.77770036,   0.        ],
               [  0.03086193,   0.01601807,   0.98721998],
               [  0.11131999,  32.18584124,   0.        ]]), 1: array([[  0.28739415,  26.52753638,   0.        ],
               [  0.09597031,  17.19606554,   0.        ],
               [  0.04089547,  -5.30204786,   0.00000011],
               [  0.13586789,  28.79638723,   0.        ]])}

    """

    def __init__(self,bigy,bigX,bigyend,bigq,nonspat_diag=True,\
        name_bigy=None,name_bigX=None,name_bigyend=None,name_bigq=None,\
        name_ds=None):
        
        #need checks on match between bigy, bigX dimensions
        BaseThreeSLS.__init__(self,bigy=bigy,bigX=bigX,bigyend=bigyend,\
            bigq=bigq)

        self.name_ds = USER.set_name_ds(name_ds)
        #initialize names - should be generated by sur_stack
        if name_bigy:
            self.name_bigy = name_bigy
        else: # need to construct y names
            self.name_bigy = {}
            for r in range(self.n_eq):
                yn = 'dep_var_' + str(r+1)
                self.name_bigy[r] = yn               
        if name_bigX:
            self.name_bigX = name_bigX
        else: # need to construct x names
            self.name_bigX = {}
            for r in range(self.n_eq):
                k = bigX[r].shape[1] - 1
                name_x = ['var_' + str(i + 1) + "_" + str(r+1) for i in range(k)]
                ct = 'Constant_' + str(r+1)  # NOTE: constant always included in X
                name_x.insert(0, ct)
                self.name_bigX[r] = name_x
        if name_bigyend:
            self.name_bigyend = name_bigyend
        else: # need to construct names
            self.name_bigyend = {}
            for r in range(self.n_eq):
                ky = bigyend[r].shape[1]
                name_ye = ['end_' + str(i + 1) + "_" + str(r+1) for i in range(ky)]
                self.name_bigyend[r] = name_ye
        if name_bigq:
            self.name_bigq = name_bigq
        else: # need to construct names
            self.name_bigq = {}
            for r in range(self.n_eq):
                ki = bigq[r].shape[1]
                name_i = ['inst_' + str(i + 1) + "_" + str(r+1) for i in range(ki)]
                self.name_bigq[r] = name_i                
               
        #inference
        self.tsls_inf = sur_setp(self.b3SLS,self.varb)
        
        # test on constancy of coefficients across equations
        if check_k(self.bigK):   # only for equal number of variables
            self.surchow = sur_chow(self.n_eq,self.bigK,self.b3SLS,self.varb)
        else:
            self.surchow = None                
        
        #Listing of the results
        self.title = "THREE STAGE LEAST SQUARES (3SLS)"                
        SUMMARY.SUR(reg=self, tsls=True, nonspat_diag=nonspat_diag)
        
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
    #Example SUR
    #"""
    w = pysal.queen_from_shapefile(pysal.examples.get_path("NAT.shp"))
    w.transform='r'
    bigy0,bigX0,bigyvars0,bigXvars0 = sur_dictxy(db,y_var,x_var)
    reg0 = SUR(bigy0,bigX0,w=w,name_bigy=bigyvars0,name_bigX=bigXvars0,\
          spat_diag=True,name_ds="nat")
    print reg0.summary       
    """
    #Example 3SLS
    yend_var = [['RD80'],['RD90']]
    q_var = [['FP79'],['FP89']]

    bigy1,bigX1,bigyvars1,bigXvars1 = sur_dictxy(db,y_var,x_var)
    bigyend1,bigyendvars1 = sur_dictZ(db,yend_var)
    bigq1,bigqvars1 = sur_dictZ(db,q_var)

    reg1 = ThreeSLS(bigy1,bigX1,bigyend1,bigq1,name_ds="nat")
    
    print reg1.summary       
    #"""
