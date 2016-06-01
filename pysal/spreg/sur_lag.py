"""
Spatial Lag SUR estimation
"""

__author__= "Luc Anselin lanselin@gmail.com,    \
             Pedro V. Amaral pedrovma@gmail.com"
            

import numpy as np
import pysal
import summary_output as SUMMARY
import user_output as USER
from sur import BaseThreeSLS
from diagnostics_sur import sur_setp, sur_chow, sur_joinrho
from sur_utils import check_k

__all__ = ["SURlagIV"]

class SURlagIV(BaseThreeSLS):
    """ User class for spatial lag estimation using IV
    
        Parameters
        ----------
        
        bigy       : dictionary with vector for dependent variable by equation
        bigX       : dictionary with matrix of explanatory variables by equation
                     (note, already includes constant term)
        bigyend    : dictionary with matrix of endogenous variables by equation
                     (optional)
        bigq       : dictionary with matrix of instruments by equation
                     (optional)
        w          : spatial weights object, required
        vm         : boolean
                     listing of full variance-covariance matrix, default = False
        w_lags     : integer
                     order of spatial lags for WX instruments, default = 1
        lag_q      : boolean
                     flag to apply spatial lag to other instruments,
                     default = True
        nonspat_diag : boolean; flag for non-spatial diagnostics, default = True
        spat_diag    : boolean; flag for spatial diagnostics, default = False
        name_bigy  : dictionary with name of dependent variable for each equation
                     default = None, but should be specified
                     is done when sur_stackxy is used
        name_bigX  : dictionary with names of explanatory variables for each
                     equation
                     default = None, but should be specified
                     is done when sur_stackxy is used
        name_bigyend : dictionary with names of endogenous variables for each
                       equation
                       default = None, but should be spedified
                       is done when sur_stackZ is used
        name_bigq  : dictionary with names of instrumental variables for each
                     equations
                     default = None, but should be specified
                     is done when sur_stackZ is used
        name_ds    : string; name for the data set
        name_w     : string; name for the spatial weights

                     
        Attributes
        ----------
        
        w           : spatial weights object
        bigy        : dictionary with y values
        bigZ        : dictionary with matrix of exogenous and endogenous variables
                      for each equation
        bigyend     : dictionary with matrix of endogenous variables for each
                      equation; contains Wy only if no other endogenous specified
        bigq        : dictionary with matrix of instrumental variables for each
                      equation; contains WX only if no other endogenous specified
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
        resids      : n by n_eq array of residuals
        corr        : inter-equation 3SLS error correlation matrix
        tsls_inf    : dictionary with standard error, asymptotic t and p-value,
                      one for each equation
        joinrho     : test on joint significance of spatial autoregressive coefficient
                      tuple with test statistic, degrees of freedom, p-value
        surchow     : list with tuples for Chow test on regression coefficients
                      each tuple contains test value, degrees of freedom, p-value
        name_w     : string; name for the spatial weights
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
        Each equation should be listed separately. Although not required,
        in this example we will specify additional endogenous regressors.
        Equation 1 has HR80 as dependent variable, PS80 and UE80 as exogenous regressors,
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

        To run a spatial lag model, we need to specify the spatial weights matrix. 
        To do that, we can open an already existing gal file or create a new one.
        In this example, we will create a new one from NAT.shp and transform it to
        row-standardized.

        >>> w = pysal.queen_from_shapefile(pysal.examples.get_path("NAT.shp"))
        >>> w.transform='r'

        We can now run the regression and then have a summary of the output by typing:
        print(reg.summary)

        Alternatively, we can just check the betas and standard errors, asymptotic t 
        and p-value of the parameters:        

        >>> reg = SURlagIV(bigy,bigX,bigyend,bigq,w=w,name_bigy=bigyvars,name_bigX=bigXvars,name_bigyend=bigyendvars,name_bigq=bigqvars,name_ds="NAT",name_w="nat_queen")
        >>> reg.b3SLS
        {0: array([[ 6.95472387],
               [ 1.44044301],
               [-0.00771893],
               [ 3.65051153],
               [ 0.00362663]]), 1: array([[ 5.61101925],
               [ 1.38716801],
               [-0.15512029],
               [ 3.1884457 ],
               [ 0.25832185]])}

        >>> reg.tsls_inf
        {0: array([[  0.49128435,  14.15620899,   0.        ],
               [  0.11516292,  12.50787151,   0.        ],
               [  0.03204088,  -0.2409087 ,   0.80962588],
               [  0.1876025 ,  19.45875745,   0.        ],
               [  0.05450628,   0.06653605,   0.94695106]]), 1: array([[  0.44969956,  12.47726211,   0.        ],
               [  0.10440241,  13.28674277,   0.        ],
               [  0.04150243,  -3.73761961,   0.00018577],
               [  0.19133145,  16.66451427,   0.        ],
               [  0.04394024,   5.87893596,   0.        ]])}
    """

    def __init__(self,bigy,bigX,bigyend=None,bigq=None,w=None,vm=False,\
                 w_lags=1, lag_q=True, nonspat_diag=True, spat_diag=False,\
                 name_bigy=None,name_bigX=None,name_bigyend=None,\
                 name_bigq=None,name_ds=None,name_w=None):
        if w == None:
            raise Exception, "Spatial weights required for SUR-Lag"
        self.w = w
        WS = w.sparse
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        if bigyend and not(bigq):
            raise Exception, "Instruments needed when endogenous variables"
        #initialize
        self.bigy = bigy
        self.n_eq = len(self.bigy.keys())
        if name_bigy:
            self.name_bigy = name_bigy
        else: # need to construct y names
            self.name_bigy = {}
            for r in range(self.n_eq):
                yn = 'dep_var_' + str(r+1)
                self.name_bigy[r] = yn               
        self.bigX = bigX
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
        if bigyend: # check on other endogenous
            self.bigyend = bigyend
            if name_bigyend:
                self.name_bigyend = name_bigyend
            else: # need to construct names
                self.name_bigyend = {}
                for r in range(self.n_eq):
                    ky = self.bigyend[r].shape[1]
                    name_ye = ['end_' + str(i + 1) + "_" + str(r+1) for i in range(ky)]
                    self.name_bigyend[r] = name_ye
        if bigq:  # check on instruments
            self.bigq = bigq
            if name_bigq:
                self.name_bigq = name_bigq
            else: # need to construct names
                self.name_bigq = {}
                for r in range(self.n_eq):
                    ki = self.bigq[r].shape[1]
                    name_i = ['inst_' + str(i + 1) + "_" + str(r+1) for i in range(ki)]
                    self.name_bigq[r] = name_i                
                
        #spatial lag dependent variable
        bigylag = {}
        for r in range(self.n_eq):
            bigylag[r] = WS*self.bigy[r]
        if bigyend:
            self.bigyend=bigyend
            for r in range(self.n_eq):
                self.bigyend[r] = np.hstack((self.bigyend[r],bigylag[r]))
            # adjust variable names
            for r in range(self.n_eq):
                wyname = "W_" + self.name_bigy[r]
                self.name_bigyend[r].append(wyname)            
        else: # no other endogenous variables
            self.bigyend={}
            for r in range(self.n_eq):
                self.bigyend[r] = bigylag[r]
            # variable names
            self.name_bigyend = {}
            for r in range(self.n_eq):
                wyname = ["W_" + self.name_bigy[r]]
                self.name_bigyend[r] = wyname
                    
        #spatially lagged exogenous variables
        bigwx = {}
        wxnames = {}
        if w_lags == 1:
            for r in range(self.n_eq):
                bigwx[r] = WS* self.bigX[r][:,1:]
                wxnames[r] = [ "W_" + i for i in self.name_bigX[r][1:]]
            if bigq: # other instruments
                if lag_q:  # also lags for instruments
                    bigwq = {}
                    for r in range(self.n_eq):
                        bigwq = WS* self.bigq[r]
                        self.bigq[r] = np.hstack((self.bigq[r],bigwx[r],bigwq))
                        wqnames = [ "W_" + i for i in self.name_bigq[r]]
                        wxnames[r] = wxnames[r] + wqnames
                        self.name_bigq[r] = self.name_bigq[r] + wxnames[r]
                else:  # no lags for other instruments
                    for r in range(self.n_eq):
                        self.bigq[r] = np.hstack((self.bigq[r],bigwx[r]))
                        self.name_bigq[r] = self.name_bigq[r] + wxnames[r]
            else: #no other instruments only wx
                self.bigq = {}
                self.name_bigq = {}
                for r in range(self.n_eq):
                    self.bigq[r]=bigwx[r]
                    self.name_bigq[r] = wxnames[r]
        elif w_lags > 1:  # higher order lags for WX
            for r in range(self.n_eq):
                bigwxwork = WS* self.bigX[r][:,1:]
                bigwx[r] = bigwxwork
                nameswork = [ "W_" + i for i in self.name_bigX[r][1:]]
                wxnames[r] = nameswork
                for i in range(1,w_lags):
                    bigwxwork = WS*bigwxwork
                    bigwx[r] = np.hstack((bigwx[r],bigwxwork))
                    nameswork = [ "W" + i for i in nameswork ]
                    wxnames[r] = wxnames[r] + nameswork
            if bigq: # other instruments
                if lag_q: # lags for other instruments
                    wq = {}
                    wqnames = {}
                    for r in range(self.n_eq):
                        bigwq = WS* self.bigq[r]
                        wqnameswork = [ "W_" + i for i in self.name_bigq[r]]
                        wqnames[r] = wqnameswork
                        wq[r] = bigwq                        
                        for i in range(1,w_lags):
                            bigwq = WS* bigwq
                            wq[r] = np.hstack((wq[r],bigwq))
                            wqnameswork = [ "W" + i for i in wqnameswork ]
                            wqnames[r] = wqnames[r] + wqnameswork
                        self.bigq[r] = np.hstack((self.bigq[r],bigwx[r],wq[r]))
                        self.name_bigq[r] = self.name_bigq[r] + wxnames[r] + wqnames[r]
                            
                else:  # no lags for other instruments
                    for r in range(self.n_eq):
                        self.bigq[r] = np.hstack((self.bigq[r],bigwx[r]))
                        self.name_bigq[r] = self.name_bigq[r] + wxnames[r]
            else: # no other instruments only wx
                self.bigq = {}
                self.name_bigq = {}
                for r in range(self.n_eq):
                    self.bigq[r] = bigwx[r]
                    self.name_bigq[r] = wxnames[r]

        else:
            raise Exception, "Lag order must be 1 or higher"
            
        BaseThreeSLS.__init__(self,bigy=self.bigy,bigX=self.bigX,bigyend=self.bigyend,\
            bigq=self.bigq)
        
        #inference
        self.tsls_inf = sur_setp(self.b3SLS,self.varb)
        
        # test on joint significance of spatial coefficients
        if spat_diag:   
            self.joinrho = sur_joinrho(self.n_eq,self.bigK,self.b3SLS,self.varb)
        else:
            self.joinrho = None
        
        # test on constancy of coefficients across equations
        if check_k(self.bigK):   # only for equal number of variables
            self.surchow = sur_chow(self.n_eq,self.bigK,self.b3SLS,self.varb)
        else:
            self.surchow = None   
        
        #list results
        self.title = "SEEMINGLY UNRELATED REGRESSIONS (SUR) - SPATIAL LAG MODEL"                
        SUMMARY.SUR(reg=self, tsls=True, spat_diag=spat_diag, nonspat_diag=nonspat_diag)

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
    from sur_utils import sur_dictxy

    db = pysal.open(pysal.examples.get_path('NAT.dbf'), 'r')
    w = pysal.queen_from_shapefile(pysal.examples.get_path("NAT.shp"))
    w.transform='r'
    y_var0 = ['HR80','HR90']
    x_var0 = [['PS80','UE80'],['PS90','UE90']]

    bigy0,bigX0,bigyvars0,bigXvars0 = sur_dictxy(db,y_var0,x_var0)

    reg = SURlagIV(bigy0,bigX0,w=w,name_bigy=bigyvars0,name_bigX=bigXvars0,\
               name_ds="NAT",name_w="nat_queen",nonspat_diag=True,spat_diag=True)
    print reg.summary
