"""Internal helper files for user output."""

__author__ = "Luc Anselin luc.anselin@asu.edu, David C. Folch david.folch@asu.edu, Jing Yao jingyao@asu.edu"
import textwrap as TW
import numpy as np
import copy
import diagnostics
import diagnostics_sp

__all__ = []


class DiagnosticBuilder:
    """
    Dispatch appropriate diagnostics to various regression types. This is is
    generally inherited by a regression class.

    Parameters
    ----------

    x           : array
                  Array of independent variables
    constant    : boolean
                  Boolean indicating if the regression class automatically
                  inserted a constant as an independent variable
    w           : pysal spatial weights object
                  This triggers if spatial diagnostics are run
    vm          : boolean
                  If True then include the variance-covariance matrix in the
                  output
    pred        : boolean
                  If True then include the predicted values in the output
    instruments : boolean
                  If True then the class assumes the regression is some form
                  of 2SLS

    Attributes
    ----------
    r2       : float
               R squared
    ar2      : float
               Adjusted R squared
    utu      : float
               Sum of the squared residuals
    sig2     : float
               Sigma squared
    sig2ML   : float
               Sigma squared ML 
    f_stat   : tuple
               Statistic (float), p-value (float)
    logll    : float
               Log likelihood        
    aic      : float
               Akaike info criterion 
    schwarz  : float
               Schwarz criterion     
    std_err  : array
               1xk array of Std.Error    
    t_stat   : list of tuples
               Each tuple contains the pair (statistic, p-value), where each is
               a float; same order as self.x
    mulColli : float
               Multicollinearity condition number
    jarque_bera : dictionary
               'jb': Jarque-Bera statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    breusch_pagan : dictionary
               'bp': Breusch-Pagan statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    koenker_bassett : dictionary
               'kb': Koenker-Bassett statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    white    : dictionary
               'wh': White statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    lm_error : tuple
               Lagrange multiplier test for spatial error model; each tuple contains
               the pair (statistic, p-value), where each is a float; only available 
               if w defined
    lm_lag   : tuple
               Lagrange multiplier test for spatial lag model; each tuple contains
               the pair (statistic, p-value), where each is a float; only available 
               if w defined
    rlm_error : tuple
               Robust lagrange multiplier test for spatial error model; each tuple 
               contains the pair (statistic, p-value), where each is a float; only 
               available if w defined
    rlm_lag   : tuple
               Robust lagrange multiplier test for spatial lag model; each tuple 
               contains the pair (statistic, p-value), where each is a float; only 
               available if w defined
    lm_sarma : tuple
               Lagrange multiplier test for spatial SARMA model; each tuple contains
               the pair (statistic, p-value), where each is a float; only available 
               if w defined
    moran_res : tuple
                Tuple containing the triple (Moran's I, stansardized Moran's
                I, p-value); only available if w defined
    summary  : string
               Including all the information in OLS class in nice format          

    """
    def __init__(self, x, constant, w, vm, pred, instruments=False):
        #general information
        self.r2 = diagnostics.r2(self)    
        self.ar2 = diagnostics.ar2(self)   
        self.sigML = self.sig2n  
        self.f_stat = diagnostics.f_stat(self)  
        self.logll = diagnostics.log_likelihood(self) 
        self.aic = diagnostics.akaike(self) 
        self.schwarz = diagnostics.schwarz(self) 
        
        #Coefficient, Std.Error, t-Statistic, Probability 
        self.std_err = diagnostics.se_betas(self)
        if instruments:
            self.z_stat = diagnostics.t_stat(self, z_stat=True)
        else:
            self.t_stat = diagnostics.t_stat(self)
        
        #part 2: REGRESSION DIAGNOSTICS 
        if instruments:
            self.mulColli = None
        else:
            self.mulColli = diagnostics.condition_index(self)
        self.jarque_bera = diagnostics.jarque_bera(self)
        
        #part 3: DIAGNOSTICS FOR HETEROSKEDASTICITY         
        self.breusch_pagan = diagnostics.breusch_pagan(self)
        self.koenker_bassett = diagnostics.koenker_bassett(self)
        if instruments:
            self.white = None
        else:
            self.white = diagnostics.white(self)
        
        #part 4: spatial diagnostics
        if w:
            if instruments:
                pass
            else:
                lm_tests = diagnostics_sp.LMtests(self, w)
                self.lm_error = lm_tests.lme
                self.lm_lag = lm_tests.lml
                self.rlm_error = lm_tests.rlme
                self.rlm_lag = lm_tests.rlml
                self.lm_sarma = lm_tests.sarma
                moran_res = diagnostics_sp.MoranRes(self, w, z=True)
                self.moran_res = moran_res.I, moran_res.zI, moran_res.p_norm 

        #part 5: summary output
        self.summary = summary_results(self, w, vm, pred, instruments)

def set_name_ds(name_ds):
    """Set the dataset name in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_ds     : string
                  User provided dataset name.

    Returns
    -------
    
    name_ds     : string
                  
    """
    if not name_ds:
        name_ds = 'unknown'
    return name_ds

def set_name_y(name_y):
    """Set the dataset name in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_ds     : string
                  User provided dataset name.

    Returns
    -------
    
    name_ds     : string
                  
    """
    if not name_y:
        name_y = 'dep_var'
    return name_y

def set_name_x(name_x, x, constant):
    """Set the independent variable names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_x      : list of string
                  User provided exogenous variable names.

    Returns
    -------
    
    name_x      : list of strings
                  
    """
    if not name_x:
        name_x = ['var_'+str(i+1) for i in range(len(x[0]))]
    if constant:
        name_x.insert(0, 'CONSTANT')
    return name_x
    
def set_name_yend(name_yend, yend):
    """Set the endogenous variable names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_yend   : list of strings
                  User provided exogenous variable names.

    Returns
    -------
    
    name_yend   : list of strings
                  
    """
    if not name_yend:
        name_yend = ['endogenous_'+str(i+1) for i in range(len(yend[0]))]
    return name_yend
    
def set_name_q(name_q, q):
    """Set the external instrument names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_q      : string
                  User provided instrument names.
    q           : array
                  Array of instruments

    Returns
    -------
    
    name_q      : list of strings
                  
    """
    if not name_q:
        name_q = ['instrument_'+str(i+1) for i in range(len(q[0]))]
    return name_q

def set_name_yend_sp(name_y):
    """Set the spatial lag name in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_y      : string
                  User provided dependent variable name.

    Returns
    -------
    
    name_yend_sp : string
                  
    """
    return 'lag_' + name_y

def set_name_q_sp(name_x, w_lags):
    """Set the spatial instrument names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_x      : list of strings
                  User provided exogenous variable names.

    w_lags      : int
                  User provided number of spatial instruments lags

    Returns
    -------
    
    name_q_sp   : list of strings
                  
    """
    sp_inst_names = []
    for i in range(w_lags):
        for j in name_x:
            sp_inst_names.append('lag'+str(i+1)+'_'+j)
    return sp_inst_names
    
def set_name_h(name_x, name_q):
    """Set the full instruments names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_x      : list of strings
                  User provided exogenous variable names.
    name_q      : list of strings
                  User provided instrument variable names.

    Returns
    -------
    
    name_h      : list of strings
                  
    """
    return name_x + name_q

def summary_results(reg, w, vm, pred, instruments):
    """
    nice output for regressions
    
    Parameters
    ----------

    reg     : regression object
              output instance from a regression model

    vm      : boolean
              if True, print out variance matrix

    pred    : boolean
              if True, print out y, predicted values and residuals
    
    Returns
    ----------

    strSummary   : string
                   formatted information from regression class

    """     
    strSummary = ""
    
    # general information 1
    strSummary += "REGRESSION\n"
    strSummary += "----------\n"
    title = "SUMMARY OF OUTPUT: " + reg.title + " ESTIMATION\n"
    strSummary += title
    strSummary += "-" * (len(title)-1) + "\n"
    strSummary += "%-20s:%12s\n" % ('Data set',reg.name_ds)
    strSummary += "%-20s:%12s  %-22s:%12d\n" % ('Dependent Variable',reg.name_y,'Number of Observations',reg.n)
    strSummary += "%-20s:%12.4f  %-22s:%12d\n" % ('Mean dependent var',reg.mean_y,'Number of Variables',reg.k)
    strSummary += "%-20s:%12.4f  %-22s:%12d\n" % ('S.D. dependent var',reg.std_y,'Degrees of Freedom',reg.n-reg.k)
    strSummary += '\n'

    # general information 2
    strSummary += "%-20s:%12.6f  %-22s:%12.4f\n" % ('R-squared',reg.r2,'F-statistic',reg.f_stat[0])
    strSummary += "%-20s:%12.6f  %-22s:%12.8g\n" % ('Adjusted R-squared',reg.ar2,'Prob(F-statistic)',reg.f_stat[1])
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('Sum squared residual',reg.utu,'Log likelihood',reg.logll)
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('Sigma-square',reg.sig2,'Akaike info criterion',reg.aic)
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('S.E. of regression',np.sqrt(reg.sig2),'Schwarz criterion',reg.schwarz)
    strSummary += "%-20s:%12.3f\n%-20s:%12.4f\n" % ('Sigma-square ML',reg.sigML,'S.E of regression ML',np.sqrt(reg.sigML))
    strSummary += '\n'
    
    # Variable    Coefficient     Std.Error    t-Statistic   Probability 
    strSummary += "----------------------------------------------------------------------------\n"
    if instruments:
        strSummary += "    Variable     Coefficient       Std.Error     z-Statistic     Probability\n"
    else:
        strSummary += "    Variable     Coefficient       Std.Error     t-Statistic     Probability\n"
    strSummary += "----------------------------------------------------------------------------\n"
    i = 0
    if instruments:
        for name in reg.name_x:        
            strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],reg.z_stat[i][0],reg.z_stat[i][1])
            i += 1
        for name in reg.name_yend:        
            strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],reg.z_stat[i][0],reg.z_stat[i][1])
            i += 1
        strSummary += "----------------------------------------------------------------------------\n"
        insts = "Instruments: "
        for name in reg.name_h:
            insts += name + ", "
        text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
        insts = text_wrapper.fill(insts[:-2])
        strSummary += insts + "\n"
    else:
        for name in reg.name_x:        
            strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],reg.t_stat[i][0],reg.t_stat[i][1])
            i += 1
        strSummary += "----------------------------------------------------------------------------\n"
    
    # diagonostics
    strSummary += "\n\nREGRESSION DIAGNOSTICS\n"
    if reg.mulColli:
        strSummary += "MULTICOLLINEARITY CONDITION NUMBER%12.6f\n" % (reg.mulColli)
    strSummary += "TEST ON NORMALITY OF ERRORS\n"
    strSummary += "TEST                  DF          VALUE            PROB\n"
    strSummary += "%-22s%2d       %12.6f        %9.7f\n\n" % ('Jarque-Bera',reg.jarque_bera['df'],reg.jarque_bera['jb'],reg.jarque_bera['pvalue'])
    strSummary += "DIAGNOSTICS FOR HETEROSKEDASTICITY\n"
    strSummary += "RANDOM COEFFICIENTS\n"
    strSummary += "TEST                  DF          VALUE            PROB\n"
    strSummary += "%-22s%2d       %12.6f        %9.7f\n" % ('Breusch-Pagan test',reg.breusch_pagan['df'],reg.breusch_pagan['bp'],reg.breusch_pagan['pvalue'])
    strSummary += "%-22s%2d       %12.6f        %9.7f\n" % ('Koenker-Bassett test',reg.koenker_bassett['df'],reg.koenker_bassett['kb'],reg.koenker_bassett['pvalue'])
    if reg.white:
        strSummary += "SPECIFICATION ROBUST TEST\n"
        strSummary += "TEST                  DF          VALUE            PROB\n"
        strSummary += "%-22s%2d       %12.6f        %9.7f\n\n" % ('White',reg.white['df'],reg.white['wh'],reg.white['pvalue'])

    # spatial diagonostics
    if w and not instruments:   # OLS diagnostics
        strSummary += "DIAGNOSTICS FOR SPATIAL DEPENDENCE\n"
        strSummary += "TEST                          MI/DF      VALUE          PROB\n" 
        strSummary += "%-22s  %12.6f %12.6f       %9.7f\n" % ("Moran's I (error)", reg.moran_res[0], reg.moran_res[1], reg.moran_res[2])
        strSummary += "%-22s      %2d    %12.6f       %9.7f\n" % ("Lagrange Multiplier (lag)", 1, reg.lm_lag[0], reg.lm_lag[1])
        strSummary += "%-22s         %2d    %12.6f       %9.7f\n" % ("Robust LM (lag)", 1, reg.rlm_lag[0], reg.rlm_lag[1])
        strSummary += "%-22s    %2d    %12.6f       %9.7f\n" % ("Lagrange Multiplier (error)", 1, reg.lm_error[0], reg.lm_error[1])
        strSummary += "%-22s         %2d    %12.6f       %9.7f\n" % ("Robust LM (error)", 1, reg.rlm_error[0], reg.rlm_error[1])
        strSummary += "%-22s    %2d    %12.6f       %9.7f\n\n" % ("Lagrange Multiplier (SARMA)", 2, reg.lm_sarma[0], reg.lm_sarma[1])
    if w and instruments:       # 2SLS diagnostics
        pass 

    # variance matrix
    if vm:
        strVM = ""
        strVM += "COEFFICIENTS VARIANCE MATRIX\n"
        strVM += "----------------------------\n"
        strVM += "%12s" % ('CONSTANT')
        for name in reg.name_x:
            strVM += "%12s" % (name)
        strVM += "\n"
        nrow = reg.vm.shape[0]
        ncol = reg.vm.shape[1]
        for i in range(nrow):
            for j in range(ncol):
                strVM += "%12.6f" % (reg.vm[i][j]) 
            strVM += "\n"
        strSummary += strVM
        
    # y, PREDICTED, RESIDUAL 
    if pred:
        strPred = "\n\n"
        strPred += "%16s%16s%16s%16s\n" % ('OBS',reg.name_y,'PREDICTED','RESIDUAL')
        for i in range(reg.n):
            strPred += "%16d%16.5f%16.5f%16.5f\n" % (i+1,reg.y[i][0],reg.predy[i][0],reg.u[i][0])
        strSummary += strPred
            
    # end of report
    strSummary += "========================= END OF REPORT =============================="
        
    return strSummary



def check_arrays(*arrays):
    """Check if the objects passed by a user to a regression class are
    correctly structured. If the user's data is correctly formed this function
    returns nothing, if not then an exception is raised. Note, this does not 
    check for model setup, simply the shape and types of the objects.

    Parameters
    ----------

    *arrays : anything
              Objects passed by the user to a regression class; any type
              object can be passed and any number of objects can be passed
     
    Returns
    -------

    Returns : nothing
              Nothing is returned

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("../examples/columbus.dbf","r")
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> check_arrays(y, X)
    >>> # should not raise an exception

    """
    rows = []
    for i in arrays:
        if not issubclass(type(i), np.ndarray):
            raise Exception, "all input data must be numpy arrays"
        shape = i.shape
        if len(shape) > 2:
            raise Exception, "all input arrays must have exactly two dimensions"
        if len(shape) == 1:
            raise Exception, "all input arrays must have exactly two dimensions"
        if shape[0] < shape[1]:
            raise Exception, "one or more input arrays have more columns than rows"
        rows.append(shape[0])
    if len(set(rows)) > 1:
        raise Exception, "arrays not all of same length"

def check_weights(w, y):
    """Check if the w parameter passed by the user is a pysal.W object and
    check that its dimensionality matches the y parameter.  Note that this
    check is not performed if w set to None.

    Parameters
    ----------

    w       : any python object
              Object passed by the user to a regression class; any type
              object can be passed
    y       : numpy array
              Any shape numpy array can be passed. Note: if y passed
              check_arrays, then it will be valid for this function

    Returns
    -------

    Returns : nothing
              Nothing is returned
              
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("../examples/columbus.dbf","r")
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> w = pysal.open("../examples/columbus.gal", 'r').read()
    >>> check_weights(w, y)
    >>> # should not raise an exception

    """
    if w:
        if type(w).__name__ != 'W':
            raise Exception, "w must be a pysal.W object"
        if w.n != y.shape[0]:
            raise Exception, "y must be nx1, and w must be an nxn PySAL W object"



def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



