"""Internal helper files for user output."""

__author__ = "Luc Anselin luc.anselin@asu.edu, David C. Folch david.folch@asu.edu, Jing Yao jingyao@asu.edu"
import textwrap as TW
import numpy as np
import copy as COPY
import diagnostics as diagnostics
import diagnostics_tsls as diagnostics_tsls
import diagnostics_sp as diagnostics_sp
import ak as AK
import pysal

__all__ = []


class DiagnosticBuilder:
    """
    Dispatch appropriate diagnostics to various regression types. This is
    generally inherited by a regression class.

    """
    def __init__(self, w, vm, instruments=False, beta_diag=True,\
                        nonspat_diag=True, spat_diag=False, lamb=False,\
                        moran=False, std_err=None, ols=False, spatial_lag=False):

        #Coefficient, Std.Error, t-Statistic, Probability 
        if beta_diag:
            self.std_err = diagnostics.se_betas(self)
            if ols:
                self.t_stat = diagnostics.t_stat(self)
                self.r2 = diagnostics.r2(self)    
                self.ar2 = diagnostics.ar2(self)   
            else:
                self.z_stat = diagnostics.t_stat(self, z_stat=True)
                self.pr2 = diagnostics_tsls.pr2_aspatial(self)
                if spatial_lag:
                    if self.predy_e != None:
                        self.pr2_e = diagnostics_tsls.pr2_spatial(self)
                    else:
                        self.pr2_e = None

        if nonspat_diag:
            if not instruments:  # quicky hack until we figure out the global nonspatial diag rules
                #general information
                self.sig2ML = self.sig2n  
                self.f_stat = diagnostics.f_stat(self)  
                self.logll = diagnostics.log_likelihood(self) 
                self.aic = diagnostics.akaike(self) 
                self.schwarz = diagnostics.schwarz(self) 
                
                #part 2: REGRESSION DIAGNOSTICS 
                self.mulColli = diagnostics.condition_index(self)
                self.jarque_bera = diagnostics.jarque_bera(self)
                
                #part 3: DIAGNOSTICS FOR HETEROSKEDASTICITY         
                self.breusch_pagan = diagnostics.breusch_pagan(self)
                self.koenker_bassett = diagnostics.koenker_bassett(self)
                self.white = diagnostics.white(self)
        
        if spat_diag:
            #part 4: spatial diagnostics
            if spat_diag:
                if instruments:
                    cache = diagnostics_sp.spDcache(self, w)
                    mi, ak, ak_p = AK.akTest(self, w, cache)
                    self.ak_test = ak, ak_p
                else:
                    lm_tests = diagnostics_sp.LMtests(self, w)
                    self.lm_error = lm_tests.lme
                    self.lm_lag = lm_tests.lml
                    self.rlm_error = lm_tests.rlme
                    self.rlm_lag = lm_tests.rlml
                    self.lm_sarma = lm_tests.sarma
                    if moran:
                        moran_res = diagnostics_sp.MoranRes(self, w, z=True)
                        self.moran_res = moran_res.I, moran_res.zI, moran_res.p_norm 

        #part 5: summary output
        if not hasattr(self, 'summary'):
            summary = summary_intro(self)
            summary += summary_r2(self, ols, spatial_lag)
            self.summary = summary
        else:
            self.summary = summary_unclose(self.summary)
            weights_text = "%-20s:%12s\n" % ('Weights matrix',self.name_w)
            break_point = self.summary.find('Dependent Variable')
            self.summary = self.summary[:break_point] + weights_text + self.summary[break_point:]
        if nonspat_diag:
            if ols:
                self.summary += summary_nonspat_diag_1(self)
        if beta_diag:
            self.summary += summary_coefs(self, instruments, lamb, std_err, ols)
        if nonspat_diag:
            if ols:
                self.summary += summary_nonspat_diag_2(self)
        if spat_diag:
            self.summary += summary_spat_diag(self, instruments, moran)
        if vm:
            self.summary += summary_vm(self, instruments)
        self.summary += summary_close()

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

def set_name_x(name_x, x):
    """Set the independent variable names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_x      : list of string
                  User provided exogenous variable names.

    x           : array
                  User provided exogenous variables.

    Returns
    -------
    
    name_x      : list of strings
                  
    """
    if not name_x:
        name_x = ['var_'+str(i+1) for i in range(len(x[0]))]
    else:
        name_x = name_x[:]
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
    if yend != None:
        if not name_yend:
            return ['endogenous_'+str(i+1) for i in range(len(yend[0]))]
        else:
            return name_yend[:]
    else:
        return []

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
    if q != None:
        if not name_q:
            return ['instrument_'+str(i+1) for i in range(len(q[0]))]
        else:
            return name_q[:]
    else:
        return []

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
    return 'W_' + name_y

def set_name_q_sp(name_x, w_lags, name_q, lag_q):
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
    if lag_q:
        names = name_x[1:] + name_q   # drop the constant
    else:
        names = name_x[1:]   # drop the constant
    sp_inst_names = []
    for j in names:
        sp_inst_names.append('W_'+j)
    if w_lags > 1:
        for i in range(2, w_lags+1):
            for j in names:
                sp_inst_names.append('W'+str(i)+'_'+j)
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

def set_robust(robust):
    """Return generic name if user passes None to the robust parameter in a
    regression. Note: already verified that the name is valid in
    check_robust() if the user passed anything besides None to robust.

    Parameters
    ----------

    robust      : string or None
                  Object passed by the user to a regression class

    Returns
    -------
    
    robust      : string
                  
    """
    if not robust:
        return 'unadjusted'
    return robust

def set_name_w(name_w, w):
    """Return generic name if user passes None to the robust parameter in a
    regression. Note: already verified that the name is valid in
    check_robust() if the user passed anything besides None to robust.

    Parameters
    ----------

    name_w      : string
                  Name passed in by user. Default is None.
    w           : W object
                  pysal W object passed in by user

    Returns
    -------
    
    name_w      : string
                  
    """
    if w != None:
        if name_w != None:
            return name_w
        else:
            return 'unknown'
    return None


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
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
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
            if i == None:
                break
            else:
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
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read()
    >>> check_weights(w, y)
    >>> # should not raise an exception

    """
    if w != None:
        if not isinstance(w, pysal.W):
            raise Exception, "w must be a pysal.W object"
        if w.n != y.shape[0]:
            raise Exception, "y must be nx1, and w must be an nxn PySAL W object"
        diag = w.sparse.diagonal()
        # check to make sure all entries equal 0
        if diag.min() != 0:
            raise Exception, "All entries on diagonal must equal 0."
        if diag.max() != 0:
            raise Exception, "All entries on diagonal must equal 0."


def check_robust(robust, wk):
    """Check if the combination of robust and wk parameters passed by the user
    are valid. Note: this does not check if the W object is a valid adaptive 
    kernel weights matrix needed for the HAC.

    Parameters
    ----------

    robust  : string or None
              Object passed by the user to a regression class
    w       : any python object
              Object passed by the user to a regression class; any type
              object can be passed

    Returns
    -------

    Returns : nothing
              Nothing is returned
              
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> wk = None
    >>> check_robust('White', wk)
    >>> # should not raise an exception

    """
    if robust:
        if robust.lower() == 'hac':
            if type(wk).__name__ != 'W' and type(wk).__name__ != 'Kernel':
                raise Exception, "HAC requires that wk be a pysal.W object"
            diag = wk.sparse.diagonal()
            # check to make sure all entries equal 1
            if diag.min() < 1.0:
                print diag.min()
                raise Exception, "All entries on diagonal of kernel weights matrix must equal 1."
            if diag.max() > 1.0:
                print diag.max()
                raise Exception, "All entries on diagonal of kernel weights matrix must equal 1."
            # ensure off-diagonal entries are in the set of real numbers [0,1)
            wegt = wk.weights
            for i in wk.id_order:
                vals = wegt[i]
                vmin = min(vals)
                vmax = max(vals)
                if vmin < 0.0:
                    raise Exception, "Off-diagonal entries must be greater than or equal to 0."
                if vmax > 1.0:
                    ##### NOTE: we are not checking for the case of exactly 1.0 #####
                    raise Exception, "Off-diagonal entries must be less than 1."
        elif robust.lower() == 'white':
            if wk:
                raise Exception, "White requires that wk be set to None"
        else:
            raise Exception, "invalid value passed to robust, see docs for valid options"

def check_spat_diag(spat_diag, w):
    """Check if there is a w parameter passed by the user if the user also
    requests spatial diagnostics.

    Parameters
    ----------

    spat_diag   : boolean
                  Value passed by a used to a regression class
    w           : any python object
                  Object passed by the user to a regression class; any type
                  object can be passed

    Returns
    -------

    Returns : nothing
              Nothing is returned
              
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read()
    >>> check_spat_diag(True, w)
    >>> # should not raise an exception

    """
    if spat_diag:
        if type(w).__name__ != 'W':
            raise Exception, "w must be a pysal.W object to run spatial diagnostics"


def check_constant(x):
    """Check if the X matrix contains a constant, raise exception if it does
    not

    Parameters
    ----------

    x           : array
                  Value passed by a used to a regression class

    Returns
    -------

    Returns : nothing
              Nothing is returned
              
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> check_constant(X)
    >>> # should not raise an exception

    """
    if not diagnostics.constant_check:
        raise Exception, "x array cannot contain a constant vector"


def summary_intro(reg):
    strSummary = ""
    strSummary += "REGRESSION\n"
    strSummary += "----------\n"
    title = "SUMMARY OF OUTPUT: " + reg.title + " ESTIMATION\n"
    strSummary += title
    strSummary += "-" * (len(title)-1) + "\n"
    strSummary += "%-20s: %12s\n" % ('Data set',reg.name_ds)
    if reg.name_w:
        strSummary += "%-20s: %12s\n" % ('Weights matrix',reg.name_w)
    strSummary += "%-20s:%12s  %-22s:%12d\n" % ('Dependent Variable',reg.name_y,'Number of Observations',reg.n)
    strSummary += "%-20s:%12.4f  %-22s:%12d\n" % ('Mean dependent var',reg.mean_y,'Number of Variables',reg.k)
    strSummary += "%-20s:%12.4f  %-22s:%12d\n" % ('S.D. dependent var',reg.std_y,'Degrees of Freedom',reg.n-reg.k)
    strSummary += '\n'
    return strSummary

def summary_coefs(reg, instruments, lamb, std_err ,ols):
    strSummary = "\n"
    if std_err:
        if std_err.lower() == 'white':
            strSummary += "White Standard Errors\n"
        elif std_err.lower() == 'hac':
            strSummary += "HAC Standard Errors; Kernel Weights: " + reg.name_gwk +"\n"
        elif std_err.lower() == 'het':
            strSummary += "Heteroskedastic Corrected Standard Errors\n"
    strSummary += "----------------------------------------------------------------------------\n"
    if ols:
        strSummary += "    Variable     Coefficient       Std.Error     t-Statistic     Probability\n"
    else:
        strSummary += "    Variable     Coefficient       Std.Error     z-Statistic     Probability\n"
    strSummary += "----------------------------------------------------------------------------\n"
    if ols:
        zt_stat = reg.t_stat
    else:
        zt_stat = reg.z_stat
    i = 0
    if instruments:
        for name in reg.name_x:        
            strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],zt_stat[i][0],zt_stat[i][1])
            i += 1
        for name in reg.name_yend:        
            strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],zt_stat[i][0],zt_stat[i][1])
            i += 1
        if lamb:
            if len(reg.betas) == len(zt_stat):
                strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % ('lambda',reg.betas[-1][0],reg.std_err[-1],zt_stat[-1][0],zt_stat[-1][1])
            else:
                strSummary += "%12s    %12.7f    \n" % ('lambda',reg.betas[-1][0])
            i += 1
    else:
        if lamb:
            for name in reg.name_x[0:-1]:        
                strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],zt_stat[i][0],zt_stat[i][1])
                i += 1
            if len(reg.betas) == len(zt_stat):
                strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % ('lambda',reg.betas[-1][0],reg.std_err[-1],zt_stat[-1][0],zt_stat[-1][1])
            else:
                strSummary += "%12s    %12.7f    \n" % ('lambda',reg.betas[-1][0])
            i += 1
        else:
            for name in reg.name_x:        
                strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],zt_stat[i][0],zt_stat[i][1])
                i += 1
    strSummary += "----------------------------------------------------------------------------\n"
    if instruments:
        insts = "Instruments: "
        for name in reg.name_q:
            insts += name + ", "
        text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
        insts = text_wrapper.fill(insts[:-2])
        strSummary += insts + "\n"
    return strSummary

def summary_r2(reg, ols, spatial_lag):
    if ols:
        strSummary = "%-20s:%12.6f\n%-20s:%12.4f\n" % ('R-squared',reg.r2,'Adjusted R-squared',reg.ar2)
    else:
        strSummary = "%-20s:%12.6f\n" % ('Pseudo R-squared',reg.pr2)
        if spatial_lag:
            if reg.pr2_e != None: 
                strSummary += "%-20s:%12.6f\n" % ('Spatial Pseudo R-squared',reg.pr2_e)
    return strSummary


def summary_nonspat_diag_1(reg):
    strSummary = ""
    strSummary += "%-20s:%12.3f  %-22s:%12.4f\n" % ('Sum squared residual',reg.utu,'F-statistic',reg.f_stat[0])
    strSummary += "%-20s:%12.3f  %-22s:%12.7g\n" % ('Sigma-square',reg.sig2,'Prob(F-statistic)',reg.f_stat[1])
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('S.E. of regression',np.sqrt(reg.sig2),'Log likelihood',reg.logll)
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('Sigma-square ML',reg.sig2ML,'Akaike info criterion',reg.aic)
    strSummary += "%-20s:%12.4f  %-22s:%12.3f\n" % ('S.E of regression ML',np.sqrt(reg.sig2ML),'Schwarz criterion',reg.schwarz)
    return strSummary
    

def summary_nonspat_diag_2(reg):
    strSummary = ""
    strSummary += "\nREGRESSION DIAGNOSTICS\n"
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
        strSummary += "\nSPECIFICATION ROBUST TEST\n"
        if len(reg.white)>3:
            strSummary += reg.white+'\n'
        else:
            strSummary += "TEST                  DF          VALUE            PROB\n"
            strSummary += "%-22s%2d       %12.6f        %9.7f\n" %('White',reg.white['df'],reg.white['wh'],reg.white['pvalue'])
    return strSummary

def summary_spat_diag(reg, instruments, moran):
    strSummary = ""
    strSummary += "\nDIAGNOSTICS FOR SPATIAL DEPENDENCE\n"
    strSummary += "TEST                          MI/DF      VALUE          PROB\n" 
    if instruments:
        strSummary += "%-22s      %2d    %12.6f       %9.7f\n" % ("Anselin-Kelejian Test", 1, reg.ak_test[0], reg.ak_test[1])
    else:
        if moran:
            strSummary += "%-22s  %12.6f %12.6f       %9.7f\n" % ("Moran's I (error)", reg.moran_res[0], reg.moran_res[1], reg.moran_res[2])
        strSummary += "%-22s      %2d    %12.6f       %9.7f\n" % ("Lagrange Multiplier (lag)", 1, reg.lm_lag[0], reg.lm_lag[1])
        strSummary += "%-22s         %2d    %12.6f       %9.7f\n" % ("Robust LM (lag)", 1, reg.rlm_lag[0], reg.rlm_lag[1])
        strSummary += "%-22s    %2d    %12.6f       %9.7f\n" % ("Lagrange Multiplier (error)", 1, reg.lm_error[0], reg.lm_error[1])
        strSummary += "%-22s         %2d    %12.6f       %9.7f\n" % ("Robust LM (error)", 1, reg.rlm_error[0], reg.rlm_error[1])
        strSummary += "%-22s    %2d    %12.6f       %9.7f\n\n" % ("Lagrange Multiplier (SARMA)", 2, reg.lm_sarma[0], reg.lm_sarma[1])
    return strSummary

def summary_vm(reg, instruments):
    strVM = "\n"
    strVM += "COEFFICIENTS VARIANCE MATRIX\n"
    strVM += "----------------------------\n"
    if instruments:
        for name in reg.name_z:
            strVM += "%12s" % (name)
    else:
        for name in reg.name_x:
            strVM += "%12s" % (name)
    strVM += "\n"
    nrow = reg.vm.shape[0]
    ncol = reg.vm.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            strVM += "%12.6f" % (reg.vm[i][j]) 
        strVM += "\n"
    return strVM

def summary_pred(reg):
    strPred = "\n\n"
    strPred += "%16s%16s%16s%16s\n" % ('OBS',reg.name_y,'PREDICTED','RESIDUAL')
    for i in range(reg.n):
        strPred += "%16d%16.5f%16.5f%16.5f\n" % (i+1,reg.y[i][0],reg.predy[i][0],reg.u[i][0])
    return strPred
            
def summary_close():
    return "========================= END OF REPORT =============================="
    
def summary_unclose(summary):
    return summary[:-70]


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



