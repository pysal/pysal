"""
Classic (a-spatial) Econometric  methods for PySAL
"""


__author__  = "Sergio J. Rey <srey@asu.edu>"

import numpy as num
import scipy.stats as stats
import numpy.linalg as la
import pysal

def Jarque_Bera(y):
    """
    Jarque Bera test for Normality

    Parameters
    ----------
    y : array
        variable to test for normality

    Returns
    -------
    results : dict
        jb : float
            value of the statistic
        pvalue: float
            probability value for a chi-2 with 2 dof
    """

    yd=y-y.mean()
    y3=sum(yd**3)
    y2=sum(yd*2)
    n=len(y)
    s=num.sqrt(y2)
    S=(1/n)*(y3/(s**3))
    K=(1/n)*(sum(yd**4)/(s**4))
    jb=(n/6)*(S**2 + ((K-3)** 2)/4)
    pvalue=stats.chi2.pdf(jb,2)
    results={"jb":jb,'pvalue':pvalue}
    return results


class Ols:
    """
    Ordinary Least Squares Estimation.
    
    
    Attributes
    ----------
    tss : float
        Total sum of squares of dependent variable
    sig2 : float
        Estimate of error variance
    sig2ml : float
        maximum likelihood estimate of error variance
    ess : float
        error sum of squares
    dof : float
        degrees of freedom
    n   : int
        number of observations
    k   : int
        number of explanatory variables (includes constant)
    e   : array
        ols residuals
    yhat : array
        predicted values
    b   : array
        ols parameter estimates
    ixx : array
        inverse of cross-products matrix
    bvcv : array
        ols parameter estimates variance-covariance matrix
    bse : array
        ols parameter standard errors
    t   : array
        t-statistics for parameter estimates
    r2  : float
        coefficient of determination
    r2a : float
        adjusted r2

    
    """
    def __init__(self, y, X):
        """

        Parameters
        ----------
        y : array
            dependent variable
        X : array
            explanatory variables (including constant as first column)


        Examples
        --------

        >>> db=pysal.open("../examples/columbus.dbf","r")
        >>> var_names=db.header
        >>> data=num.array(db[:])
        >>> y=data[:,var_names.index("CRIME")]
        >>> X=data[:,[var_names.index(v) for v in ["INC","HOVAL"]]]
        >>> X=num.hstack((num.ones((db.n_records,1)),X))
        >>> ols=Ols(y,X)
        >>> ols.b
        array([ 68.6189611 ,  -1.59731083,  -0.27393148])
        >>> ols.t
        array([ 14.49037314,  -4.78049619,  -2.65440864])
        >>> ols.r2
        0.55240404083742323
        >>> ols.r2a
        0.53294334696078938
        >>> ols.sig2
        130.75853773444271
        """
        XT=num.transpose(X)
        xx=num.dot(XT,X)
        ixx=la.inv(xx)
        ixxx=num.dot(ixx,XT)
        b=num.dot(ixxx,y)
        yhat=num.dot(X,b)
        e=y-yhat
        n,k=X.shape
        dof=n-k
        ess=num.dot(num.transpose(e),e)
        sig2=ess/dof
        yd=y-y.mean()
        tss=num.dot(num.transpose(yd),yd)
        self.tss=tss
        self.sig2=sig2
        self.sig2ml=ess/n
        self.ess=ess
        self.dof=dof
        self.n=n
        self.k=k
        self.e=e
        self.yhat=yhat
        self.b=b
        self.ixx=ixx
        self.bvcv=sig2*ixx
        self.bse=num.sqrt(num.diag(self.bvcv))
        self.t=b/self.bse
        self.r2=1.0-ess/tss
        self.r2a=1.-(1-self.r2)*(n-1)/(n-k)


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
