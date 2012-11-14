"""
Maximum likelihood estimation of spatial process models

"""

__author__ = "Serge Rey srey@asu.edu, Luc Anselin luc.anselin@asu.edu"

import numpy as np
import pysal as ps

SMALL = 500

def defl_lag(r, w, e1, e2, evals=None):
    """
    Derivative of the likelihood for the lag model

    Parameters
    ----------

    r: estimate of rho

    w: spatial weights object

    e1: ols residuals of X on y

    e2: ols residuals of X on Wy


    Returns
    -------

    tuple: 
        tuple[0]: dfl (scalar) value of derivative at parameter values
        tuple[1]: tr (scalar) trace of  (I-rW)^{-1} W

    """
    n = w.n
    e1re2 = e1 - r * e2
    num = np.dot(e1re2.T, e2)
    den = np.dot(e1re2.T, e1re2)
    a = -r * w.full()[0]
    np.fill_diagonal(a, np.ones((w.n,1)))
    a = np.linalg.inv(a)
    tr = np.trace(np.dot(a, w.full()[0]))
    dfl = n * (num/den) - tr
    return dfl[0][0], tr


    

def _logJacobian(w, rho):
    """
    Log determinant of I-\rho W

    Brute force initially
    """
    return np.linalg.det(np.eye(w.n) - rho * w.full()[0])



def ML_Lag(y, w, X, precrit=0.0000001, verbose=False):
    """
    Maximum likelihood estimation of spatial lag model
    """

    # step 1 OLS of X on y yields b1
    d = np.linalg.inv(np.dot(X.T, X))
    b1 = np.dot(d, np.dot(X.T, y))

    # step 2 OLS of X on Wy: yields b2
    wy = ps.lag_spatial(w,y)
    b2 = np.dot(d, np.dot(X.T, wy))

    # step 3 compute residuals e1, e2
    e1 = y - np.dot(X,b1)
    e2 = wy - np.dot(X,b2)

    # step 4 given e1, e2 find rho that maximizes Lc

    # ols estimate of rho
    XA = np.hstack((wy,X))
    bols = np.dot(np.linalg.inv(np.dot(XA.T, XA)), np.dot(XA.T,y))
    rols = bols[0][0]

    while np.abs(rols) > 1.0:
        rols = rols/2.0

    if rols > 0.0:
        r1 = rols
        r2 = r1 / 5.0
    else:
        r2 = rols
        r1 = r2 / 5.0
    #print r1, r2

    df1 = 0
    df2 = 0
    tr = 0
    df1, tr = defl_lag(r1, w, e1, e2)
    df2, tr = defl_lag(r2, w, e1, e2)

    if df1*df2 <= 0:
        ll = r2
        ul = r1
    elif df1 >= 0.0 and df1 >= df2:
        ll = -0.999
        ul = r2
        df1 = df2
        df2 = -(10.0**10)
    elif df1 >= 0.0 and df1 < df2:
        ll = r1
        ul = 0.999
        df2 = df1
        df1 = -(10.0**10)
    elif df1 < 0.0 and df1 >= df2:
        ul = 0.999
        ll = r1
        df2 = df1
        df1 = 10.0**10
    else:
        ul = r2
        ll = -0.999
        df1 = df2
        df2 = 10.0**10

    # main bisection iteration

    err = 10
    t = rols
    ro = (ll+ul) / 2.
    if verbose:
        line ="%-5s\t%12s\t%12s\t%12s\t%12s"%("Iter.","LL","RHO","UL","DFR")
        print line

    i = 0
    while err > precrit:
        if verbose:
            print "%d\t%12.8f\t%12.8f\t%12.8f\t%12.8f"  % (i,ll, ro, ul, df1)
        dfr, tr = defl_lag(ro, w, e1, e2)
        if dfr*df1 < 0.0:
            ll = ro
        else:
            ul = ro
            df1 = dfr
        err = np.abs(t-ro)
        t = ro
        ro =(ul+ll)/2.
        i += 1
    ro = t
    tr1 = tr
    bml = b1 - (ro * b2)
    b = [ro,bml]

    return (b, tr1)




if __name__ == '__main__':

    db =  ps.open(ps.examples.get_path('columbus.dbf'),'r')
    y = np.array(db.by_col("CRIME"))
    y.shape = (len(y),1)
    X = []
    X.append(db.by_col("INC"))
    X.append(db.by_col("HOVAL"))
    X = np.array(X).T
    w = ps.open(ps.examples.get_path("columbus.gal")).read()
    w.transform = 'r'
    X = np.hstack((np.ones((w.n,1)),X))
    bml = ML_Lag(y,w,X, verbose=True)
    bml = ML_Lag(y,w,X, verbose=True, precrit=10.0**-10)
