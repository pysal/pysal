import numpy as np

def CPC(model):
    """
    Common part of commuters based on Sorensen index
    Lenormand et al. 2012
    """
    y = model.y
    try:
        yhat = model.yhat.resahpe((-1,1))
    except:
        yhat = model.mu((-1,1))
    N = model.n 
    YYhat = np.hstack([y, yhat])
    NCC = np.sum(np.min(YYhat, axis=1))
    NCY = np.sum(Y)
    NCYhat = np.sum(yhat)
    return (N*NCC) / (NCY + NCYhat)

def sorensen(model):
    """
    Sorensen similarity index

    For use on spatial interaction models; N = sample size
    rather than N = number of locations and normalized by N instead of N**2
    """
    try:
        y = model.y.reshape((-1,1))
    except:
        y = model.f.reshape((-1,1))
    try:
        yhat = model.yhat.reshape((-1,1))
    except:
        yhat = model.mu.reshape((-1,1))
    N = model.n
    YYhat = np.hstack([y, yhat])
    num = 2.0 * np.min(YYhat, axis = 1)
    den = yhat + y
    return (1.0/N) * (np.sum(num.reshape((-1,1))/den.reshape((-1,1))))
    
def srmse(model):
    """
    Standardized root mean square error
    """
    n = model.n
    try:
        y = model.y.reshape((-1,1))
    except:
        y = model.f.reshape((-1,1))
    try:
        yhat = model.yhat.reshape((-1,1))
    except:
        yhat = model.mu.reshape((-1,1))
    srmse = ((np.sum((y-yhat)**2)/n)**.5)/(np.sum(y)/n)
    return srmse
