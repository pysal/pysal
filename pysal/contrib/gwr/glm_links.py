import numpy as np

def get_y_hat(family, v, y_offset, y_fix):
    """
    get y_hat
    """
    if family == 'Gaussian':
        return v + y_fix
    if family == 'Poisson':
        return np.exp(v + y_fix) * y_offset
    if family == 'logistic':
        return 1.0/(1 + np.exp(-1.0*v - y_fix))

def link_g(v, y, y_offset, y_fix):
    """
    link function for Gaussian model

    Method: p189, Table(8.1)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Arguments:
        v              : array
                         n*1, v =  Beta * X
        y              : array
                         n*1, dependent variable
        y_offset:      : array
                         n*1, for Poisson model
        y_fix          : array
                         n*1, the fix intercept value of y

    Return:
           z           : array
                         n*1, adjusted dependent variable
           w           : array
                         n*1, weight to multiple with x
    """
    n = len(y)
    w = np.ones(shape=(n,1))
    z = y - y_fix

    return z, w

def link_p(v, y, y_offset, y_fix):
    """
    link function for Poisson model

    Method: p189, Table(8.1)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Arguments:
        v              : array
                         n*1, v =  Beta * X
        y              : array
                         n*1, dependent variable
        y_offset:      : array
                         n*1, for Poisson model
        y_fix          : array
                         n*1, the fix intercept value of y

    Return:
           z           : array
                         n*1, adjusted dependent variable
           w           : array
                         n*1, weight to multiple with x
    """
    y_hat = get_y_hat('Poisson', v, y_offset, y_fix)
    y_hat = y_hat
    w = y_hat
    z = v + y_fix +(y-y_hat)/y_hat

    return z, w

def link_l(v, y, y_offset, y_fix):
    """
    link function for Logistic model

    Method: p189, Table(8.1)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Arguments:
        v              : array
                         n*1, v =  Beta * X
        y              : array
                         n*1, dependent variable
        y_offset:      : array
                         n*1, for Poisson model
        y_fix          : array
                         n*1, the fix intercept value of y

    Return:
           z           : array
                         n*1, adjusted dependent variable
           w           : array
                         n*1, weight to multiple with x
    """
    y_hat = get_y_hat('logistic', v, y_offset, y_fix)
    deriv = y_hat * (1.0 - y_hat)
    n = len(y)
    for i in range(n):
        if (deriv[i] < .01):
            deriv[i] = .01
    z = v + y_fix + (y - y_hat) / deriv
    w = deriv

    return z, w

