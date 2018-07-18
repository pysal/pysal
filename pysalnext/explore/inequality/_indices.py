'''
Diversity indices as suggested in Nijkamp & Poot (2013) [1] and Nijkamp & Poot
(2015) [2]

References
----------

[1]_ Nijkamp & Poot
[2]_ Nijkamp, P. and Poot, J. "Cultural Diversity: A Matter of Measurement".
     IZA Discussion Paper Series No. 8782
'''

import itertools
import numpy as np

SMALL = np.finfo('float').tiny

def abundance(x):
    '''
    Abundance index
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Abundance index
    '''
    xs = x.sum(axis=0)
    return np.sum([1 for i in xs if i>0])

def margalev_md(x):
    '''
    Margalev MD index
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Margalev MD index
    '''
    a = abundance(x)
    return (a - 1.) / np.log(x.sum())

def menhinick_mi(x):
    '''
    Menhinick MI index
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Menhinick MI index
    '''
    a = abundance(x)
    return (a - 1.) / np.sqrt(x.sum())

def simpson_so(x):
    '''
    Simpson diversity index SO
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Simpson diversity index SO
    '''
    xs0 = x.sum(axis=0)
    xs = x.sum()
    num = (xs0 * (xs0 - 1.)).sum()
    den = xs * (xs - 1.)
    return num / den

def simpson_sd(x):
    '''
    Simpson diversity index SD
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Simpson diversity index SD
    '''
    return 1. - simpson_so(x)

def herfindahl_hd(x):
    '''
    Herfindahl index HD
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Herfindahl index HD
    '''
    pgs = x.sum(axis=0)
    p = pgs.sum()
    return ((pgs * 1. / p)**2).sum()

def theil_th(x, ridz=True):
    '''
    Theil index TH as expressed in equation (32) of [2]
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    ridz    : boolean
              [Optional. Default=True] Flag to add a small amount to zero
              values to avoid zero division problems
    Returns
    -------
    a       : float
              Theil index TH
    '''
    if ridz:
        x = x + SMALL * (x == 0)  # can't have 0 values
    pa = x.sum(axis=1).astype(float) # Area totals
    pg = x.sum(axis=0).astype(float) # Group totals
    p = pa.sum()
    num = (x / pa[:, None]) * ( np.log(pg / p) - np.log(x / pa[:, None]) )
    den = ( (pg / p) * np.log(pg / p) ).sum()
    th = (pa / p)[:, None] * (num / den)
    return th.sum().sum()

def theil_th_brute(x, ridz=True):
    '''
    Theil index TH using inefficient computation

    NOTE: just for result comparison, it matches `theil_th`
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    ridz    : boolean
              [Optional. Default=True] Flag to add a small amount to zero
              values to avoid zero division problems
    Returns
    -------
    a       : float
              Theil index TH
    '''
    if ridz:
        x = x + SMALL * (x == 0)  # can't have 0 values
    pas = x.sum(axis=1).astype(float) # Area totals
    pgs = x.sum(axis=0).astype(float) # Group totals
    p = pas.sum()
    th = np.zeros(x.shape)
    for g in np.arange(x.shape[1]):
        pg = pgs[g]
        for a in np.arange(x.shape[0]):
            pa = pas[a]
            pga = x[a, g]
            num = (pga / pa) * ((np.log(pg/p)) - np.log(pga/pa))
            den = ((pgs / p) * np.log(pgs / p) ).sum()
            th[a, g] = (pa / p) * (num / den)
    return th.sum().sum()

def fractionalization_gs(x):
    '''
    Fractionalization Gini-Simpson index GS
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Fractionalization Gini-Simpson index GS
    '''
    return 1. - herfindahl_hd(x)

def polarization(x):
    return 'Not implemented'

def shannon_se(x):
    '''
    Shannon index SE
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Shannon index SE
    '''
    pgs = x.sum(axis=0)
    p = pgs.sum()
    ratios = pgs * 1. / p
    return - (ratios * np.log(ratios)).sum()

def gini_gi(x):
    '''
    Gini GI index

    NOTE: based on 3rd eq. of "Calculation" in:

            http://en.wikipedia.org/wiki/Gini_coefficient

         Returns same value as `gini` method in the R package `reldist` (see
         http://rss.acs.unt.edu/Rdoc/library/reldist/html/gini.html) if every
         category has at least one observation
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Gini GI index
    '''
    ys = x.sum(axis=0)
    return _gini(ys)

def gini_gi_m(x):
    '''
    Gini GI index (equivalent to `gini_gi`, not vectorized)

    NOTE: based on Wolfram Mathworld formula in:

            http://mathworld.wolfram.com/GiniCoefficient.html

         Returns same value as `gini_gi`.
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Gini GI index
    '''
    xs = x.sum(axis=0)
    num = np.sum([np.abs(xi - xj) for xi, xj in itertools.permutations(xs, 2)])
    den = 2. * xs.shape[0]**2 * np.mean(xs)
    return num / den

def _gini(ys):
    '''
    Gini for a single row to be used both by `gini_gi` and `gini_gig`
    '''
    n = ys.flatten().shape[0]
    ys.sort()
    num = 2. * ((np.arange(n)+1) * ys).sum()
    den = n * ys.sum()
    return (num / den) - ((n + 1.) / n)

def hoover_hi(x):
    '''
    Hoover index HI

    NOTE: based on

            http://en.wikipedia.org/wiki/Hoover_index

    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : float
              Hoover HI index
    '''
    es = x.sum(axis=0)
    e_total = es.sum()
    a_total = es.shape[0]
    s = np.abs((es*1./e_total) - (1./a_total)).sum()
    return s / 2.

def similarity_w_wd(x, tau):
    '''
    Similarity weighted diversity
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    tau     : array
              k x k array where tau_ij represents dissimilarity between group
              i and group j. Diagonal elements are assumed to be one.

    Returns
    -------
    a       : float
              Similarity weighted diversity index
    '''
    pgs = x.sum(axis=0)
    pgs = pgs * 1. / pgs.sum()
    s = sum([pgs[i] * pgs[j] * tau[i, j] for i,j in \
            itertools.product(np.arange(pgs.shape[0]), repeat=2)])
    return 1. - s

def segregation_gsg(x):
    '''
    Segregation index GS

    This is a Duncan&Duncan index of a group against the rest combined
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : array
              Array with GSg indices for the k groups
    '''
    pgs = x.sum(axis=0)
    pas = x.sum(axis=1)
    p = pgs.sum()
    first = (x.T * 1. / pgs[:, None]).T
    paMpga = pas[:, None] - x
    pMpg = p - pgs
    second = paMpga * 1. / pMpg[None, :]
    return 0.5 * (np.abs(first - second)).sum(axis=0)

def modified_segregation_msg(x):
    '''
    Modified segregation index GS

    This is a modified version of GSg index as used by Van Mourik et al. (1989)
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : array
              Array with MSg indices for the k groups
    '''
    pgs = x.sum(axis=0)
    p = pgs.sum()
    ms_inds = segregation_gsg(x) # To be updated in loop below
    for gi in np.arange(x.shape[1]):
        pg = pgs[gi]
        pgp = pg * 1. / p
        ms_inds[gi] = 2. * pgp * (1. - pgp) * ms_inds[gi]
    return ms_inds

def isolation_isg(x):
    '''
    Isolation index IS

    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : array
              Array with ISg indices for the k groups
    '''
    ws = x * 1. / x.sum(axis=0)
    pgapa = (x.T * 1. / x.sum(axis=1)).T
    pgp = x.sum(axis=0) * 1. / x.sum()
    return (ws * pgapa / pgp).sum(axis=0)

def isolation_ii(x):
    '''
    Isolation index II_g as in equation (23) of [2].

    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : array
              Array with IIg indices for the k groups
    '''
    pa = x.sum(axis=1).astype(float) # Area totals
    pg = x.sum(axis=0).astype(float) # Group totals
    p = pa.sum()
    ws = x / pg

    block = ( ws * (x / pa[:, None]) ).sum(axis=0)
    num = ( block / (pg / p) ) - (pg / p)
    den = 1. - (pg / p)
    return num / den

def gini_gig(x):
    '''
    Gini GI index

    NOTE: based on Wolfram Mathworld formula in:

            http://mathworld.wolfram.com/GiniCoefficient.html

         Returns same value as `gini_gi`.
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : array
              Gini GI index for every group k
    '''
    return np.apply_along_axis(_gini, 0, x)

def ellison_glaeser_egg(x, hs=None):
    '''
    Ellison and Glaeser (1997) [1]_ index of concentration. Implemented as in
    equation (5) of original reference
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per area) and k columns
              (one per industry). Each cell indicates employment figures for
              area n and industry k
    hs      : array
              [Optional] Array of dimension (k,) containing the Herfindahl
              indices of each industry's plant sizes. If not passed, it is
              assumed every plant contains one and only one worker and thus
              H_k = 1 / P_k, where P_k is the total employment in k

    Returns
    -------
    a       : array
              EG index for every group k

    References
    ----------

    .. [1] Ellison, G. and Glaeser, E. L. "Geographic Concentration in U.S.
    Manufacturing Industries: A Dartboard Approach". Journal of Political
    Economy. 105: 889-927

    '''
    industry_totals = x.sum(axis=0)
    if hs==None:
        hs = 1. / industry_totals
    xs = x.sum(axis=1) * 1. / x.sum()
    part = 1. - (xs**2).sum()
    eg_inds = np.zeros(x.shape[1])
    for gi in np.arange(x.shape[1]):
        ss = x[:, gi] * 1. / industry_totals[gi]
        g = ((ss - xs)**2).sum()
        h = hs[gi]
        eg_inds[gi] = (g - part * h) / (part * (1. - h))
    return eg_inds

def ellison_glaeser_egg_pop(x):
    '''
    Ellison and Glaeser (1997) [1]_ index of concentration. Implemented to be
    computed with data about people (segregation/diversity) rather than as
    industry concentration, following Mare et al (2012) [2]_
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : array
              EG index for every group k

    References
    ----------

    .. [1] Ellison, G. and Glaeser, E. L. "Geographic Concentration in U.S.
    Manufacturing Industries: A Dartboard Approach". Journal of Political
    Economy. 105: 889-927

    .. [2] Mare, D., Pinkerton, R., Poot, J. and Coleman, A. (2012)
    Residential Sorting Across Auckland Neighbourhoods. Mimeo. Wellington:
    Motu Economic and Public Policy Research.

    '''
    pas = x.sum(axis=1)
    pgs = x.sum(axis=0)
    p = pas.sum()
    pap = pas * 1. / p
    opg = 1./ pgs
    oopg = 1. - opg
    eg_inds = np.zeros(x.shape[1])
    for g in np.arange(x.shape[1]):
        pgas = x[:, g]
        pg = pgs[g]
        num1n = (((pgas * 1. / pg) - (pas * 1. / p))**2).sum()
        num1d = 1. - ((pas * 1. / p)**2).sum()
        num2 = opg[g]
        den = oopg[g]
        eg_inds[g] = ((num1n / num1d) - num2) / den
    return eg_inds

def maurel_sedillot_msg(x, hs=None):
    '''
    Maurel and Sedillot (1999) [1]_ index of concentration. Implemented as in
    equation (7) of original reference
    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    hs      : array
              [Optional] Array of dimension (k,) containing the Herfindahl
              indices of each industry's plant sizes. If not passed, it is
              assumed every plant contains one and only one worker and thus
              H_k = 1 / P_k, where P_k is the total employment in k

    Returns
    -------
    a       : array
              MS index for every group k

    References
    ----------

    .. [1] Maurel, F. and Sedillot, B. (1999). "A Measure of the Geographic
    Concentration in French Manufacturing Industries". Regional Science
    and Urban Economics 29: 575-604
    
    '''
    industry_totals = x.sum(axis=0)
    if hs==None:
        hs = 1. / industry_totals
    x2s = np.sum((x.sum(axis=1) * 1. / x.sum())**2)
    ms_inds = np.zeros(x.shape[1])
    for gi in np.arange(x.shape[1]):
        s2s = np.sum((x[:, gi] * 1. / industry_totals[gi])**2)
        h = hs[gi]
        num = ((s2s - x2s) / (1. - x2s)) - h
        den = 1. - h
        ms_inds[gi] = num / den
    return ms_inds

def maurel_sedillot_msg_pop(x):
    '''
    Maurel and Sedillot (1999) [1]_ index of concentration. Implemented to be
    computed with data about people (segregation/diversity) rather than as
    industry concentration, following Mare et al (2012) [2]_

    ...

    Arguments
    ---------
    x       : array
              N x k array containing N rows (one per neighborhood) and k columns
              (one per cultural group)
    Returns
    -------
    a       : array
              MS index for every group k

    References
    ----------

    .. [1] Maurel, F. and Sedillot, B. (1999). "A Measure of the Geographic
    Concentration in French Manufacturing Industries". Regional Science
    and Urban Economics 29: 575-604

    .. [2] Mare, D., Pinkerton, R., Poot, J. and Coleman, A. (2012)
    Residential Sorting Across Auckland Neighbourhoods. Mimeo. Wellington:
    Motu Economic and Public Policy Research.
    
    '''
    pas = x.sum(axis=1)
    pgs = x.sum(axis=0)
    p = pas.sum()
    pap = pas * 1. / p
    eg_inds = np.zeros(x.shape[1])
    for g in np.arange(x.shape[1]):
        pgas = x[:, g]
        pg = pgs[g]
        num1n = ((pgas * 1. / pg)**2 - (pas * 1. / p)**2).sum()
        num1d = 1. - ((pas * 1. / p)**2).sum()
        num2 = 1. / pg
        den = 1. - (1. / pg)
        eg_inds[g] = ((num1n / num1d) - num2) / den
    return eg_inds

if __name__=='__main__':
    np.random.seed(1)
    x = np.round(np.random.random((10, 3)) * 100).astype(int)
    #x[:, 2] = 0
    ids = [ \
           #abundance, \
           #margalev_md, \
           #menhinick_mi, \
           #simpson_so, \
           #simpson_sd, \
           #fractionalization_gs, \
           #herfindahl_hd, \
           #shannon_se, \
           #gini_gi, \
           #gini_gi_m, \
           #hoover_hi, \
           #segregation_gsg, \
           #modified_segregation_msg, \
           #isolation_isg, \
            isolation_ii, \
           #gini_gig, \
           #ellison_glaeser_egg, \
           #ellison_glaeser_egg_pop, \
           #maurel_sedillot_msg, \
           #maurel_sedillot_msg_pop, \
           #theil_th, \
           #theil_th_brute, \
            ]
    res = [(f_i.func_name, f_i(x)) for f_i in ids]
    print '\nIndices'
    for r in res:
        print r[1], '\t', r[0]

    tau = np.random.random((x.shape[1], x.shape[1]))
    for i in range(tau.shape[0]):
        tau[i, i] = 1.
    #print similarity_w_wd(x, tau)

