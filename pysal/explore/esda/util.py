import numpy as np


def fdr(pvalues, alpha=0.05):
    """
    Calculate the p-value cut-off to control for
    the false discovery rate (FDR) for multiple testing.

    If by controlling for FDR, all of n null hypotheses
    are rejected, the conservative Bonferroni bound (alpha/n)
    is returned instead.

    Arguments
    ---------
    pvalues     : array
                  (n, ), p values for n multiple tests.
    alpha       : float, optional
                  Significance level. Default is 0.05.

    Returns
    -------
                : float
                  Adjusted criterion for rejecting the null hypothesis.
                  If by controlling for FDR, all of n null hypotheses
                  are rejected, the conservative Bonferroni bound (alpha/n)
                  is returned.

    Notes
    -----

    For technical details see :cite:`Benjamini:2001` and
    :cite:`Castro:2006tz`.


    Examples
    --------
    >>> import pysal.lib
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> w = pysal.lib.io.open(pysal.lib.examples.get_path("stl.gal")).read()
    >>> f = pysal.lib.io.open(pysal.lib.examples.get_path("stl_hom.txt"))
    >>> y = np.array(f.by_col['HR8893'])
    >>> from pysal.explore.esda.moran import Moran_Local
    >>> from pysal.explore.esda import fdr
    >>> lm = Moran_Local(y, w, transformation = "r", permutations = 999)
    >>> fdr(lm.p_sim, 0.1)
    0.002564102564102564
    >>> fdr(lm.p_sim, 0.05) #return the conservative Bonferroni bound
    0.000641025641025641

    """

    n = len(pvalues)
    p_sort = np.sort(pvalues)[::-1]
    index = np.arange(n, 0, -1)
    p_fdr = index * alpha / n
    search = p_sort < p_fdr
    sig_all = np.where(search)[0]
    if len(sig_all) == 0:
        return alpha/n
    else:
        return p_fdr[sig_all[0]]
