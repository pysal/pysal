"""
Directional Analysis of Dynamic LISAs

"""
__author__ = "Sergio J. Rey <srey@asu.edu"

__all__ = ['rose']

import numpy as np
import pysal


def rose(Y, w, k=8, permutations=0):
    """
    Calculation of rose diagram for local indicators of spatial association.

    Parameters
    ----------
    Y             : array 
                    (n, 2), variable observed on n spatial units over 2 time. 
                    periods
    w             : W
                    spatial weights object.
    k             : int, optional
                    number of circular sectors in rose diagram (the default is 
                    8).
    permutations  : int, optional
                    number of random spatial permutations for calculation of 
                    pseudo p-values (the default is 0).

    Returns
    -------
    results       : dictionary 
                    (keys defined below)
    counts        : array 
                    (k, 1), number of vectors with angular movement falling in 
                    each sector.
    cuts          : array 
                    (k, 1), intervals defining circular sectors (in radians).
    random_counts : array 
                    (permutations, k), counts from random permutations.
    pvalues       : array 
                    (k, 1), one sided (upper tail) pvalues for observed counts.

    Notes
    -----
    Based on [Rey2011]_ .

    Examples
    --------
    Constructing data for illustration of directional LISA analytics.
    Data is for the 48 lower US states over the period 1969-2009 and
    includes per capita income normalized to the national average. 

    Load comma delimited data file in and convert to a numpy array

    >>> f=open(pysal.examples.get_path("spi_download.csv"),'r')
    >>> lines=f.readlines()
    >>> f.close()
    >>> lines=[line.strip().split(",") for line in lines]
    >>> names=[line[2] for line in lines[1:-5]]
    >>> data=np.array([map(int,line[3:]) for line in lines[1:-5]])

    Bottom of the file has regional data which we don't need for this example
    so we will subset only those records that match a state name

    >>> sids=range(60)
    >>> out=['"United States 3/"',
    ...      '"Alaska 3/"',
    ...      '"District of Columbia"',
    ...      '"Hawaii 3/"',
    ...      '"New England"',
    ...      '"Mideast"',
    ...      '"Great Lakes"',
    ...      '"Plains"',
    ...      '"Southeast"',
    ...      '"Southwest"',
    ...      '"Rocky Mountain"',
    ...      '"Far West 3/"']
    >>> snames=[name for name in names if name not in out]
    >>> sids=[names.index(name) for name in snames]
    >>> states=data[sids,:]
    >>> us=data[0]
    >>> years=np.arange(1969,2009)

    Now we convert state incomes to express them relative to the national
    average

    >>> rel=states/(us*1.)

    Create our contiguity matrix from an external GAL file and row standardize
    the resulting weights

    >>> gal=pysal.open(pysal.examples.get_path('states48.gal'))
    >>> w=gal.read()
    >>> w.transform='r'

    Take the first and last year of our income data as the interval to do the
    directional directional analysis

    >>> Y=rel[:,[0,-1]]

    Set the random seed generator which is used in the permutation based
    inference for the rose diagram so that we can replicate our example
    results

    >>> np.random.seed(100)

    Call the rose function to construct the directional histogram for the
    dynamic LISA statistics. We will use four circular sectors for our
    histogram

    >>> r4=rose(Y,w,k=4,permutations=999)

    What are the cut-offs for our histogram - in radians

    >>> r4['cuts']
    array([ 0.        ,  1.57079633,  3.14159265,  4.71238898,  6.28318531])

    How many vectors fell in each sector

    >>> r4['counts']
    array([32,  5,  9,  2])

    What are the pseudo-pvalues for these counts based on 999 random spatial
    permutations of the state income data

    >>> r4['pvalues']
    array([ 0.02 ,  0.001,  0.001,  0.001])

    Repeat the exercise but now for 8 rather than 4 sectors

    >>> r8=rose(Y,w,permutations=999)
    >>> r8['counts']
    array([19, 13,  3,  2,  7,  2,  1,  1])
    >>> r8['pvalues']
    array([ 0.445,  0.042,  0.079,  0.003,  0.005,  0.1  ,  0.269,  0.002])


    """
    results = {}
    sw = 2 * np.pi / k
    cuts = np.arange(0.0, 2 * np.pi + sw, sw)
    wY = pysal.lag_spatial(w, Y)
    dx = Y[:, -1] - Y[:, 0]
    dy = wY[:, -1] - wY[:, 0]
    theta = np.arctan2(dy, dx)
    neg = theta < 0.0
    utheta = theta * (1 - neg) + neg * (2 * np.pi + theta)
    counts, bins = np.histogram(utheta, cuts)
    results['counts'] = counts
    results['cuts'] = cuts
    if permutations:
        n, k1 = Y.shape
        ids = np.arange(n)
        all_counts = np.zeros((permutations, k))
        for i in range(permutations):
            rid = np.random.permutation(ids)
            YR = Y[rid, :]
            wYR = pysal.lag_spatial(w, YR)
            dx = YR[:, -1] - YR[:, 0]
            dy = wYR[:, -1] - wYR[:, 0]
            theta = np.arctan2(dy, dx)
            neg = theta < 0.0
            utheta = theta * (1 - neg) + neg * (2 * np.pi + theta)
            c, b = np.histogram(utheta, cuts)
            c.shape = (1, k)
            all_counts[i, :] = c
        larger = sum(all_counts >= counts)
        p_l = permutations - larger
        extreme = (p_l) < larger
        extreme = np.where(extreme, p_l, larger)
        p = (extreme + 1.) / (permutations + 1.)
        results['pvalues'] = p
        results['random_counts'] = all_counts

    return results
