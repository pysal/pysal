from __future__ import division
"""
Apply smoothing to rate computation

[Longer Description]

Author(s):
    Myunghwa Hwang mhwang4@gmail.com
    David Folch dfolch@asu.edu
    Luc Anselin luc.anselin@asu.edu
    Serge Rey srey@asu.edu

"""

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>, David Folch <dfolch@asu.edu>, Luc Anselin <luc.anselin@asu.edu>, Serge Rey <srey@asu.edu"

import pysal
from ..weights import comb, Kernel, W
from ..weights.util import get_points_array
from ..cg import Point, Ray, LineSegment
from ..cg import get_angle_between, get_points_dist, get_segment_point_dist,\
                 get_point_at_angle_and_dist, convex_hull, get_bounding_box
from ..common import np, KDTree, requires as _requires
from ..weights.spatial_lag import lag_spatial as slag
from scipy.stats import gamma, norm, chi2, poisson

__all__ = ['Excess_Risk', 'Empirical_Bayes', 'Spatial_Empirical_Bayes', 'Spatial_Rate', 'Kernel_Smoother', 'Age_Adjusted_Smoother', 'Disk_Smoother', 'Spatial_Median_Rate', 'Spatial_Filtering', 'Headbanging_Triples', 'Headbanging_Median_Rate', 'flatten', 'weighted_median', 'sum_by_n', 'crude_age_standardization', 'direct_age_standardization', 'indirect_age_standardization', 'standardized_mortality_ratio', 'choynowski', 'assuncao_rate']


def flatten(l, unique=True):
    """flatten a list of lists

    Parameters
    ----------
    l          : list
                 of lists
    unique     : boolean
                 whether or not only unique items are wanted (default=True)

    Returns
    -------
    list
        of single items

    Examples
    --------

    Creating a sample list whose elements are lists of integers

    >>> l = [[1, 2], [3, 4, ], [5, 6]]

    Applying flatten function

    >>> flatten(l)
    [1, 2, 3, 4, 5, 6]

    """
    l = reduce(lambda x, y: x + y, l)
    if not unique:
        return list(l)
    return list(set(l))


def weighted_median(d, w):
    """A utility function to find a median of d based on w

    Parameters
    ----------
    d          : array
                 (n, 1), variable for which median will be found
    w          : array
                 (n, 1), variable on which d's median will be decided

    Notes
    -----
    d and w are arranged in the same order

    Returns
    -------
    float
        median of d

    Examples
    --------

    Creating an array including five integers.
    We will get the median of these integers.

    >>> d = np.array([5,4,3,1,2])

    Creating another array including weight values for the above integers.
    The median of d will be decided with a consideration to these weight
    values.

    >>> w = np.array([10, 22, 9, 2, 5])

    Applying weighted_median function

    >>> weighted_median(d, w)
    4

    """
    dtype = [('w', '%s' % w.dtype), ('v', '%s' % d.dtype)]
    d_w = np.array(zip(w, d), dtype=dtype)
    d_w.sort(order='v')
    reordered_w = d_w['w'].cumsum()
    cumsum_threshold = reordered_w[-1] * 1.0 / 2
    median_inx = (reordered_w >= cumsum_threshold).nonzero()[0][0]
    if reordered_w[median_inx] == cumsum_threshold and len(d) - 1 > median_inx:
        return np.sort(d)[median_inx:median_inx + 2].mean()
    return np.sort(d)[median_inx]


def sum_by_n(d, w, n):
    """A utility function to summarize a data array into n values
       after weighting the array with another weight array w

    Parameters
    ----------
    d          : array
                 (t, 1), numerical values
    w          : array
                 (t, 1), numerical values for weighting
    n          : integer
                 the number of groups
                 t = c*n (c is a constant)

    Returns
    -------
               : array
                 (n, 1), an array with summarized values

    Examples
    --------

    Creating an array including four integers.
    We will compute weighted means for every two elements.

    >>> d = np.array([10, 9, 20, 30])

    Here is another array with the weight values for d's elements.

    >>> w = np.array([0.5, 0.1, 0.3, 0.8])

    We specify the number of groups for which the weighted mean is computed.

    >>> n = 2

    Applying sum_by_n function

    >>> sum_by_n(d, w, n)
    array([  5.9,  30. ])

    """
    t = len(d)
    h = t // n #must be floor!
    d = d * w
    return np.array([sum(d[i: i + h]) for i in range(0, t, h)])


def crude_age_standardization(e, b, n):
    """A utility function to compute rate through crude age standardization

    Parameters
    ----------
    e          : array
                 (n*h, 1), event variable measured for each age group across n spatial units
    b          : array
                 (n*h, 1), population at risk variable measured for each age group across n spatial units
    n          : integer
                 the number of spatial units

    Notes
    -----
    e and b are arranged in the same order

    Returns
    -------
               : array
                 (n, 1), age standardized rate

    Examples
    --------

    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    Specifying the number of regions.

    >>> n = 2

    Applying crude_age_standardization function to e and b

    >>> crude_age_standardization(e, b, n)
    array([ 0.2375    ,  0.26666667])

    """
    r = e * 1.0 / b
    b_by_n = sum_by_n(b, 1.0, n)
    age_weight = b * 1.0 / b_by_n.repeat(len(e) // n)
    return sum_by_n(r, age_weight, n)


def direct_age_standardization(e, b, s, n, alpha=0.05):
    """A utility function to compute rate through direct age standardization

    Parameters
    ----------
    e          : array
                 (n*h, 1), event variable measured for each age group across n spatial units
    b          : array
                 (n*h, 1), population at risk variable measured for each age group across n spatial units
    s          : array
                 (n*h, 1), standard population for each age group across n spatial units
    n          : integer
                 the number of spatial units
    alpha      : float
                 significance level for confidence interval

    Notes
    -----
    e, b, and s are arranged in the same order

    Returns
    -------
    list
        a list of n tuples; a tuple has a rate and its lower and upper limits
        age standardized rates and confidence intervals

    Examples
    --------

    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([1000, 1000, 1100, 900, 1000, 900, 1100, 900])

    For direct age standardization, we also need the data for standard population.
    Standard population is a reference population-at-risk (e.g., population distribution for the U.S.)
    whose age distribution can be used as a benchmarking point for comparing age distributions
    across regions (e.g., population distribution for Arizona and California).
    Another array including standard population is created.

    >>> s = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])

    Specifying the number of regions.

    >>> n = 2

    Applying direct_age_standardization function to e and b

    >>> [i[0] for i in direct_age_standardization(e, b, s, n)]
    [0.023744019138755977, 0.026650717703349279]

    """
    age_weight = (1.0 / b) * (s * 1.0 / sum_by_n(s, 1.0, n).repeat(len(s) // n))
    adjusted_r = sum_by_n(e, age_weight, n)
    var_estimate = sum_by_n(e, np.square(age_weight), n)
    g_a = np.square(adjusted_r) / var_estimate
    g_b = var_estimate / adjusted_r
    k = [age_weight[i:i + len(b) // n].max() for i in range(0, len(b),
                                                           len(b) // n)]
    g_a_k = np.square(adjusted_r + k) / (var_estimate + np.square(k))
    g_b_k = (var_estimate + np.square(k)) / (adjusted_r + k)
    summed_b = sum_by_n(b, 1.0, n)
    res = []
    for i in range(len(adjusted_r)):
        if adjusted_r[i] == 0:
            upper = 0.5 * chi2.ppf(1 - 0.5 * alpha)
            lower = 0.0
        else:
            lower = gamma.ppf(0.5 * alpha, g_a[i], scale=g_b[i])
            upper = gamma.ppf(1 - 0.5 * alpha, g_a_k[i], scale=g_b_k[i])
        res.append((adjusted_r[i], lower, upper))
    return res


def indirect_age_standardization(e, b, s_e, s_b, n, alpha=0.05):
    """A utility function to compute rate through indirect age standardization

    Parameters
    ----------
    e          : array
                 (n*h, 1), event variable measured for each age group across n spatial units
    b          : array
                 (n*h, 1), population at risk variable measured for each age group across n spatial units
    s_e        : array
                 (n*h, 1), event variable measured for each age group across n spatial units in a standard population
    s_b        : array
                 (n*h, 1), population variable measured for each age group across n spatial units in a standard population
    n          : integer
                 the number of spatial units
    alpha      : float
                 significance level for confidence interval

    Notes
    -----
    e, b, s_e, and s_b are arranged in the same order

    Returns
    -------
    list
        a list of n tuples; a tuple has a rate and its lower and upper limits
        age standardized rate

    Examples
    --------

    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    For indirect age standardization, we also need the data for standard population and event.
    Standard population is a reference population-at-risk (e.g., population distribution for the U.S.)
    whose age distribution can be used as a benchmarking point for comparing age distributions
    across regions (e.g., popoulation distribution for Arizona and California).
    When the same concept is applied to the event variable,
    we call it standard event (e.g., the number of cancer patients in the U.S.).
    Two additional arrays including standard population and event are created.

    >>> s_e = np.array([100, 45, 120, 100, 50, 30, 200, 80])
    >>> s_b = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])

    Specifying the number of regions.

    >>> n = 2

    Applying indirect_age_standardization function to e and b

    >>> [i[0] for i in indirect_age_standardization(e, b, s_e, s_b, n)]
    [0.23723821989528798, 0.2610803324099723]

    """
    smr = standardized_mortality_ratio(e, b, s_e, s_b, n)
    s_r_all = sum(s_e * 1.0) / sum(s_b * 1.0)
    adjusted_r = s_r_all * smr

    e_by_n = sum_by_n(e, 1.0, n)
    log_smr = np.log(smr)
    log_smr_sd = 1.0 / np.sqrt(e_by_n)
    norm_thres = norm.ppf(1 - 0.5 * alpha)
    log_smr_lower = log_smr - norm_thres * log_smr_sd
    log_smr_upper = log_smr + norm_thres * log_smr_sd
    smr_lower = np.exp(log_smr_lower) * s_r_all
    smr_upper = np.exp(log_smr_upper) * s_r_all
    res = zip(adjusted_r, smr_lower, smr_upper)
    return res


def standardized_mortality_ratio(e, b, s_e, s_b, n):
    """A utility function to compute standardized mortality ratio (SMR).

    Parameters
    ----------
    e          : array
                 (n*h, 1), event variable measured for each age group across n spatial units
    b          : array
                 (n*h, 1), population at risk variable measured for each age group across n spatial units
    s_e        : array
                 (n*h, 1), event variable measured for each age group across n spatial units in a standard population
    s_b        : array
                 (n*h, 1), population variable measured for each age group across n spatial units in a standard population
    n          : integer
                 the number of spatial units

    Notes
    -----
    e, b, s_e, and s_b are arranged in the same order

    Returns
    -------
    array
        (nx1)

    Examples
    --------

    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    To compute standardized mortality ratio (SMR),
    we need two additional arrays for standard population and event.
    Creating s_e and s_b for standard event and population, respectively.

    >>> s_e = np.array([100, 45, 120, 100, 50, 30, 200, 80])
    >>> s_b = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])

    Specifying the number of regions.

    >>> n = 2

    Applying indirect_age_standardization function to e and b

    >>> standardized_mortality_ratio(e, b, s_e, s_b, n)
    array([ 2.48691099,  2.73684211])

    """
    s_r = s_e * 1.0 / s_b
    e_by_n = sum_by_n(e, 1.0, n)
    expected = sum_by_n(b, s_r, n)
    smr = e_by_n * 1.0 / expected
    return smr


def choynowski(e, b, n, threshold=None):
    """Choynowski map probabilities [Choynowski1959]_ .

    Parameters
    ----------
    e          : array(n*h, 1)
                 event variable measured for each age group across n spatial units
    b          : array(n*h, 1)
                 population at risk variable measured for each age group across n spatial units
    n          : integer
                 the number of spatial units
    threshold  : float
                 Returns zero for any p-value greater than threshold

    Notes
    -----
    e and b are arranged in the same order

    Returns
    -------
               : array (nx1)

    Examples
    --------

    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    Specifying the number of regions.

    >>> n = 2

    Applying indirect_age_standardization function to e and b

    >>> print choynowski(e, b, n)
    [ 0.30437751  0.29367033]

    """
    e_by_n = sum_by_n(e, 1.0, n)
    b_by_n = sum_by_n(b, 1.0, n)
    r_by_n = sum(e_by_n) * 1.0 / sum(b_by_n)
    expected = r_by_n * b_by_n
    p = []
    for index, i in enumerate(e_by_n):
        if i <= expected[index]:
            p.append(poisson.cdf(i, expected[index]))
        else:
            p.append(1 - poisson.cdf(i - 1, expected[index]))
    if threshold:
        p = [i if i < threshold else 0.0 for i in p]
    return np.array(p)


def assuncao_rate(e, b):
    """The standardized rates where the mean and stadard deviation used for
    the standardization are those of Empirical Bayes rate estimates
    The standardized rates resulting from this function are used to compute
    Moran's I corrected for rate variables [Choynowski1959]_ .

    Parameters
    ----------
    e          : array(n, 1)
                 event variable measured at n spatial units
    b          : array(n, 1)
                 population at risk variable measured at n spatial units

    Notes
    -----
    e and b are arranged in the same order

    Returns
    -------
               : array (nx1)

    Examples
    --------

    Creating an array of an event variable (e.g., the number of cancer patients)
    for 8 regions.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same 8 regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    Computing the rates

    >>> print assuncao_rate(e, b)[:4]
    [ 1.04319254 -0.04117865 -0.56539054 -1.73762547]

    """

    y = e * 1.0 / b
    e_sum, b_sum = sum(e), sum(b)
    ebi_b = e_sum * 1.0 / b_sum
    s2 = sum(b * ((y - ebi_b) ** 2)) / b_sum
    ebi_a = s2 - ebi_b / (float(b_sum) / len(e))
    ebi_v = ebi_a + ebi_b / b
    return (y - ebi_b) / np.sqrt(ebi_v)

class _Smoother(object):
    """
    This is a helper class that implements things that all smoothers should do.
    Right now, the only thing that we need to propagate is the by_col function. 

    TBQH, most of these smoothers should be functions, not classes (aside from
    maybe headbanging triples), since they're literally only inits + one
    attribute. 
    """
    def __init__(self):
        pass

    @classmethod
    def by_col(cls, df, e,b, inplace=False, **kwargs):
        """
        Compute smoothing by columns in a dataframe. 

        Parameters
        -----------
        df      :  pandas.DataFrame
                   a dataframe containing the data to be smoothed
        e       :  string or list of strings
                   the name or names of columns containing event variables to be
                   smoothed
        b       :  string or list of strings
                   the name or names of columns containing the population
                   variables to be smoothed
        inplace :  bool
                   a flag denoting whether to output a copy of `df` with the
                   relevant smoothed columns appended, or to append the columns
                   directly to `df` itself. 
        **kwargs:  optional keyword arguments
                   optional keyword options that are passed directly to the
                   smoother. 

        Returns
        ---------
        a copy of `df` containing the columns. Or, if `inplace`, this returns
        None, but implicitly adds columns to `df`.  
        """
        if not inplace:
            new = df.copy()
            cls.by_col(new, e, b, inplace=True, **kwargs)
            return new
        if isinstance(e, str):
            e = [e]
        if isinstance(b, str):
            b = [b]
        if len(b) == 1 and len(e) > 1:
            b = b * len(e)
        try:
            assert len(e) == len(b)
        except AssertionError:
            raise ValueError('There is no one-to-one mapping between event'
                             ' variable and population at risk variable!')
        for ei, bi in zip(e,b):
            ename = ei
            bname = bi
            ei = df[ename]
            bi = df[bname]
            outcol = '_'.join(('-'.join((ename, bname)), cls.__name__.lower()))
            df[outcol] = cls(ei, bi, **kwargs).r

class Excess_Risk(_Smoother):
    """Excess Risk

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units

    Attributes
    ----------
    r           : array (n, 1)
                  execess risk values

    Examples
    --------

    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> stl = pysal.open(pysal.examples.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the number of homocides and population.
    Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating an instance of Excess_Risk class using stl_e and stl_b

    >>> er = Excess_Risk(stl_e, stl_b)

    Extracting the excess risk values through the property r of the Excess_Risk instance, er

    >>> er.r[:10]
    array([ 0.20665681,  0.43613787,  0.42078261,  0.22066928,  0.57981596,
            0.35301709,  0.56407549,  0.17020994,  0.3052372 ,  0.25821905])

    """
    def __init__(self, e, b):
        e = np.asarray(e).reshape(-1,1)
        b = np.asarray(b).reshape(-1,1)
        r_mean = e.sum() * 1.0 / b.sum()
        self.r = e * 1.0 / (b * r_mean)


class Empirical_Bayes(_Smoother):
    """Aspatial Empirical Bayes Smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from Empirical Bayes Smoothing

    Examples
    --------

    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> stl = pysal.open(pysal.examples.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the number of homocides and population.
    Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating an instance of Empirical_Bayes class using stl_e and stl_b

    >>> eb = Empirical_Bayes(stl_e, stl_b)

    Extracting the risk values through the property r of the Empirical_Bayes instance, eb

    >>> eb.r[:10]
    array([  2.36718950e-05,   4.54539167e-05,   4.78114019e-05,
             2.76907146e-05,   6.58989323e-05,   3.66494122e-05,
             5.79952721e-05,   2.03064590e-05,   3.31152999e-05,
             3.02748380e-05])

    """
    def __init__(self, e, b):
        e = np.asarray(e).reshape(-1,1)
        b = np.asarray(b).reshape(-1,1)
        e_sum, b_sum = e.sum() * 1.0, b.sum() * 1.0
        r_mean = e_sum / b_sum
        rate = e * 1.0 / b
        r_variat = rate - r_mean
        r_var_left = (b * r_variat * r_variat).sum() * 1.0 / b_sum
        r_var_right = r_mean * 1.0 / b.mean()
        r_var = r_var_left - r_var_right
        weight = r_var / (r_var + r_mean / b)
        self.r = weight * rate + (1.0 - weight) * r_mean

class _Spatial_Smoother(_Smoother):
    """
    This is a helper class that implements things that all the things that
    spatial smoothers should do. 
    .
    Right now, the only thing that we need to propagate is the by_col function. 

    TBQH, most of these smoothers should be functions, not classes (aside from
    maybe headbanging triples), since they're literally only inits + one
    attribute. 
    """
    def __init__(self):
        pass

    @classmethod
    def by_col(cls, df, e,b, w=None, inplace=False, **kwargs):
        """
        Compute smoothing by columns in a dataframe. 

        Parameters
        -----------
        df      :  pandas.DataFrame
                   a dataframe containing the data to be smoothed
        e       :  string or list of strings
                   the name or names of columns containing event variables to be
                   smoothed
        b       :  string or list of strings
                   the name or names of columns containing the population
                   variables to be smoothed
        w       :  pysal.weights.W or list of pysal.weights.W
                   the spatial weights object or objects to use with the
                   event-population pairs. If not provided and a weights object
                   is in the dataframe's metadata, that weights object will be
                   used. 
        inplace :  bool
                   a flag denoting whether to output a copy of `df` with the
                   relevant smoothed columns appended, or to append the columns
                   directly to `df` itself. 
        **kwargs:  optional keyword arguments
                   optional keyword options that are passed directly to the
                   smoother. 

        Returns
        ---------
        a copy of `df` containing the columns. Or, if `inplace`, this returns
        None, but implicitly adds columns to `df`.  
        """
        if not inplace:
            new = df.copy()
            cls.by_col(new, e, b, w=w, inplace=True, **kwargs)
            return new
        if isinstance(e, str):
            e = [e]
        if isinstance(b, str):
            b = [b]
        if w is None:
            found = False
            for k in df._metadata:
                w = df.__dict__.get(w, None)
                if isinstance(w, W):
                    found = True
            if not found:
                raise Exception('Weights not provided and no weights attached to frame!'
                                    ' Please provide a weight or attach a weight to the'
                                    ' dataframe')
        if isinstance(w, W):
            w = [w] * len(e)
        if len(b) == 1 and len(e) > 1:
            b = b * len(e)
        try:
            assert len(e) == len(b)
        except AssertionError:
            raise ValueError('There is no one-to-one mapping between event'
                             ' variable and population at risk variable!')
        for ei, bi, wi in zip(e, b, w):
            ename = ei
            bname = bi
            ei = df[ename]
            bi = df[bname]
            outcol = '_'.join(('-'.join((ename, bname)), cls.__name__.lower()))
            df[outcol] = cls(ei, bi, w=wi, **kwargs).r

class Spatial_Empirical_Bayes(_Spatial_Smoother):
    """Spatial Empirical Bayes Smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : spatial weights instance

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from Empirical Bayes Smoothing

    Examples
    --------

    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> stl = pysal.open(pysal.examples.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the number of homocides and population.
    Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating a spatial weights instance by reading in stl.gal file.

    >>> stl_w = pysal.open(pysal.examples.get_path('stl.gal'), 'r').read()

    Ensuring that the elements in the spatial weights instance are ordered
    by the given sequential numbers from 1 to the number of observations in stl_hom.csv

    >>> if not stl_w.id_order_set: stl_w.id_order = range(1,len(stl) + 1)

    Creating an instance of Spatial_Empirical_Bayes class using stl_e, stl_b, and stl_w

    >>> s_eb = Spatial_Empirical_Bayes(stl_e, stl_b, stl_w)

    Extracting the risk values through the property r of s_eb

    >>> s_eb.r[:10]
    array([  4.01485749e-05,   3.62437513e-05,   4.93034844e-05,
             5.09387329e-05,   3.72735210e-05,   3.69333797e-05,
             5.40245456e-05,   2.99806055e-05,   3.73034109e-05,
             3.47270722e-05])
    """
    def __init__(self, e, b, w):
        if not w.id_order_set:
            raise ValueError("w id_order must be set to align with the order of e an b")
        e = np.asarray(e).reshape(-1,1)
        b = np.asarray(b).reshape(-1,1)
        r_mean = Spatial_Rate(e, b, w).r
        rate = e * 1.0 / b
        r_var_left = np.ones_like(e) * 1.
        ngh_num = np.ones_like(e)
        bi = slag(w, b) + b
        for i, idv in enumerate(w.id_order):
            ngh = w[idv].keys() + [idv]
            nghi = [w.id2i[k] for k in ngh]
            ngh_num[i] = len(nghi)
            v = sum(np.square(rate[nghi] - r_mean[i]) * b[nghi])
            r_var_left[i] = v
        r_var_left = r_var_left / bi
        r_var_right = r_mean / (bi / ngh_num)
        r_var = r_var_left - r_var_right
        r_var[r_var < 0] = 0.0
        self.r = r_mean + (rate - r_mean) * (r_var / (r_var + (r_mean / b)))

class Spatial_Rate(_Spatial_Smoother):
    """Spatial Rate Smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : spatial weights instance

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from spatial rate smoothing

    Examples
    --------

    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> stl = pysal.open(pysal.examples.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the number of homocides and population.
    Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating a spatial weights instance by reading in stl.gal file.

    >>> stl_w = pysal.open(pysal.examples.get_path('stl.gal'), 'r').read()

    Ensuring that the elements in the spatial weights instance are ordered
    by the given sequential numbers from 1 to the number of observations in stl_hom.csv

    >>> if not stl_w.id_order_set: stl_w.id_order = range(1,len(stl) + 1)

    Creating an instance of Spatial_Rate class using stl_e, stl_b, and stl_w

    >>> sr = Spatial_Rate(stl_e,stl_b,stl_w)

    Extracting the risk values through the property r of sr

    >>> sr.r[:10]
    array([  4.59326407e-05,   3.62437513e-05,   4.98677081e-05,
             5.09387329e-05,   3.72735210e-05,   4.01073093e-05,
             3.79372794e-05,   3.27019246e-05,   4.26204928e-05,
             3.47270722e-05])
    """
    def __init__(self, e, b, w):
        if not w.id_order_set:
            raise ValueError("w id_order must be set to align with the order of e and b")
        else:
            e = np.asarray(e).reshape(-1,1)
            b = np.asarray(b).reshape(-1,1)
            w.transform = 'b'
            w_e, w_b = slag(w, e), slag(w, b)
            self.r = (e + w_e) / (b + w_b)
            w.transform = 'o'


class Kernel_Smoother(_Spatial_Smoother):
    """Kernal smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : Kernel weights instance

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from spatial rate smoothing

    Examples
    --------

    Creating an array including event values for 6 regions

    >>> e = np.array([10, 1, 3, 4, 2, 5])

    Creating another array including population-at-risk values for the 6 regions

    >>> b = np.array([100, 15, 20, 20, 80, 90])

    Creating a list containing geographic coordinates of the 6 regions' centroids

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a kernel-based spatial weights instance by using the above points

    >>> kw=Kernel(points)

    Ensuring that the elements in the kernel-based weights are ordered
    by the given sequential numbers from 0 to 5

    >>> if not kw.id_order_set: kw.id_order = range(0,len(points))

    Applying kernel smoothing to e and b

    >>> kr = Kernel_Smoother(e, b, kw)

    Extracting the smoothed rates through the property r of the Kernel_Smoother instance

    >>> kr.r
    array([ 0.10543301,  0.0858573 ,  0.08256196,  0.09884584,  0.04756872,
            0.04845298])
    """
    def __init__(self, e, b, w):
        if type(w) != Kernel:
            raise Error('w must be an instance of Kernel weights')
        if not w.id_order_set:
            raise ValueError("w id_order must be set to align with the order of e and b")
        else:
            e = np.asarray(e).reshape(-1,1)
            b = np.asarray(b).reshape(-1,1)
            w_e, w_b = slag(w, e), slag(w, b)
            self.r = w_e / w_b


class Age_Adjusted_Smoother(_Spatial_Smoother):
    """Age-adjusted rate smoothing

    Parameters
    ----------
    e           : array (n*h, 1)
                  event variable measured for each age group across n spatial units
    b           : array (n*h, 1)
                  population at risk variable measured for each age group across n spatial units
    w           : spatial weights instance
    s           : array (n*h, 1)
                  standard population for each age group across n spatial units

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from spatial rate smoothing

    Notes
    -----
    Weights used to smooth age-specific events and populations are simple binary weights

    Examples
    --------

    Creating an array including 12 values for the 6 regions with 2 age groups

    >>> e = np.array([10, 8, 1, 4, 3, 5, 4, 3, 2, 1, 5, 3])

    Creating another array including 12 population-at-risk values for the 6 regions

    >>> b = np.array([100, 90, 15, 30, 25, 20, 30, 20, 80, 80, 90, 60])

    For age adjustment, we need another array of values containing standard population
    s includes standard population data for the 6 regions

    >>> s = np.array([98, 88, 15, 29, 20, 23, 33, 25, 76, 80, 89, 66])

    Creating a list containing geographic coordinates of the 6 regions' centroids

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a kernel-based spatial weights instance by using the above points

    >>> kw=Kernel(points)

    Ensuring that the elements in the kernel-based weights are ordered
    by the given sequential numbers from 0 to 5

    >>> if not kw.id_order_set: kw.id_order = range(0,len(points))

    Applying age-adjusted smoothing to e and b

    >>> ar = Age_Adjusted_Smoother(e, b, kw, s)

    Extracting the smoothed rates through the property r of the Age_Adjusted_Smoother instance

    >>> ar.r
    array([ 0.10519625,  0.08494318,  0.06440072,  0.06898604,  0.06952076,
            0.05020968])
    """
    def __init__(self, e, b, w, s, alpha=0.05):
        e = np.asarray(e).reshape(-1, 1)
        b = np.asarray(b).reshape(-1, 1)
        s = np.asarray(s).flatten()
        t = len(e)
        h = t // w.n
        w.transform = 'b'
        e_n, b_n = [], []
        for i in range(h):
            e_n.append(slag(w, e[i::h]).tolist())
            b_n.append(slag(w, b[i::h]).tolist())
        e_n = np.array(e_n).reshape((1, t), order='F')[0]
        b_n = np.array(b_n).reshape((1, t), order='F')[0]
        e_n = e_n.reshape(s.shape)
        b_n = b_n.reshape(s.shape)
        r = direct_age_standardization(e_n, b_n, s, w.n, alpha=alpha)
        self.r = np.array([i[0] for i in r])
        w.transform = 'o'
    
    @_requires('pandas')
    @classmethod
    def by_col(cls, df, e,b, w=None, s=None, **kwargs):
        """
        Compute smoothing by columns in a dataframe. 

        Parameters
        -----------
        df      :  pandas.DataFrame
                   a dataframe containing the data to be smoothed
        e       :  string or list of strings
                   the name or names of columns containing event variables to be
                   smoothed
        b       :  string or list of strings
                   the name or names of columns containing the population
                   variables to be smoothed
        w       :  pysal.weights.W or list of pysal.weights.W
                   the spatial weights object or objects to use with the
                   event-population pairs. If not provided and a weights object
                   is in the dataframe's metadata, that weights object will be
                   used. 
        s       :  string or list of strings
                   the name or names of columns to use as a standard population
                   variable for the events `e` and at-risk populations `b`. 
        inplace :  bool
                   a flag denoting whether to output a copy of `df` with the
                   relevant smoothed columns appended, or to append the columns
                   directly to `df` itself. 
        **kwargs:  optional keyword arguments
                   optional keyword options that are passed directly to the
                   smoother. 

        Returns
        ---------
        a copy of `df` containing the columns. Or, if `inplace`, this returns
        None, but implicitly adds columns to `df`.  
        """
        if s is None:
            raise Exception('Standard population variable "s" must be supplied.')
        import pandas as pd
        if isinstance(e, str):
            e = [e]
        if isinstance(b, str):
            b = [b]
        if isinstance(s, str):
            s = [s]
        if w is None:
            found = False
            for k in df._metadata:
                w = df.__dict__.get(w, None)
                if isinstance(w, W):
                    found = True
                    break
            if not found:
                raise Exception('Weights not provided and no weights attached to frame!'
                                    ' Please provide a weight or attach a weight to the'
                                    ' dataframe.')
        if isinstance(w, W):
            w = [w] * len(e)
        if not all(isinstance(wi, W) for wi in w):
            raise Exception('Weights object must be an instance of '
                            ' pysal.weights.W!')
        b = b * len(e) if len(b) == 1 and len(e) > 1 else b
        s = s * len(e) if len(s) == 1 and len(e) > 1 else s
        try:
            assert len(e) == len(b)
            assert len(e) == len(s)
            assert len(e) == len(w)
        except AssertionError:
            raise ValueError('There is no one-to-one mapping between event'
                             ' variable and population at risk variable, and '
                             ' standard population variable, and spatial '
                             ' weights!')
        rdf = []
        max_len = 0
        for ei, bi, wi, si in zip(e, b, w, s):
            ename = ei
            bname = bi
            h = len(ei) // wi.n
            outcol = '_'.join(('-'.join((ename, bname)), cls.__name__.lower()))
            this_r = cls(df[ei], df[bi], w=wi, s=df[si], **kwargs).r
            max_len = 0 if len(this_r) > max_len else max_len
            rdf.append((outcol, this_r.tolist()))
        padded = (r[1] + [None] * max_len for r in rdf)
        rdf = zip((r[0] for r in rdf), padded)
        rdf = pd.DataFrame.from_items(rdf)
        return rdf


class Disk_Smoother(_Spatial_Smoother):
    """Locally weighted averages or disk smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : spatial weights matrix

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from disk smoothing

    Examples
    --------

    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> stl = pysal.open(pysal.examples.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the number of homocides and population.
    Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating a spatial weights instance by reading in stl.gal file.

    >>> stl_w = pysal.open(pysal.examples.get_path('stl.gal'), 'r').read()

    Ensuring that the elements in the spatial weights instance are ordered
    by the given sequential numbers from 1 to the number of observations in stl_hom.csv

    >>> if not stl_w.id_order_set: stl_w.id_order = range(1,len(stl) + 1)

    Applying disk smoothing to stl_e and stl_b

    >>> sr = Disk_Smoother(stl_e,stl_b,stl_w)

    Extracting the risk values through the property r of s_eb

    >>> sr.r[:10]
    array([  4.56502262e-05,   3.44027685e-05,   3.38280487e-05,
             4.78530468e-05,   3.12278573e-05,   2.22596997e-05,
             2.67074856e-05,   2.36924573e-05,   3.48801587e-05,
             3.09511832e-05])
    """

    def __init__(self, e, b, w):
        if not w.id_order_set:
            raise ValueError("w id_order must be set to align with the order of e and b")
        else:
            e = np.asarray(e).reshape(-1,1)
            b = np.asarray(b).reshape(-1,1)
            r = e * 1.0 / b
            weight_sum = []
            for i in w.id_order:
                weight_sum.append(sum(w.weights[i]))
            self.r = slag(w, r) / np.array(weight_sum).reshape(-1,1)


class Spatial_Median_Rate(_Spatial_Smoother):
    """Spatial Median Rate Smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : spatial weights instance
    aw          : array (n, 1)
                  auxiliary weight variable measured across n spatial units
    iteration   : integer
                  the number of interations

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from spatial median rate smoothing
    w           : spatial weights instance
    aw          : array (n, 1)
                  auxiliary weight variable measured across n spatial units

    Examples
    --------

    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> stl = pysal.open(pysal.examples.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the number of homocides and population.
    Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating a spatial weights instance by reading in stl.gal file.

    >>> stl_w = pysal.open(pysal.examples.get_path('stl.gal'), 'r').read()

    Ensuring that the elements in the spatial weights instance are ordered
    by the given sequential numbers from 1 to the number of observations in stl_hom.csv

    >>> if not stl_w.id_order_set: stl_w.id_order = range(1,len(stl) + 1)

    Computing spatial median rates without iteration

    >>> smr0 = Spatial_Median_Rate(stl_e,stl_b,stl_w)

    Extracting the computed rates through the property r of the Spatial_Median_Rate instance

    >>> smr0.r[:10]
    array([  3.96047383e-05,   3.55386859e-05,   3.28308921e-05,
             4.30731238e-05,   3.12453969e-05,   1.97300409e-05,
             3.10159267e-05,   2.19279204e-05,   2.93763432e-05,
             2.93763432e-05])

    Recomputing spatial median rates with 5 iterations

    >>> smr1 = Spatial_Median_Rate(stl_e,stl_b,stl_w,iteration=5)

    Extracting the computed rates through the property r of the Spatial_Median_Rate instance

    >>> smr1.r[:10]
    array([  3.11293620e-05,   2.95956330e-05,   3.11293620e-05,
             3.10159267e-05,   2.98436066e-05,   2.76406686e-05,
             3.10159267e-05,   2.94788171e-05,   2.99460806e-05,
             2.96981070e-05])

    Computing spatial median rates by using the base variable as auxilliary weights
    without iteration

    >>> smr2 = Spatial_Median_Rate(stl_e,stl_b,stl_w,aw=stl_b)

    Extracting the computed rates through the property r of the Spatial_Median_Rate instance

    >>> smr2.r[:10]
    array([  5.77412020e-05,   4.46449551e-05,   5.77412020e-05,
             5.77412020e-05,   4.46449551e-05,   3.61363528e-05,
             3.61363528e-05,   4.46449551e-05,   5.77412020e-05,
             4.03987355e-05])

    Recomputing spatial median rates by using the base variable as auxilliary weights
    with 5 iterations

    >>> smr3 = Spatial_Median_Rate(stl_e,stl_b,stl_w,aw=stl_b,iteration=5)

    Extracting the computed rates through the property r of the Spatial_Median_Rate instance

    >>> smr3.r[:10]
    array([  3.61363528e-05,   4.46449551e-05,   3.61363528e-05,
             3.61363528e-05,   4.46449551e-05,   3.61363528e-05,
             3.61363528e-05,   4.46449551e-05,   3.61363528e-05,
             4.46449551e-05])
    >>>
    """
    def __init__(self, e, b, w, aw=None, iteration=1):
        if not w.id_order_set:
            raise ValueError("w id_order must be set to align with the order of e and b")
        e = np.asarray(e).flatten()
        b = np.asarray(b).flatten() 
        self.r = e * 1.0 / b
        self.aw, self.w = aw, w
        while iteration:
            self.__search_median()
            iteration -= 1

    def __search_median(self):
        r, aw, w = self.r, self.aw, self.w
        new_r = []
        if self.aw is None:
            for i, id in enumerate(w.id_order):
                r_disk = np.append(r[i], r[w.neighbor_offsets[id]])
                new_r.append(np.median(r_disk))
        else:
            for i, id in enumerate(w.id_order):
                id_d = [i] + list(w.neighbor_offsets[id])
                aw_d, r_d = aw[id_d], r[id_d]
                new_r.append(weighted_median(r_d, aw_d))
        self.r = np.asarray(new_r).reshape(r.shape)


class Spatial_Filtering(_Smoother):
    """Spatial Filtering

    Parameters
    ----------
    bbox        : a list of two lists where each list is a pair of coordinates
                  a bounding box for the entire n spatial units
    data        : array (n, 2)
                  x, y coordinates
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    x_grid      : integer
                  the number of cells on x axis
    y_grid      : integer
                  the number of cells on y axis
    r           : float
                  fixed radius of a moving window
    pop         : integer
                  population threshold to create adaptive moving windows

    Attributes
    ----------
    grid        : array (x_grid*y_grid, 2)
                  x, y coordinates for grid points
    r           : array (x_grid*y_grid, 1)
                  rate values for grid points

    Notes
    -----
    No tool is provided to find an optimal value for r or pop.

    Examples
    --------

    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> stl = pysal.open(pysal.examples.get_path('stl_hom.csv'), 'r')

    Reading the stl data in the WKT format so that
    we can easily extract polygon centroids

    >>> fromWKT = pysal.core.util.WKTParser()
    >>> stl.cast('WKT',fromWKT)

    Extracting polygon centroids through iteration

    >>> d = np.array([i.centroid for i in stl[:,0]])

    Specifying the bounding box for the stl_hom data.
    The bbox should includes two points for the left-bottom and the right-top corners

    >>> bbox = [[-92.700676, 36.881809], [-87.916573, 40.3295669]]

    The 11th and 14th columns in stl_hom.csv includes the number of homocides and population.
    Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Applying spatial filtering by using a 10*10 mesh grid and a moving window
    with 2 radius

    >>> sf_0 = Spatial_Filtering(bbox,d,stl_e,stl_b,10,10,r=2)

    Extracting the resulting rates through the property r of the Spatial_Filtering instance

    >>> sf_0.r[:10]
    array([  4.23561763e-05,   4.45290850e-05,   4.56456221e-05,
             4.49133384e-05,   4.39671835e-05,   4.44903042e-05,
             4.19845497e-05,   4.11936548e-05,   3.93463504e-05,
             4.04376345e-05])

    Applying another spatial filtering by allowing the moving window to grow until
    600000 people are found in the window

    >>> sf = Spatial_Filtering(bbox,d,stl_e,stl_b,10,10,pop=600000)

    Checking the size of the reulting array including the rates

    >>> sf.r.shape
    (100,)

    Extracting the resulting rates through the property r of the Spatial_Filtering instance

    >>> sf.r[:10]
    array([  3.73728738e-05,   4.04456300e-05,   4.04456300e-05,
             3.81035327e-05,   4.54831940e-05,   4.54831940e-05,
             3.75658628e-05,   3.75658628e-05,   3.75658628e-05,
             3.75658628e-05])
    """

    def __init__(self, bbox, data, e, b, x_grid, y_grid, r=None, pop=None):
        e= np.asarray(e).reshape(-1,1)
        b= np.asarray(b).reshape(-1,1)
        data_tree = KDTree(data)
        x_range = bbox[1][0] - bbox[0][0]
        y_range = bbox[1][1] - bbox[0][1]
        x, y = np.mgrid[bbox[0][0]:bbox[1][0]:float(x_range) / x_grid,
                        bbox[0][1]:bbox[1][1]:float(y_range) / y_grid]
        self.grid = zip(x.ravel(), y.ravel())
        self.r = []
        if r is None and pop is None:
            raise ValueError("Either r or pop should not be None")
        if r is not None:
            pnts_in_disk = data_tree.query_ball_point(self.grid, r=r)
            for i in pnts_in_disk:
                r = e[i].sum() * 1.0 / b[i].sum()
                self.r.append(r)
        if pop is not None:
            half_nearest_pnts = data_tree.query(self.grid, k=len(e))[1]
            for i in half_nearest_pnts:
                e_n, b_n = e[i].cumsum(), b[i].cumsum()
                b_n_filter = b_n <= pop
                e_n_f, b_n_f = e_n[b_n_filter], b_n[b_n_filter]
                if len(e_n_f) == 0:
                    e_n_f = e_n[[0]]
                    b_n_f = b_n[[0]]
                self.r.append(e_n_f[-1] * 1.0 / b_n_f[-1])
        self.r = np.array(self.r)
    
    @_requires('pandas')
    @classmethod
    def by_col(cls, df, e, b, x_grid, y_grid, geom_col='geometry', **kwargs):
        """
        Compute smoothing by columns in a dataframe. The bounding box and point
        information is computed from the geometry column. 

        Parameters
        -----------
        df      :  pandas.DataFrame
                   a dataframe containing the data to be smoothed
        e       :  string or list of strings
                   the name or names of columns containing event variables to be
                   smoothed
        b       :  string or list of strings
                   the name or names of columns containing the population
                   variables to be smoothed
        x_grid  :  integer
                   number of grid cells to use along the x-axis
        y_grid  :  integer
                   number of grid cells to use along the y-axis
        geom_col:  string
                   the name of the column in the dataframe containing the
                   geometry information. 
        **kwargs:  optional keyword arguments
                   optional keyword options that are passed directly to the
                   smoother. 
        Returns
        ---------
        a new dataframe of dimension (x_grid*y_grid, 3), containing the 
        coordinates of the grid cells and the rates associated with those grid
        cells.
        """
        import pandas as pd
        # prep for application over multiple event/population pairs
        if isinstance(e, str):
            e = [e]
        if isinstance(b, str):
            b = [b]
        if len(e) > len(b):
            b = b * len(e)
        if isinstance(x_grid, (int, float)):
            x_grid = [x_grid] * len(e)
        if isinstance(y_grid, (int, float)):
            y_grid = [y_grid] * len(e)

        bbox = get_bounding_box(df[geom_col])
        bbox = [[bbox.left, bbox.lower], [bbox.right, bbox.upper]]
        data = get_points_array(df[geom_col])
        res = []
        for ename, bname, xgi, ygi in zip(e, b, x_grid, y_grid):
            r = cls(bbox, data, df[ename], df[bname], xgi, ygi, **kwargs)
            grid = np.asarray(r.grid).reshape(-1,2)
            name = '_'.join(('-'.join((ename, bname)), cls.__name__.lower()))
            colnames = ('_'.join((name, suffix)) for suffix in ['X', 'Y', 'R'])
            items = [(name, col) for name,col in zip(colnames, [grid[:,0], 
                                                                grid[:,1], 
                                                                r.r])]
            res.append(pd.DataFrame.from_items(items))
        outdf = pd.concat(res)
        return outdf

class Headbanging_Triples(object):
    """Generate a pseudo spatial weights instance that contains headbanging triples

    Parameters
    ----------
    data        : array (n, 2)
                  numpy array of x, y coordinates
    w           : spatial weights instance
    k           : integer number of nearest neighbors
    t           : integer
                  the number of triples
    angle       : integer between 0 and 180
                  the angle criterium for a set of triples
    edgecorr    : boolean
                  whether or not correction for edge points is made

    Attributes
    ----------
    triples     : dictionary
                  key is observation record id, value is a list of lists of triple ids
    extra       : dictionary
                  key is observation record id, value is a list of the following:
                  tuple of original triple observations
                  distance between original triple observations
                  distance between an original triple observation and its extrapolated point

    Examples
    --------

    importing k-nearest neighbor weights creator

    >>> from pysal import knnW_from_array

    Reading data in stl_hom.csv into stl_db to extract values
    for event and population-at-risk variables

    >>> stl_db = pysal.open(pysal.examples.get_path('stl_hom.csv'),'r')

    Reading the stl data in the WKT format so that
    we can easily extract polygon centroids

    >>> fromWKT = pysal.core.util.WKTParser()
    >>> stl_db.cast('WKT',fromWKT)

    Extracting polygon centroids through iteration

    >>> d = np.array([i.centroid for i in stl_db[:,0]])

    Using the centroids, we create a 5-nearst neighbor weights

    >>> w = knnW_from_array(d,k=5)

    Ensuring that the elements in the spatial weights instance are ordered
    by the order of stl_db's IDs

    >>> if not w.id_order_set: w.id_order = w.id_order

    Finding headbaning triples by using 5 nearest neighbors

    >>> ht = Headbanging_Triples(d,w,k=5)

    Checking the members of triples

    >>> for k, item in ht.triples.items()[:5]: print k, item
    0 [(5, 6), (10, 6)]
    1 [(4, 7), (4, 14), (9, 7)]
    2 [(0, 8), (10, 3), (0, 6)]
    3 [(4, 2), (2, 12), (8, 4)]
    4 [(8, 1), (12, 1), (8, 9)]

    Opening sids2.shp file

    >>> sids = pysal.open(pysal.examples.get_path('sids2.shp'),'r')

    Extracting the centroids of polygons in the sids data

    >>> sids_d = np.array([i.centroid for i in sids])

    Creating a 5-nearest neighbors weights from the sids centroids

    >>> sids_w = knnW_from_array(sids_d,k=5)

    Ensuring that the members in sids_w are ordered by
    the order of sids_d's ID

    >>> if not sids_w.id_order_set: sids_w.id_order = sids_w.id_order

    Finding headbaning triples by using 5 nearest neighbors

    >>> s_ht = Headbanging_Triples(sids_d,sids_w,k=5)

    Checking the members of the found triples

    >>> for k, item in s_ht.triples.items()[:5]: print k, item
    0 [(1, 18), (1, 21), (1, 33)]
    1 [(2, 40), (2, 22), (22, 40)]
    2 [(39, 22), (1, 9), (39, 17)]
    3 [(16, 6), (19, 6), (20, 6)]
    4 [(5, 15), (27, 15), (35, 15)]

    Finding headbanging triples by using 5 nearest neighbors with edge correction

    >>> s_ht2 = Headbanging_Triples(sids_d,sids_w,k=5,edgecor=True)

    Checking the members of the found triples

    >>> for k, item in s_ht2.triples.items()[:5]: print k, item
    0 [(1, 18), (1, 21), (1, 33)]
    1 [(2, 40), (2, 22), (22, 40)]
    2 [(39, 22), (1, 9), (39, 17)]
    3 [(16, 6), (19, 6), (20, 6)]
    4 [(5, 15), (27, 15), (35, 15)]

    Checking the extrapolated point that is introduced into the triples
    during edge correction

    >>> extrapolated = s_ht2.extra[72]

    Checking the observation IDs constituting the extrapolated triple

    >>> extrapolated[0]
    (89, 77)

    Checking the distances between the extraploated point and the observation 89 and 77

    >>> round(extrapolated[1],5), round(extrapolated[2],6)
    (0.33753, 0.302707)
    """
    def __init__(self, data, w, k=5, t=3, angle=135.0, edgecor=False):
        if k < 3:
            raise ValueError("w should be NeareastNeighbors instance & the number of neighbors should be more than 3.")
        if not w.id_order_set:
            raise ValueError("w id_order must be set to align with the order of data")
        self.triples, points = {}, {}
        for i, pnt in enumerate(data):
            ng = w.neighbor_offsets[i]
            points[(i, Point(pnt))] = dict(zip(ng, [Point(d)
                                                    for d in data[ng]]))
        for i, pnt in points.keys():
            ng = points[(i, pnt)]
            tr, tr_dis = {}, []
            for c in comb(ng.keys(), 2):
                p2, p3 = ng[c[0]], ng[c[-1]]
                ang = get_angle_between(Ray(pnt, p2), Ray(pnt, p3))
                if ang > angle or (ang < 0.0 and ang + 360 > angle):
                    tr[tuple(c)] = (p2, p3)
            if len(tr) > t:
                for c in tr.keys():
                    p2, p3 = tr[c]
                    tr_dis.append((get_segment_point_dist(
                        LineSegment(p2, p3), pnt), c))
                tr_dis = sorted(tr_dis)[:t]
                self.triples[i] = [trp for dis, trp in tr_dis]
            else:
                self.triples[i] = tr.keys()
        if edgecor:
            self.extra = {}
            ps = dict([(p, i) for i, p in points.keys()])
            chull = convex_hull(ps.keys())
            chull = [p for p in chull if len(self.triples[ps[p]]) == 0]
            for point in chull:
                key = (ps[point], point)
                ng = points[key]
                ng_dist = [(get_points_dist(point, p), p) for p in ng.values()]
                ng_dist_s = sorted(ng_dist, reverse=True)
                extra = None
                while extra is None and len(ng_dist_s) > 0:
                    p2 = ng_dist_s.pop()[-1]
                    p3s = ng.values()
                    p3s.remove(p2)
                    for p3 in p3s:
                        dist_p2_p3 = get_points_dist(p2, p3)
                        dist_p_p2 = get_points_dist(point, p2)
                        dist_p_p3 = get_points_dist(point, p3)
                        if dist_p_p2 <= dist_p_p3:
                            ray1, ray2, s_pnt, dist, c = Ray(p2, point), Ray(p2, p3), p2, dist_p_p2, (ps[p2], ps[p3])
                        else:
                            ray1, ray2, s_pnt, dist, c = Ray(p3, point), Ray(p3, p2), p3, dist_p_p3, (ps[p3], ps[p2])
                        ang = get_angle_between(ray1, ray2)
                        if ang >= 90 + angle / 2 or (ang < 0 and ang + 360 >= 90 + angle / 2):
                            ex_point = get_point_at_angle_and_dist(
                                ray1, angle, dist)
                            extra = [c, dist_p2_p3, get_points_dist(
                                s_pnt, ex_point)]
                            break
                self.triples[ps[point]].append(extra[0])
                self.extra[ps[point]] = extra


class Headbanging_Median_Rate(object):
    """Headbaning Median Rate Smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    t           : Headbanging_Triples instance
    aw          : array (n, 1)
                  auxilliary weight variable measured across n spatial units
    iteration   : integer
                  the number of iterations

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from headbanging median smoothing

    Examples
    --------

    importing k-nearest neighbor weights creator

    >>> from pysal import knnW_from_array

    opening the sids2 shapefile

    >>> sids = pysal.open(pysal.examples.get_path('sids2.shp'), 'r')

    extracting the centroids of polygons in the sids2 data

    >>> sids_d = np.array([i.centroid for i in sids])

    creating a 5-nearest neighbors weights from the centroids

    >>> sids_w = knnW_from_array(sids_d,k=5)

    ensuring that the members in sids_w are ordered

    >>> if not sids_w.id_order_set: sids_w.id_order = sids_w.id_order

    finding headbanging triples by using 5 neighbors
        return outdf

    >>> s_ht = Headbanging_Triples(sids_d,sids_w,k=5)

    reading in the sids2 data table

    >>> sids_db = pysal.open(pysal.examples.get_path('sids2.dbf'), 'r')

    extracting the 10th and 9th columns in the sids2.dbf and
    using data values as event and population-at-risk variables

    >>> s_e, s_b = np.array(sids_db[:,9]), np.array(sids_db[:,8])

    computing headbanging median rates from s_e, s_b, and s_ht

    >>> sids_hb_r = Headbanging_Median_Rate(s_e,s_b,s_ht)

    extracting the computed rates through the property r of the Headbanging_Median_Rate instance

    >>> sids_hb_r.r[:5]
    array([ 0.00075586,  0.        ,  0.0008285 ,  0.0018315 ,  0.00498891])

    recomputing headbanging median rates with 5 iterations

    >>> sids_hb_r2 = Headbanging_Median_Rate(s_e,s_b,s_ht,iteration=5)

    extracting the computed rates through the property r of the Headbanging_Median_Rate instance

    >>> sids_hb_r2.r[:5]
    array([ 0.0008285 ,  0.00084331,  0.00086896,  0.0018315 ,  0.00498891])

    recomputing headbanging median rates by considring a set of auxilliary weights

    >>> sids_hb_r3 = Headbanging_Median_Rate(s_e,s_b,s_ht,aw=s_b)

    extracting the computed rates through the property r of the Headbanging_Median_Rate instance

    >>> sids_hb_r3.r[:5]
    array([ 0.00091659,  0.        ,  0.00156838,  0.0018315 ,  0.00498891])
    """
    def __init__(self, e, b, t, aw=None, iteration=1):
        self.r = e * 1.0 / b
        self.tr, self.aw = t.triples, aw
        if hasattr(t, 'extra'):
            self.extra = t.extra
        while iteration:
            self.__search_headbanging_median()
            iteration -= 1

    def __get_screens(self, id, triples, weighted=False):
        r, tr = self.r, self.tr
        if len(triples) == 0:
            return r[id]
        if hasattr(self, 'extra') and id in self.extra:
            extra = self.extra
            trp_r = r[list(triples[0])]
            # observed rate 
            # plus difference in rate scaled by ratio of extrapolated distance
            # & observed distance. 
            trp_r[-1] = trp_r[0] + (trp_r[0] - trp_r[-1]) * (
                extra[id][-1] * 1.0 / extra[id][1])
            trp_r = sorted(trp_r)
            if not weighted:
                return r[id], trp_r[0], trp_r[-1]
            else:
                trp_aw = self.aw[triples[0]]
                extra_w = trp_aw[0] + (trp_aw[0] - trp_aw[-
                                                          1]) * (extra[id][-1] * 1.0 / extra[id][1])
                return r[id], trp_r[0], trp_r[-1], self.aw[id], trp_aw[0] + extra_w
        if not weighted:
            lowest, highest = [], []
            for trp in triples:
                trp_r = np.sort(r[list(trp)])
                lowest.append(trp_r[0])
                highest.append(trp_r[-1])
            return r[id], np.median(np.array(lowest)), np.median(np.array(highest))
        if weighted:
            lowest, highest = [], []
            lowest_aw, highest_aw = [], []
            for trp in triples:
                trp_r = r[list(trp)]
                dtype = [('r', '%s' % trp_r.dtype), ('w',
                                                     '%s' % self.aw.dtype)]
                trp_r = np.array(zip(trp_r, list(trp)), dtype=dtype)
                trp_r.sort(order='r')
                lowest.append(trp_r['r'][0])
                highest.append(trp_r['r'][-1])
                lowest_aw.append(self.aw[trp_r['w'][0]])
                highest_aw.append(self.aw[trp_r['w'][-1]])
            wm_lowest = weighted_median(np.array(lowest), np.array(lowest_aw))
            wm_highest = weighted_median(
                np.array(highest), np.array(highest_aw))
            triple_members = flatten(triples, unique=False)
            return r[id], wm_lowest, wm_highest, self.aw[id] * len(triples), self.aw[triple_members].sum()
    
    def __get_median_from_screens(self, screens):
        if isinstance(screens, float):
            return screens
        elif len(screens) == 3:
            return np.median(np.array(screens))
        elif len(screens) == 5:
            rk, wm_lowest, wm_highest, w1, w2 = screens
            if rk >= wm_lowest and rk <= wm_highest:
                return rk
            elif rk < wm_lowest and w1 < w2:
                return wm_lowest
            elif rk > wm_highest and w1 < w2:
                return wm_highest
            else:
                return rk

    def __search_headbanging_median(self):
        r, tr = self.r, self.tr
        new_r = []
        for k in tr.keys():
            screens = self.__get_screens(
                k, tr[k], weighted=(self.aw is not None))
            new_r.append(self.__get_median_from_screens(screens))
        self.r = np.array(new_r)
    
    @_requires('pandas')
    @classmethod
    def by_col(cls, df, e, b, t=None, geom_col='geometry', inplace=False, **kwargs):
        """
        Compute smoothing by columns in a dataframe. The bounding box and point
        information is computed from the geometry column. 

        Parameters
        -----------
        df      :  pandas.DataFrame
                   a dataframe containing the data to be smoothed
        e       :  string or list of strings
                   the name or names of columns containing event variables to be
                   smoothed
        b       :  string or list of strings
                   the name or names of columns containing the population
                   variables to be smoothed
        t       :  Headbanging_Triples instance or list of Headbanging_Triples
                   list of headbanging triples instances. If not provided, this
                   is computed from the geometry column of the dataframe. 
        geom_col:  string
                   the name of the column in the dataframe containing the
                   geometry information. 
        inplace :  bool
                   a flag denoting whether to output a copy of `df` with the
                   relevant smoothed columns appended, or to append the columns
                   directly to `df` itself. 
        **kwargs:  optional keyword arguments
                   optional keyword options that are passed directly to the
                   smoother. 
        Returns
        ---------
        a new dataframe containing the smoothed Headbanging Median Rates for the
        event/population pairs. If done inplace, there is no return value and
        `df` is modified in place. 
        """
        import pandas as pd
        if not inplace:
            new = df.copy()
            cls.by_col(new, e, b, t=t, geom_col=geom_col, inplace=True, **kwargs)
            return new
        import pandas as pd
        # prep for application over multiple event/population pairs
        if isinstance(e, str):
            e = [e]
        if isinstance(b, str):
            b = [b]
        if len(e) > len(b):
            b = b * len(e)

        data = get_points_array(df[geom_col])

        #Headbanging_Triples doesn't take **kwargs, so filter its arguments
        # (self, data, w, k=5, t=3, angle=135.0, edgecor=False):
        
        w = kwargs.pop('w', None)
        if w is None:
            found = False
            for k in df._metadata:
                w = df.__dict__.get(w, None)
                if isinstance(w, W):
                    found = True
            if not found:
                raise Exception('Weights not provided and no weights attached to frame!'
                                    ' Please provide a weight or attach a weight to the'
                                    ' dataframe')
        
        k = kwargs.pop('k', 5)
        t = kwargs.pop('t', 3)
        angle = kwargs.pop('angle', 135.0)
        edgecor = kwargs.pop('edgecor', False)

        hbt = Headbanging_Triples(data, w, k=k, t=t, angle=angle,
                                  edgecor=edgecor)
        
        res = []
        for ename, bname in zip(e, b):
            r = cls(df[ename], df[bname], hbt, **kwargs).r
            name = '_'.join(('-'.join((ename, bname)), cls.__name__.lower()))
            df[name] = r
