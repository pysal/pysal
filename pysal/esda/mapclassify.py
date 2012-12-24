"""
A module of classification schemes for choropleth mapping.
"""
__author__ = "Sergio J. Rey"
__credits__ = "Copyright (c) 2009-10 Sergio J. Rey"

__all__ = ['quantile', 'Map_Classifier', 'Box_Plot', 'Equal_Interval', 'Fisher_Jenks', 'Fisher_Jenks_Sampled', 'Jenks_Caspall', 'Jenks_Caspall_Forced', 'Jenks_Caspall_Sampled', 'Max_P_Classifier', 'Maximum_Breaks', 'Natural_Breaks', 'Quantiles', 'Percentiles', 'Std_Mean', 'User_Defined', 'gadf', 'K_classifiers']

from pysal.common import *

K = 5  # default number of classes in any map scheme with this as an argument


def quantile(y, k=4):
    """
    Calculates the quantiles for an array

    Parameters
    ----------
    y : array (n,1)
        values to classify
    k : int
        number of quantiles

    Returns
    -------
    implicit  : array (n,1)
                quantile values

    Examples
    --------
    >>> x = np.arange(1000)
    >>> quantile(x)
    array([ 249.75,  499.5 ,  749.25,  999.  ])
    >>> quantile(x, k = 3)
    array([ 333.,  666.,  999.])
    >>>

    Note that if there are enough ties that the quantile values repeat, we
    collapse to pseudo quantiles in which case the number of classes will be less than k

    >>> x = [1.0] * 100
    >>> x.extend([3.0] * 40)
    >>> len(x)
    140
    >>> y = np.array(x)
    >>> quantile(y)
    array([ 1.,  3.])
    """
    w = 100. / k
    p = np.arange(w, 100 + w, w)
    if p[-1] > 100.0:
        p[-1] = 100.0
    q = np.array([stats.scoreatpercentile(y, pct) for pct in p])
    return np.unique(q)


def binC(y, bins):
    """
    Bin categorical/qualitative data

    Parameters
    ----------
    y : array (n,q)
        categorical values
    bins :  array (k,1)
        unique values associated with each bin

    Return
    ------
    b : array (n,q)
        bin membership, values between 0 and k-1

    Examples
    --------
    >>> np.random.seed(1)
    >>> x = np.random.randint(2, 8, (10, 3))
    >>> bins = range(2, 8)
    >>> x
    array([[7, 5, 6],
           [2, 3, 5],
           [7, 2, 2],
           [3, 6, 7],
           [6, 3, 4],
           [6, 7, 4],
           [6, 5, 6],
           [4, 6, 7],
           [4, 6, 3],
           [3, 2, 7]])
    >>> y = binC(x, bins)
    >>> y
    array([[5, 3, 4],
           [0, 1, 3],
           [5, 0, 0],
           [1, 4, 5],
           [4, 1, 2],
           [4, 5, 2],
           [4, 3, 4],
           [2, 4, 5],
           [2, 4, 1],
           [1, 0, 5]])
    >>>
    """

    if np.rank(y) == 1:
        k = 1
        n = np.shape(y)[0]
    else:
        n, k = np.shape(y)
    b = np.zeros((n, k), dtype='int')
    for i, bin in enumerate(bins):
        b[np.nonzero(y == bin)] = i

    # check for non-binned items and print a warning if needed
    vals = set(y.flatten())
    for val in vals:
        if val not in bins:
            print 'warning: value not in bin: ', val
            print 'bins: ', bins

    return b


def bin(y, bins):
    """
    bin interval/ratio data

    Parameters
    ----------
    y : array (n,q)
        values to bin
    bins : array (k,1)
        upper bounds of each bin (monotonic)

    Returns
    -------
    b : array (n,q)
        values of values between 0 and k-1

    Examples
    --------
    >>> np.random.seed(1)
    >>> x = np.random.randint(2, 20, (10, 3))
    >>> bins = [10, 15, 20]
    >>> b = bin(x, bins)
    >>> x
    array([[ 7, 13, 14],
           [10, 11, 13],
           [ 7, 17,  2],
           [18,  3, 14],
           [ 9, 15,  8],
           [ 7, 13, 12],
           [16,  6, 11],
           [19,  2, 15],
           [11, 11,  9],
           [ 3,  2, 19]])
    >>> b
    array([[0, 1, 1],
           [0, 1, 1],
           [0, 2, 0],
           [2, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [2, 0, 1],
           [2, 0, 1],
           [1, 1, 0],
           [0, 0, 2]])
    >>>
    """
    if np.rank(y) == 1:
        k = 1
        n = np.shape(y)[0]
    else:
        n, k = np.shape(y)
    b = np.zeros((n, k), dtype='int')
    i = len(bins)
    if type(bins) != list:
        bins = bins.tolist()
    binsc = copy.copy(bins)
    while binsc:
        i -= 1
        c = binsc.pop(-1)
        b[np.nonzero(y <= c)] = i
    return b


def bin1d(x, bins):
    """
    place values of a 1-d array into bins and determine counts of values in
    each bin

    Parameters
    ----------
    y : 1-d array
        values to bin
    bins : array (k,1)
        upper bounds of each bin (monotonic)

    Returns
    -------
    tuple(binIds,counts)

    binIds: 1-d array of integer bin Ids

    counts: number of elements of x falling in each bin

    Examples
    --------
    >>> x = np.arange(100, dtype = 'float')
    >>> bins = [25, 74, 100]
    >>> binIds, counts = bin1d(x, bins)
    >>> binIds
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2])
    >>> counts
    array([26, 49, 25])
    """
    left = [-sys.maxint]
    left.extend(bins[0:-1])
    right = bins
    cuts = zip(left, right)
    k = len(bins)
    binIds = np.zeros(x.shape, dtype='int')
    while cuts:
        k -= 1
        l, r = cuts.pop(-1)
        binIds += (x > l) * (x <= r) * k
    counts = np.bincount(binIds)
    return (binIds, counts)


def load_example():
    """
    Helper function for doc tests"""
    import pysal
    np.random.seed(10)
    dat = pysal.open(pysal.examples.get_path('calempdensity.csv'))
    cal = np.array([record[-1] for record in dat])
    return cal


def natural_breaks(values, k=5, itmax=100):
    """
    natural breaks helper function
    """
    values = np.array(values)
    n = len(values)
    uv = np.unique(values)
    uvk = len(uv)
    if  uvk < k:
        print 'Warning: Not enough unique values in array to form k classes'
        print "Warning: setting k to %d" % uvk
        k = uvk
    sids = np.random.permutation(range(len(uv)))[0:k]
    seeds = uv[sids]
    seeds.sort()
    diffs = abs(np.matrix([values - seed for seed in seeds]))
    c0 = diffs.argmin(axis=0)
    c0 = np.array(c0)[0]
    solving = True
    solved = False
    rk = range(k)
    it = 0
    while solving:
        # get centroids of clusters
        seeds = [np.median(values[c0 == c]) for c in rk]
        seeds.sort()
        # for each value find closest centroid
        diffs = abs(np.matrix([values - seed for seed in seeds]))
        # assign value to that centroid
        c1 = diffs.argmin(axis=0)
        c1 = np.array(c1)[0]
        #compare new classids to previous
        d = abs(c1 - c0)
        if d.sum() == 0:
            solving = False
            solved = True
        else:
            c0 = c1
        it += 1
        if it == itmax:
            solving = False
    class_ids = c1
    cuts = [max(values[c1 == c]) for c in rk]
    return sids, seeds, diffs, class_ids, solved, it, cuts


def _fisher_jenks_means(values, classes=5, sort=True):
    """
    Jenks Optimal (Natural Breaks) algorithm implemented in Python.
    The original Python code comes from here:
    http://danieljlewis.org/2010/06/07/jenks-natural-breaks-algorithm-in-python/
    and is based on a JAVA and Fortran code available here:
    https://stat.ethz.ch/pipermail/r-sig-geo/2006-March/000811.html

    Returns class breaks such that classes are internally homogeneous while
    assuring heterogeneity among classes.

    """

    if sort:
        values.sort()
    mat1 = []
    for i in range(0, len(values) + 1):
        temp = []
        for j in range(0, classes + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(0, len(values) + 1):
        temp = []
        for j in range(0, classes + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, classes + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(values) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(values) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(values[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, classes + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v

    k = len(values)

    kclass = []
    for i in range(0, classes + 1):
        kclass.append(0)
    kclass[classes] = float(values[len(values) - 1])
    kclass[0] = float(values[0])
    countNum = classes
    while countNum >= 2:
        pivot = mat1[k][countNum]
        id = int(pivot - 2)
        kclass[countNum - 1] = values[id]
        k = int(pivot - 1)
        countNum -= 1
    return kclass


class Map_Classifier:
    """
    Abstract class for all map classifications """
    def __init__(self, y):
        self.name = 'Map Classifier'
        self.y = y
        self._classify()
        self._summary()

    def _summary(self):
        yb = self.yb
        self.classes = [np.nonzero(yb == c)[0].tolist() for c in range(self.k)]
        self.tss = self.get_tss()
        self.adcm = self.get_adcm()
        self.gadf = self.get_gadf()

    def _classify(self):
        self._set_bins()
        self.yb, self.counts = bin1d(self.y, self.bins)

    def __str__(self):
        st = self._table_string()
        return st

    def __repr__(self):
        return self._table_string()

    def get_tss(self):
        """
        Total sum of squares around class means

        Returns sum of squares over all class means
        """
        tss = 0
        for class_def in self.classes:
            if len(class_def) > 0:
                yc = self.y[class_def]
                css = yc - yc.mean()
                css *= css
                tss += sum(css)
        return tss

    def _set_bins(self):
        pass

    def get_adcm(self):
        """
        Absolute deviation around class median (ADCM).

        Calculates the absolute deviations of each observation about its class
        median as a measure of fit for the classification method.

        Returns sum of ADCM over all classes
        """
        adcm = 0
        for class_def in self.classes:
            if len(class_def) > 0:
                yc = self.y[class_def]
                yc_med = np.median(yc)
                ycd = np.abs(yc - yc_med)
                adcm += sum(ycd)
        return adcm

    def get_gadf(self):
        """
        Goodness of absolute deviation of fit
        """
        adam = (np.abs(self.y - np.median(self.y))).sum()
        gadf = 1 - self.adcm / adam
        return gadf

    def _table_string(self, width=12, decimal=3):
        fmt = ".%df" % decimal
        fmt = "%" + fmt
        largest = max([len(fmt % i) for i in self.bins])
        width = largest
        fmt = "%d.%df" % (width, decimal)
        fmt = "%" + fmt
        k1 = self.k - 1
        h1 = "Lower"
        h1 = h1.center(largest)
        h2 = " "
        h2 = h2.center(10)
        h3 = "Upper"
        h3 = h3.center(largest + 1)

        largest = "%d" % max(self.counts)
        largest = len(largest) + 15
        h4 = "Count"

        h4 = h4.rjust(largest)
        table = []
        header = h1 + h2 + h3 + h4
        table.append(header)
        table.append("=" * len(header))

        rows = []
        for i, up in enumerate(self.bins):
            if i == 0:
                left = " " * width
                left += "   x[i] <= "
            else:
                left = fmt % self.bins[i - 1]
                left += " < x[i] <= "
            right = fmt % self.bins[i]
            row = left + right
            cnt = "%d" % self.counts[i]
            cnt = cnt.rjust(largest)
            row += cnt
            table.append(row)
        name = self.name
        top = name.center(len(row))
        table.insert(0, top)
        table.insert(1, " ")
        table = "\n".join(table)
        return table


class Equal_Interval(Map_Classifier):
    """
    Equal Interval Classification

    Parameters
    ----------
    y : array (n,1)
        values to classify
    k : int
        number of classes required

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations,
              each value is the id of the class the observation belongs to
              yb[i] = j  for j>=1  if bins[j-1] < y[i] <= bins[j], yb[i] = 0  otherwise
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class

    Examples
    --------
    >>> cal = load_example()
    >>> ei = Equal_Interval(cal, k = 5)
    >>> ei.k
    5
    >>> ei.counts
    array([57,  0,  0,  0,  1])
    >>> ei.bins
    array([  822.394,  1644.658,  2466.922,  3289.186,  4111.45 ])
    >>>


    Notes
    -----
    Intervals defined to have equal width:

    .. math::

        bins_j = min(y)+w*(j+1)

    with :math:`w=\\frac{max(y)-min(j)}{k}`
    """

    def __init__(self, y, k=K):
        """
        see class docstring

        """

        self.k = k
        Map_Classifier.__init__(self, y)
        self.name = 'Equal Interval'

    def _set_bins(self):
        y = self.y
        k = self.k
        max_y = max(y)
        min_y = min(y)
        rg = max_y - min_y
        width = rg * 1. / k
        cuts = np.arange(min_y + width, max_y + width, width)
        if len(cuts) > self.k:  # handle overshooting
            cuts = cuts[0:k]
        cuts[-1] = max_y
        bins = cuts.copy()
        self.bins = bins


class Percentiles(Map_Classifier):
    """
    Percentiles Map Classification

    Parameters
    ----------

    y    : array
           attribute to classify
    pct  : array
           percentiles default=[1,10,50,90,99,100]

    Attributes
    ----------
    yb     : array
             bin ids for observations (numpy array n x 1)

    bins   : array
             the upper bounds of each class (numpy array k x 1)

    k      : int
             the number of classes

    counts : int
             the number of observations falling in each class (numpy array k x 1)

    Examples
    --------
    >>> cal = load_example()
    >>> p = Percentiles(cal)
    >>> p.bins
    array([  1.35700000e-01,   5.53000000e-01,   9.36500000e+00,
             2.13914000e+02,   2.17994800e+03,   4.11145000e+03])
    >>> p.counts
    array([ 1,  5, 23, 23,  5,  1])
    >>> p2 = Percentiles(cal, pct = [50, 100])
    >>> p2.bins
    array([    9.365,  4111.45 ])
    >>> p2.counts
    array([29, 29])
    >>> p2.k
    2
    """

    def __init__(self, y, pct=[1, 10, 50, 90, 99, 100]):
        self.pct = pct
        Map_Classifier.__init__(self, y)
        self.name = 'Percentiles'

    def _set_bins(self):
        y = self.y
        pct = self.pct
        self.bins = np.array([stats.scoreatpercentile(y, p) for p in pct])
        self.k = len(self.bins)


class Box_Plot(Map_Classifier):
    """
    Box_Plot Map Classification


    Parameters
    ----------
    y     : array
            attribute to classify
    hinge : float
            multiplier for IQR

    Attributes
    ----------
    yb : array (n,1)
        bin ids for observations
    bins : array (n,1)
        the upper bounds of each class  (monotonic)
    k : int
        the number of classes
    counts : array (k,1)
        the number of observations falling in each class
    low_outlier_ids : array
        indices of observations that are low outliers
    high_outlier_ids : array
        indices of observations that are high outliers

    Notes
    -----

    The bins are set as follows::

        bins[0] = q[0]-hinge*IQR
        bins[1] = q[0]
        bins[2] = q[1]
        bins[3] = q[2]
        bins[4] = q[2]+hinge*IQR
        bins[5] = inf  (see Notes)

    where q is an array of the first three quartiles of y and
    IQR=q[2]-q[0]


    If q[2]+hinge*IQR > max(y) there will only be 5 classes and no high outliers,
        otherwise, there will be 6 classes and at least one high outlier.

    Examples
    --------
    >>> cal = load_example()
    >>> bp = Box_Plot(cal)
    >>> bp.bins
    array([ -5.28762500e+01,   2.56750000e+00,   9.36500000e+00,
             3.95300000e+01,   9.49737500e+01,   4.11145000e+03])
    >>> bp.counts
    array([ 0, 15, 14, 14,  6,  9])
    >>> bp.high_outlier_ids
    array([ 0,  6, 18, 29, 33, 36, 37, 40, 42])
    >>> cal[bp.high_outlier_ids]
    array([  329.92,   181.27,   370.5 ,   722.85,   192.05,   110.74,
            4111.45,   317.11,   264.93])
    >>> bx = Box_Plot(np.arange(100))
    >>> bx.bins
    array([ -49.5 ,   24.75,   49.5 ,   74.25,  148.5 ])

    """

    def __init__(self, y, hinge=1.5):
        """
        Parameters
        ----------
        y : array (n,1)
            attribute to classify
        hinge : float
            multiple of inter-quartile range (default=1.5)
        """
        self.hinge = hinge
        Map_Classifier.__init__(self, y)
        self.name = 'Box Plot'

    def _set_bins(self):
        y = self.y
        pct = [25, 50, 75, 100]
        bins = [stats.scoreatpercentile(y, p) for p in pct]
        iqr = bins[-2] - bins[0]
        self.iqr = iqr
        pivot = self.hinge * iqr
        left_fence = bins[0] - pivot
        right_fence = bins[-2] + pivot
        if right_fence < bins[-1]:
            bins.insert(-1, right_fence)
        else:
            bins[-1] = right_fence
        bins.insert(0, left_fence)
        self.bins = np.array(bins)
        self.k = len(pct)

    def _classify(self):
        Map_Classifier._classify(self)
        self.low_outlier_ids = np.nonzero(self.yb == 0)[0]
        self.high_outlier_ids = np.nonzero(self.yb == 5)[0]


class Quantiles(Map_Classifier):
    """Quantile Map Classification

    Parameters
    ----------
    y : array (n,1)
        values to classify
    k : int
        number of classes required

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations,
              each value is the id of the class the observation belongs to
              yb[i] = j  for j>=1  if bins[j-1] < y[i] <= bins[j], yb[i] = 0  otherwise
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class


    Examples
    --------
    >>> cal = load_example()
    >>> q = Quantiles(cal, k = 5)
    >>> q.bins
    array([  1.46400000e+00,   5.79800000e+00,   1.32780000e+01,
             5.46160000e+01,   4.11145000e+03])
    >>> q.counts
    array([12, 11, 12, 11, 12])
    >>>
    """

    def __init__(self, y, k=K):
        self.k = k
        Map_Classifier.__init__(self, y)
        self.name = 'Quantiles'

    def _set_bins(self):
        y = self.y
        k = self.k
        self.bins = quantile(y, k=k)


class Std_Mean(Map_Classifier):
    """
    Standard Deviation and Mean Map Classification

    Parameters
    ----------
    y         : array (n,1)
                values to classify
    multiples : array
                the multiples of the standard deviation to add/subtract from
                the sample mean to define the bins, default=[-2,-1,1,2]

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations,
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class


    Examples
    --------
    >>> cal = load_example()
    >>> st = Std_Mean(cal)
    >>> st.k
    5
    >>> st.bins
    array([ -967.36235382,  -420.71712519,   672.57333208,  1219.21856072,
            4111.45      ])
    >>> st.counts
    array([ 0,  0, 56,  1,  1])
    >>>
    >>> st3 = Std_Mean(cal, multiples = [-3, -1.5, 1.5, 3])
    >>> st3.bins
    array([-1514.00758246,  -694.03973951,   945.8959464 ,  1765.86378936,
            4111.45      ])
    >>> st3.counts
    array([ 0,  0, 57,  0,  1])
    >>>

    """
    def __init__(self, y, multiples=[-2, -1, 1, 2]):
        self.multiples = multiples
        Map_Classifier.__init__(self, y)
        self.name = 'Std_Mean'

    def _set_bins(self):
        y = self.y
        s = y.std(ddof=1)
        m = y.mean()
        cuts = [m + s * w for w in self.multiples]
        y_max = y.max()
        if cuts[-1] < y_max:
            cuts.append(y_max)
        self.bins = np.array(cuts)
        self.k = len(cuts)


class Maximum_Breaks(Map_Classifier):
    """
    Maximum Breaks Map Classification

    Parameters
    ----------
    y  : array (n x 1)
         values to classify

    k  : int
         number of classes required

    Attributes
    ----------
    yb : array (nx1)
         bin ids for observations

    bins : array (kx1)
           the upper bounds of each class

    k    : int
           the number of classes

    counts : array (kx1)
             the number of observations falling in each class (numpy array k x 1)

    Examples
    --------
    >>> cal = load_example()
    >>> mb = Maximum_Breaks(cal, k = 5)
    >>> mb.k
    5
    >>> mb.bins
    array([  146.005,   228.49 ,   546.675,  2417.15 ,  4111.45 ])
    >>> mb.counts
    array([50,  2,  4,  1,  1])
    >>>
    """
    def __init__(self, y, k=K, mindiff=0):
        self.k = k
        self.mindiff = mindiff
        Map_Classifier.__init__(self, y)
        self.name = 'Maximum_Breaks'

    def _set_bins(self):
        xs = self.y.copy()
        y = self.y.copy()
        k = self.k
        xs.sort()
        min_diff = self.mindiff
        d = xs[1:] - xs[:-1]
        diffs = d[np.nonzero(d > min_diff)]
        diffs = sp.unique(diffs)
        k1 = k - 1
        if len(diffs) > k1:
            diffs = diffs[-k1:]
        mp = []
        self.cids = []
        for diff in diffs:
            ids = np.nonzero(d == diff)
            for id in ids:
                self.cids.append(id[0])
                cp = ((xs[id] + xs[id + 1]) / 2.)
                mp.append(cp[0])
        mp.append(xs[-1])
        mp.sort()
        self.bins = np.array(mp)


class Natural_Breaks(Map_Classifier):
    """
    Natural Breaks Map Classification

    Parameters
    ----------
    y       : array (n,1)
              values to classify
    k       : int
              number of classes required
    initial : int (default=100)
              number of initial solutions to generate

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations,
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> cal = load_example()
    >>> nb = Natural_Breaks(cal, k = 5)
    >>> nb.k
    5
    >>> nb.counts
    array([14, 13, 14, 10,  7])
    >>> nb.bins
    [1.8100000000000001, 7.5999999999999996, 29.82, 181.27000000000001, 4111.4499999999998]
    >>> x = np.array([1] * 50)
    >>> x[-1] = 20
    >>> nb = Natural_Breaks(x, k = 5, initial = 0)
    Warning: Not enough unique values in array to form k classes
    Warning: setting k to 2
    >>> nb.bins
    [1, 20]
    >>> nb.counts
    array([49,  1])


    Notes
    -----
    There is a tradeoff here between speed and consistency of the classification
    If you want more speed, set initial to a smaller value (0
    would result in the best speed, if you want more consistent classes in
    multiple runs of Natural_Breaks on the same data, set initial to a higer
    value.


    """
    def __init__(self, y, k=K, initial=100):
        self.k = k
        self.initial = initial
        Map_Classifier.__init__(self, y)
        self.name = 'Natural_Breaks'

    def _set_bins(self):

        x = self.y.copy()
        k = self.k
        res0 = natural_breaks(x, k)
        fit = res0[2].sum()
        for i in xrange(self.initial):
            res = natural_breaks(x, k)
            fit_i = res[2].sum()
            if fit_i < fit:
                res0 = res
        self.bins = res0[-1]
        self.k = len(self.bins)
        self.iterations = res0[-2]


class Fisher_Jenks(Map_Classifier):
    """
    Fisher Jenks optimal classifier - mean based

    Parameters
    ----------
    y : array (n,1)
        values to classify
    k : int
        number of classes required

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class


    Examples
    --------

    >>> cal = load_example()
    >>> fj = Fisher_Jenks(cal)
    >>> fj.adcm
    799.24000000000001
    >>> fj.bins
    [75.290000000000006, 192.05000000000001, 370.5, 722.85000000000002, 4111.45]
    >>> fj.counts
    array([49,  3,  4,  1,  1])
    >>>
    """

    def __init__(self, y, k=K):
        self.k = k
        Map_Classifier.__init__(self, y)
        self.name = "Fisher_Jenks"

    def _set_bins(self):
        x = self.y.copy()
        self.bins = _fisher_jenks_means(x, classes=self.k)[1:]


class Fisher_Jenks_Sampled(Map_Classifier):
    """
    Fisher Jenks optimal classifier - mean based using random sample

    Parameters
    ----------
    y      : array (n,1)
             values to classify
    k      : int
             number of classes required
    pct    : float
             The percentage of n that should form the sample
             If pct is specified such that n*pct > 1000, then pct = 1000./n,
             unless truncate is False
    truncate: binary (Default True)
             truncate pct in cases where pct * n > 1000.


    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class


    Examples
    --------
    # turned off due to timing being different across hardware
    #>>> import pysal
    #>>> import numpy as np
    #>>> import time
    #>>> x = np.arange(1000)
    #>>> t0=time.time();fj=pysal.esda.mapclassify.Fisher_Jenks(x,k=4);t1=time.time()
    #>>> t2=time.time();fjs=pysal.esda.mapclassify.Fisher_Jenks_Sampled(x,k=4);t3=time.time()
    #>>> print t1-t0
    #1.84405994415
    #>>> print t3-t2
    #0.0295069217682
    #>>> fj.tss
    #5208250.0
    #>>> fjs.tss
    #5337860.0
    #>>>
    """

    def __init__(self, y, k=K, pct=0.10, truncate=True):
        self.k = k
        n = y.size

        if (pct * n > 1000) and truncate:
            pct = 1000. / n
        ids = np.random.random_integers(0, n - 1, n * pct)
        yr = y[ids]
        yr[-1] = max(y)  # make sure we have the upper bound
        yr[0] = min(y)  # make sure we have the min
        self.original_y = y
        self.pct = pct
        self.yr = yr
        self.yr_n = yr.size
        Map_Classifier.__init__(self, yr)
        self.yb, self.counts = bin1d(y, self.bins)
        self.name = "Fisher_Jenks_Sampled"
        self.y = y
        self._summary()  # have to recalculate summary stats

    def _set_bins(self):
        fj = Fisher_Jenks(self.y, self.k)
        self.bins = fj.bins


class Jenks_Caspall(Map_Classifier):
    """
    Jenks Caspall  Map Classification

    Parameters
    ----------
    y : array (n,1)
        values to classify
    k : int
        number of classes required

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations,
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class


    Examples
    --------
    >>> cal = load_example()
    >>> jc = Jenks_Caspall(cal, k = 5)
    >>> jc.bins
    array([[  1.81000000e+00],
           [  7.60000000e+00],
           [  2.98200000e+01],
           [  1.81270000e+02],
           [  4.11145000e+03]])
    >>> jc.counts
    array([14, 13, 14, 10,  7])

    """
    def __init__(self, y, k=K):
        self.k = k
        Map_Classifier.__init__(self, y)
        self.name = "Jenks_Caspall"

    def _set_bins(self):
        x = self.y.copy()
        k = self.k
        # start with quantiles
        q = quantile(x, k)
        solving = True
        xb, cnts = bin1d(x, q)
        #class means
        if x.ndim == 1:
            x.shape = (x.size, 1)
        n, k = x.shape
        xm = [np.median(x[xb == i]) for i in np.unique(xb)]
        xb0 = xb.copy()
        q = xm
        it = 0
        rk = range(self.k)
        while solving:
            xb = np.zeros(xb0.shape, int)
            d = abs(x - q)
            xb = d.argmin(axis=1)
            if (xb0 == xb).all():
                solving = False
            else:
                xb0 = xb
            it += 1
            q = np.array([np.median(x[xb == i]) for i in rk])
        cuts = [max(x[xb == i]) for i in sp.unique(xb)]
        self.bins = np.array(cuts)
        self.iterations = it


class Jenks_Caspall_Sampled(Map_Classifier):
    """
    Jenks Caspall Map Classification using a random sample

    Parameters
    ----------

    y       : array (n,1)
              values to classify
    k       : int
              number of classes required
    pct     : float
              The percentage of n that should form the sample
              If pct is specified such that n*pct > 1000, then pct = 1000./n

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations,
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class


    Examples
    --------

    >>> cal = load_example()
    >>> x = np.random.random(100000)
    >>> jc = Jenks_Caspall(x)
    >>> jcs = Jenks_Caspall_Sampled(x)
    >>> jc.bins
    array([[ 0.19770952],
           [ 0.39695769],
           [ 0.59588617],
           [ 0.79716865],
           [ 0.99999425]])
    >>> jcs.bins
    array([[ 0.18877882],
           [ 0.39341638],
           [ 0.6028286 ],
           [ 0.80070925],
           [ 0.99999425]])
    >>> jc.counts
    array([19804, 20005, 19925, 20178, 20088])
    >>> jcs.counts
    array([18922, 20521, 20980, 19826, 19751])
    >>>

    # not for testing since we get different times on different hardware
    # just included for documentation of likely speed gains
    #>>> t1 = time.time(); jc = Jenks_Caspall(x); t2 = time.time()
    #>>> t1s = time.time(); jcs = Jenks_Caspall_Sampled(x); t2s = time.time()
    #>>> t2 - t1; t2s - t1s
    #1.8292930126190186
    #0.061631917953491211

    Notes
    -----
    This is intended for large n problems. The logic is to apply
    Jenks_Caspall to a random subset of the y space and then bin the
    complete vector y on the bins obtained from the subset. This would
    trade off some "accuracy" for a gain in speed.

    """

    def __init__(self, y, k=K, pct=0.10):
        self.k = k
        n = y.size
        if pct * n > 1000:
            pct = 1000. / n
        ids = np.random.random_integers(0, n - 1, n * pct)
        yr = y[ids]
        yr[0] = max(y)  # make sure we have the upper bound
        self.original_y = y
        self.pct = pct
        self.yr = yr
        self.yr_n = yr.size
        Map_Classifier.__init__(self, yr)
        self.yb, self.counts = bin1d(y, self.bins)
        self.name = "Jenks_Caspall_Sampled"
        self.y = y
        self._summary()  # have to recalculate summary stats

    def _set_bins(self):
        jc = Jenks_Caspall(self.y, self.k)
        self.bins = jc.bins
        self.iterations = jc.iterations


class Jenks_Caspall_Forced(Map_Classifier):
    """

    Jenks Caspall  Map Classification with forced movements

    Parameters
    ----------
    y : array (n,1)
        values to classify
    k : int
        number of classes required

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations,
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class


    Examples
    --------
    >>> cal = load_example()
    >>> jcf = Jenks_Caspall_Forced(cal, k = 5)
    >>> jcf.k
    5
    >>> jcf.bins
    array([[  1.34000000e+00],
           [  5.90000000e+00],
           [  1.67000000e+01],
           [  5.06500000e+01],
           [  4.11145000e+03]])
    >>> jcf.counts
    array([12, 12, 13,  9, 12])
    >>> jcf4 = Jenks_Caspall_Forced(cal, k = 4)
    >>> jcf4.k
    4
    >>> jcf4.bins
    array([[  2.51000000e+00],
           [  8.70000000e+00],
           [  3.66800000e+01],
           [  4.11145000e+03]])
    >>> jcf4.counts
    array([15, 14, 14, 15])
    >>>
    """
    def __init__(self, y, k=K):
        self.k = k
        Map_Classifier.__init__(self, y)
        self.name = "Jenks_Caspall_Forced"

    def _set_bins(self):
        x = self.y.copy()
        k = self.k
        q = quantile(x, k)
        solving = True
        xb, cnt = bin1d(x, q)
        #class means
        if x.ndim == 1:
            x.shape = (x.size, 1)
        n, tmp = x.shape
        xm = [x[xb == i].mean() for i in np.unique(xb)]
        xb0 = xb.copy()
        q = xm
        xbar = np.array([xm[xbi] for xbi in xb])
        xbar.shape = (n, 1)
        ss = x - xbar
        ss *= ss
        ss = sum(ss)
        maxk = k - 1
        down_moves = up_moves = 0
        solving = True
        it = 0
        while solving:
            # try upward moves first
            moving_up = True
            while moving_up:
                class_ids = sp.unique(xb)
                nk = [sum(xb == j) for j in class_ids]
                candidates = nk[:-1]
                i = 0
                up_moves = 0
                while candidates:
                    nki = candidates.pop(0)
                    if nki > 1:
                        ids = np.nonzero(xb == class_ids[i])
                        mover = max(ids[0])
                        tmp = xb.copy()
                        tmp[mover] = xb[mover] + 1
                        tm = [x[tmp == j].mean() for j in sp.unique(tmp)]
                        txbar = np.array([tm[xbi] for xbi in tmp])
                        txbar.shape = (n, 1)
                        tss = x - txbar
                        tss *= tss
                        tss = sum(tss)
                        if tss < ss:
                            xb = tmp
                            ss = tss
                            candidates = []
                            up_moves += 1
                    i += 1
                if not up_moves:
                    moving_up = False
            moving_down = True
            while moving_down:
                class_ids = sp.unique(xb)
                nk = [sum(xb == j) for j in class_ids]
                candidates = nk[1:]
                i = 1
                down_moves = 0
                while candidates:
                    nki = candidates.pop(0)
                    if nki > 1:
                        ids = np.nonzero(xb == class_ids[i])
                        mover = min(ids[0])
                        mover_class = xb[mover]
                        target_class = mover_class - 1
                        tmp = xb.copy()
                        tmp[mover] = target_class
                        tm = [x[tmp == j].mean() for j in sp.unique(tmp)]
                        txbar = np.array([tm[xbi] for xbi in tmp])
                        txbar.shape = (n, 1)
                        tss = x - txbar
                        tss *= tss
                        tss = sum(tss)
                        if tss < ss:
                            xb = tmp
                            ss = tss
                            candidates = []
                            down_moves += 1
                    i += 1
                if not down_moves:
                    moving_down = False
            if not up_moves and not down_moves:
                solving = False
            it += 1
        cuts = [max(x[xb == i]) for i in sp.unique(xb)]
        self.bins = np.array(cuts)
        self.iterations = it


class User_Defined(Map_Classifier):
    """
    User Specified Binning


    Parameters
    ----------
    y    : array (n,1)
           values to classify
    bins : array (k,1)
           upper bounds of classes (have to be monotically increasing)

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations,
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class


    Examples
    --------
    >>> cal = load_example()
    >>> bins = [20, max(cal)]
    >>> bins
    [20, 4111.4499999999998]
    >>> ud = User_Defined(cal, bins)
    >>> ud.bins
    array([   20.  ,  4111.45])
    >>> ud.counts
    array([37, 21])
    >>> bins = [20, 30]
    >>> ud = User_Defined(cal, bins)
    >>> ud.bins
    array([   20.  ,    30.  ,  4111.45])
    >>> ud.counts
    array([37,  4, 17])
    >>>


    Notes
    -----
    If upper bound of user bins does not exceed max(y) we append an
    additional bin.

    """

    def __init__(self, y, bins):
        if bins[-1] < max(y):
            bins.append(max(y))
        self.k = len(bins)
        self.bins = np.array(bins)
        self.y = y
        Map_Classifier.__init__(self, y)
        self.name = 'User Defined'

    def _set_bins(self):
        pass


class Max_P_Classifier(Map_Classifier):
    """
    Max_P Map Classification

    Based on Max_p regionalization algorithm

    Parameters
    ----------
    y       : array (n,1)
              values to classify
    k       : int
              number of classes required
    initial : int
              number of initial solutions to use prior to swapping

    Attributes
    ----------

    yb      : array (n,1)
              bin ids for observations,
    bins    : array (k,1)
              the upper bounds of each class
    k       : int
              the number of classes
    counts  : array (k,1)
              the number of observations falling in each class

    Examples
    --------
    >>> import pysal
    >>> cal = pysal.esda.mapclassify.load_example()
    >>> mp = pysal.Max_P_Classifier(cal)
    >>> mp.bins
    [8.6999999999999993, 16.699999999999999, 20.469999999999999, 66.260000000000005, 4111.4499999999998]
    >>> mp.counts
    array([29,  8,  1, 10, 10])

    """
    def __init__(self, y, k=K, initial=1000):
        self.k = k
        self.initial = initial
        Map_Classifier.__init__(self, y)
        self.name = "Max_P"

    def _set_bins(self):
        x = self.y.copy()
        k = self.k
        q = quantile(x, k)
        if x.ndim == 1:
            x.shape = (x.size, 1)
        n, tmp = x.shape
        x.sort(axis=0)
        # find best of initial solutions
        solution = 0
        best_tss = x.var() * x.shape[0]
        tss_all = np.zeros((self.initial, 1))
        while solution < self.initial:
            remaining = range(n)
            seeds = [np.nonzero(di == min(
                di))[0][0] for di in [np.abs(x - qi) for qi in q]]
            rseeds = np.random.permutation(range(k)).tolist()
            tmp = [remaining.remove(seed) for seed in seeds]
            self.classes = classes = []
            tmp = [classes.append([seed]) for seed in seeds]
            while rseeds:
                seed_id = rseeds.pop()
                current = classes[seed_id]
                growing = True
                while growing:
                    current = classes[seed_id]
                    low = current[0]
                    high = current[-1]
                    left = low - 1
                    right = high + 1
                    move_made = False
                    if left in remaining:
                        current.insert(0, left)
                        remaining.remove(left)
                        move_made = True
                    if right in remaining:
                        current.append(right)
                        remaining.remove(right)
                        move_made = True
                    if move_made:
                        classes[seed_id] = current
                    else:
                        growing = False
            tss = _fit(self.y, classes)
            tss_all[solution] = tss
            if tss < best_tss:
                best_solution = classes
                best_it = solution
                best_tss = tss
            solution += 1
        classes = best_solution
        self.best_it = best_it
        self.tss = best_tss
        self.a2c = a2c = {}
        self.tss_all = tss_all
        for r, cl in enumerate(classes):
            for a in cl:
                a2c[a] = r
        swapping = True
        it = 0
        while swapping:
            rseeds = np.random.permutation(range(k)).tolist()
            total_moves = 0
            while rseeds:
                id = rseeds.pop()
                growing = True
                total_moves = 0
                while growing:
                    target = classes[id]
                    left = target[0] - 1
                    right = target[-1] + 1
                    n_moves = 0
                    if left in a2c:
                        left_class = classes[a2c[left]]
                        if len(left_class) > 1:
                            a = left_class[-1]
                            if self._swap(left_class, target, a):
                                target.insert(0, a)
                                left_class.remove(a)
                                a2c[a] = id
                                n_moves += 1
                    if right in a2c:
                        right_class = classes[a2c[right]]
                        if len(right_class) > 1:
                            a = right_class[0]
                            if self._swap(right_class, target, a):
                                target.append(a)
                                right_class.remove(a)
                                n_moves += 1
                                a2c[a] = id
                    if not n_moves:
                        growing = False
                total_moves += n_moves
            if not total_moves:
                swapping = False
        xs = self.y.copy()
        xs.sort()
        self.bins = [xs[cl][-1] for cl in classes]

    def _ss(self, class_def):
        """calculates sum of squares for a class"""
        yc = self.y[class_def]
        css = yc - yc.mean()
        css *= css
        return sum(css)

    def _swap(self, class1, class2, a):
        """evaluate cost of moving a from class1 to class2"""
        ss1 = self._ss(class1)
        ss2 = self._ss(class2)
        tss1 = ss1 + ss2
        class1c = copy.copy(class1)
        class2c = copy.copy(class2)
        class1c.remove(a)
        class2c.append(a)
        ss1 = self._ss(class1c)
        ss2 = self._ss(class2c)
        tss2 = ss1 + ss2
        if tss1 < tss2:
            return False
        else:
            return True


def _fit(y, classes):
    """Calculate the total sum of squares for a vector y classified into
    classes

    Parameters
    ----------
    y : array, variable to be classified

    classes : array, integer values denoting class membership

    """
    tss = 0
    for class_def in classes:
        yc = y[class_def]
        css = yc - yc.mean()
        css *= css
        tss += sum(css)
    return tss

kmethods = {}
kmethods["Quantiles"] = Quantiles
kmethods["Fisher_Jenks"] = Fisher_Jenks
kmethods['Natural_Breaks'] = Natural_Breaks
kmethods['Maximum_Breaks'] = Maximum_Breaks


def gadf(y, method="Quantiles", maxk=15, pct=0.8):
    """
    Evaluate the Goodness of Absolute Deviation Fit of a Classifier
    Finds the minimum value of k for which gadf>pct

    Parameters
    ----------

    y      : array (nx1)
             values to be classified
    method : string
             Name of classifier ["Quantiles,"Fisher_Jenks","Maximum_Breaks",
             "Natural_Breaks"]
    maxk   : int
             maximum value of k to evaluate
    pct    : float
             The percentage of GADF to exceed


    Returns
    -------

    implicit : tuple
               first value is k, second value is instance of classifier at k,
               third is the pct obtained


    Examples
    --------
    >>> cal = load_example()
    >>> qgadf = gadf(cal)
    >>> qgadf[0]
    15
    >>> qgadf[-1]
    0.37402575909092828

    Quantiles fail to exceed 0.80 before 15 classes. If we lower the bar to
    0.2 we see quintiles as a result

    >>> qgadf2 = gadf(cal, pct = 0.2)
    >>> qgadf2[0]
    5
    >>> qgadf2[-1]
    0.21710231966462412
    >>>

    Notes
    -----

    The GADF is defined as:

        .. math::

            GADF = 1 - \sum_c \sum_{i \in c} |y_i - y_{c,med}|  / \sum_i |y_i - y_{med}|

        where :math:`y_{med}` is the global median and :math:`y_{c,med}` is
        the median for class :math:`c`.

    See Also
    --------
    K_classifiers
    """

    y = np.array(y)
    adam = (np.abs(y - np.median(y))).sum()
    for k in range(2, maxk + 1):
        cl = kmethods[method](y, k)
        gadf = 1 - cl.adcm / adam
        if gadf > pct:
            break
    return (k, cl, gadf)


class K_classifiers:
    """
    Evaluate all k-classifers and pick optimal based on k and GADF

    Parameters
    ----------
    y      : array (nx1)
             values to be classified
    pct    : float
             The percentage of GADF to exceed


    Attributes
    ----------
    best   :  instance of Map_Classifier
              the optimal classifer
    results : dictionary
              keys are classifier names, values are the Map_Classifier instances with the best pct for each classifer

    Examples
    --------

    >>> cal = load_example()
    >>> ks = K_classifiers(cal)
    >>> ks.best.name
    'Fisher_Jenks'
    >>> ks.best.k
    4
    >>> ks.best.gadf
    0.84810327199081048
    >>>

    Notes
    -----
    This can be used to suggest a classification scheme.

    See Also
    --------
    gadf

    """
    def __init__(self, y, pct=0.8):
        results = {}
        c = 0
        best = gadf(y, "Fisher_Jenks", maxk=len(y) - 1, pct=pct)
        pct0 = best[0]
        k0 = best[-1]
        keys = kmethods.keys()
        keys.remove("Fisher_Jenks")
        results["Fisher_Jenks"] = best
        for method in keys:
            results[method] = gadf(y, method, maxk=len(y) - 1, pct=pct)
            k1 = results[method][0]
            pct1 = results[method][-1]
            if (k1 < k0) or (k1 == k0 and pct0 < pct1):
                best = results[method]
                k0 = k1
                pct0 = pct1
        self.results = results
        self.best = best[1]


def fj(x, k=5):
    y = x.copy()
    y.sort()
    d = {}
    initial = opt_part(y)
    # d has key = number of groups
    # value: list of ids, list of group tss, group size
    split_id = [initial[0]]
    tss = initial[1:]  # left and right within tss
    sizes = [split_id - 1, len(y) - split_id]
    d[2] = [split_id, tss, sizes]
    return d


def opt_part(x):
    """
    Find optimal bi-partition of x values
    """

    n = len(x)
    tss = np.inf
    opt_i = -999
    for i in xrange(1, n):
        print i
        left = x[:i].var() * i
        right = x[i:].var() * (n - i)
        tss_i = left + right
        if tss_i < tss:
            opt_i = i
            tss = tss_i
            left_min = left
            right_min = right
    return (opt_i, tss, left_min, right_min)
