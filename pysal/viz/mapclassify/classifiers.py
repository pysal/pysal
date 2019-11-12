"""
A module of classification schemes for choropleth mapping.
"""

__author__ = "Sergio J. Rey"

__all__ = [
    "Map_Classifier",
    "quantile",
    "Box_Plot",
    "BoxPlot",
    "Equal_Interval",
    "EqualInterval",
    "Fisher_Jenks",
    "Fisher_Jenks_Sampled",
    "Jenks_Caspall",
    "Jenks_Caspall_Forced",
    "Jenks_Caspall_Sampled",
    "Max_P_Classifier",
    "Maximum_Breaks",
    "Natural_Breaks",
    "Quantiles",
    "Percentiles",
    "Std_Mean",
    "User_Defined",
    "gadf",
    "K_classifiers",
    "HeadTail_Breaks",
    "HeadTailBreaks",
    "CLASSIFIERS",
]

CLASSIFIERS = (
    "BoxPlot",
    "EqualInterval",
    "FisherJenks",
    "FisherJenksSampled",
    "HeadTailBreaks",
    "JenksCaspall",
    "JenksCaspallForced",
    "JenksCaspallSampled",
    "MaxP",
    "MaximumBreaks",
    "NaturalBreaks",
    "Quantiles",
    "Percentiles",
    "StdMean",
    "UserDefined",
)

K = 5  # default number of classes in any map scheme with this as an argument
SEEDRANGE = 1000000  # range for drawing random integers from for Natural Breaks

import numpy as np
import scipy.stats as stats
import scipy as sp
import copy
from sklearn.cluster import KMeans as KMEANS
from warnings import warn as Warn
from deprecated import deprecated

try:
    from numba import jit
except ImportError:

    def jit(func):
        return func


@deprecated(reason="use head_tail_breaks")
def headTail_breaks(values, cuts):
    """
    head tail breaks helper function
    """
    return head_tail_breaks(values, cuts)


def head_tail_breaks(values, cuts):
    """
    head tail breaks helper function
    """
    values = np.array(values)
    mean = np.mean(values)
    cuts.append(mean)
    if len(values) > 1:
        return headTail_breaks(values[values >= mean], cuts)
    return cuts


def quantile(y, k=4):
    """
    Calculates the quantiles for an array

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of quantiles

    Returns
    -------
    q         : array
                (n,1), quantile values

    Examples
    --------
    >>> import numpy as np
    >>> import pysal.viz.mapclassify as mc
    >>> x = np.arange(1000)
    >>> mc.classifiers.quantile(x)
    array([249.75, 499.5 , 749.25, 999.  ])
    >>> mc.classifiers.quantile(x, k = 3)
    array([333., 666., 999.])

    Note that if there are enough ties that the quantile values repeat, we
    collapse to pseudo quantiles in which case the number of classes will be
    less than k

    >>> x = [1.0] * 100
    >>> x.extend([3.0] * 40)
    >>> len(x)
    140
    >>> y = np.array(x)
    >>> mc.classifiers.quantile(y)
    array([1., 3.])
    """

    w = 100.0 / k
    p = np.arange(w, 100 + w, w)
    if p[-1] > 100.0:
        p[-1] = 100.0
    q = np.array([stats.scoreatpercentile(y, pct) for pct in p])
    q = np.unique(q)
    k_q = len(q)
    if k_q < k:
        Warn(
            "Warning: Not enough unique values in array to form k classes", UserWarning
        )
        Warn("Warning: setting k to %d" % k_q, UserWarning)
    return q


def binC(y, bins):
    """
    Bin categorical/qualitative data

    Parameters
    ----------
    y    : array
           (n,q), categorical values
    bins : array
           (k,1),  unique values associated with each bin

    Return
    ------
    b : array
        (n,q), bin membership, values between 0 and k-1

    Examples
    --------
    >>> import numpy as np
    >>> import pysal.viz.mapclassify as mc
    >>> np.random.seed(1)
    >>> x = np.random.randint(2, 8, (10, 3))
    >>> bins = list(range(2, 8))
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
    >>> y = mc.classifiers.binC(x, bins)
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
    """

    if np.ndim(y) == 1:
        k = 1
        n = np.shape(y)[0]
    else:
        n, k = np.shape(y)
    b = np.zeros((n, k), dtype="int")
    for i, bin in enumerate(bins):
        b[np.nonzero(y == bin)] = i

    # check for non-binned items and warn if needed
    vals = set(y.flatten())
    for val in vals:
        if val not in bins:
            Warn("value not in bin: {}".format(val), UserWarning)
            Warn("bins: {}".format(bins), UserWarning)

    return b


def bin(y, bins):
    """
    bin interval/ratio data

    Parameters
    ----------
    y : array
        (n,q), values to bin
    bins : array
           (k,1), upper bounds of each bin (monotonic)

    Returns
    -------
    b : array
        (n,q), values of values between 0 and k-1

    Examples
    --------
    >>> import numpy as np
    >>> import pysal.viz.mapclassify as mc
    >>> np.random.seed(1)
    >>> x = np.random.randint(2, 20, (10, 3))
    >>> bins = [10, 15, 20]
    >>> b = mc.classifiers.bin(x, bins)
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
    """
    if np.ndim(y) == 1:
        k = 1
        n = np.shape(y)[0]
    else:
        n, k = np.shape(y)
    b = np.zeros((n, k), dtype="int")
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
    Place values of a 1-d array into bins and determine counts of values in
    each bin

    Parameters
    ----------
    x : array
        (n, 1), values to bin
    bins : array
           (k,1), upper bounds of each bin (monotonic)

    Returns
    -------
    binIds : array
             1-d array of integer bin Ids

    counts : int
            number of elements of x falling in each bin

    Examples
    --------
    >>> import numpy as np
    >>> import pysal.viz.mapclassify as mc
    >>> x = np.arange(100, dtype = 'float')
    >>> bins = [25, 74, 100]
    >>> binIds, counts = mc.classifiers.bin1d(x, bins)
    >>> binIds
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    >>> counts
    array([26, 49, 25])
    """
    left = [-float("inf")]
    left.extend(bins[0:-1])
    right = bins
    cuts = list(zip(left, right))
    k = len(bins)
    binIds = np.zeros(x.shape, dtype="int")
    while cuts:
        k -= 1
        l, r = cuts.pop(-1)
        binIds += (x > l) * (x <= r) * k
    counts = np.bincount(binIds, minlength=len(bins))
    return (binIds, counts)


def load_example():
    """
    Helper function for doc tests
    """
    from .datasets import calemp

    return calemp.load()


def _kmeans(y, k=5, n_init=10):
    """
    Helper function to do k-means in one dimension

    Parameters
    ----------

    y       : array
              (n,1), values to classify
    k       : int
              number of classes to form

    n_init : int, default: 10
              number of initial  solutions. Best of initial results is returned.
    """

    y = y * 1.0  # KMEANS needs float or double dtype
    y.shape = (-1, 1)
    result = KMEANS(n_clusters=k, init="k-means++", n_init=n_init).fit(y)
    class_ids = result.labels_
    centroids = result.cluster_centers_
    binning = []
    for c in range(k):
        values = y[class_ids == c]
        binning.append([values.max(), len(values)])
    binning = np.array(binning)
    binning = binning[binning[:, 0].argsort()]
    cuts = binning[:, 0]

    y_cent = np.zeros_like(y)
    for c in range(k):
        y_cent[class_ids == c] = centroids[c]
    diffs = y - y_cent
    diffs *= diffs

    return class_ids, cuts, diffs.sum(), centroids


def natural_breaks(values, k=5, init=10):
    """
    natural breaks helper function

    Jenks natural breaks is kmeans in one dimension

    Parameters
    ----------

    values : array
             (n, 1) values to bin

    k : int
        Number of classes

    init: int, default:10
        Number of different solutions to obtain using different centroids. Best solution is returned.


    """
    values = np.array(values)
    uv = np.unique(values)
    uvk = len(uv)
    if uvk < k:
        Warn(
            "Warning: Not enough unique values in array to form k classes", UserWarning
        )
        Warn("Warning: setting k to %d" % uvk, UserWarning)
        k = uvk
    kres = _kmeans(values, k, n_init=init)
    sids = kres[-1]  # centroids
    fit = kres[-2]
    class_ids = kres[0]
    cuts = kres[1]
    return (sids, class_ids, fit, cuts)


@jit
def _fisher_jenks_means(values, classes=5, sort=True):
    """
    Jenks Optimal (Natural Breaks) algorithm implemented in Python.

    Notes
    -----
    The original Python code comes from here:
    http://danieljlewis.org/2010/06/07/jenks-natural-breaks-algorithm-in-python/
    and is based on a JAVA and Fortran code available here:
    https://stat.ethz.ch/pipermail/r-sig-geo/2006-March/000811.html

    Returns class breaks such that classes are internally homogeneous while
    assuring heterogeneity among classes.

    """
    if sort:
        values.sort()
    n_data = len(values)
    mat1 = np.zeros((n_data + 1, classes + 1), dtype=np.int32)
    mat2 = np.zeros((n_data + 1, classes + 1), dtype=np.float32)
    mat1[1, 1:] = 1
    mat2[2:, 1:] = np.inf

    v = np.float32(0)
    for l in range(2, len(values) + 1):
        s1 = np.float32(0)
        s2 = np.float32(0)
        w = np.float32(0)
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = np.float32(values[i3 - 1])
            s2 += val * val
            s1 += val
            w += np.float32(1)
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, classes + 1):
                    if mat2[l, j] >= (v + mat2[i4, j - 1]):
                        mat1[l, j] = i3
                        mat2[l, j] = v + mat2[i4, j - 1]
        mat1[l, 1] = 1
        mat2[l, 1] = v

    k = len(values)

    kclass = np.zeros(classes + 1, dtype=values.dtype)
    kclass[classes] = values[len(values) - 1]
    kclass[0] = values[0]
    for countNum in range(classes, 1, -1):
        pivot = mat1[k, countNum]
        id = int(pivot - 2)
        kclass[countNum - 1] = values[id]
        k = int(pivot - 1)
    return kclass


def _dep_message(original, replacement, when="2020-01-31", version="2.1.0"):
    msg = "Deprecated (%s): %s" % (version, original)
    msg += " is being renamed to %s." % replacement
    msg += " %s will be removed on %s." % (original, when)
    return msg


class DeprecationHelper(object):
    def __init__(self, new_target, message="Deprecated"):
        self.new_target = new_target
        self.message = message

    def _warn(self):
        from warnings import warn

        warn(self.message)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)


class MapClassifier(object):
    """
    Abstract class for all map classifications :cite:`Slocum_2009`

    For an array :math:`y` of :math:`n` values, a map classifier places each
    value :math:`y_i` into one of :math:`k` mutually exclusive and exhaustive
    classes.  Each classifer defines the classes based on different criteria,
    but in all cases the following hold for the classifiers in PySAL:

    .. math:: C_j^l < y_i \le C_j^u \  \forall  i \in C_j

    where :math:`C_j` denotes class :math:`j` which has lower bound
          :math:`C_j^l` and upper bound :math:`C_j^u`.

    Map Classifiers Supported

    * :class:`pysal.viz.mapclassify.classifiers.BoxPlot`
    * :class:`pysal.viz.mapclassify.classifiers.EqualInterval`
    * :class:`pysal.viz.mapclassify.classifiers.FisherJenks`
    * :class:`pysal.viz.mapclassify.classifiers.FisherJenksSampled`
    * :class:`pysal.viz.mapclassify.classifiers.HeadTailBreaks`
    * :class:`pysal.viz.mapclassify.classifiers.JenksCaspall`
    * :class:`pysal.viz.mapclassify.classifiers.JenksCaspallForced`
    * :class:`pysal.viz.mapclassify.classifiers.JenksCaspallSampled`
    * :class:`pysal.viz.mapclassify.classifiers.MaxP`
    * :class:`pysal.viz.mapclassify.classifiers.MaximumBreaks`
    * :class:`pysal.viz.mapclassify.classifiers.NaturalBreaks`
    * :class:`pysal.viz.mapclassify.classifiers.Quantiles`
    * :class:`pysal.viz.mapclassify.classifiers.Percentiles`
    * :class:`pysal.viz.mapclassify.classifiers.StdMean`
    * :class:`pysal.viz.mapclassify.classifiers.UserDefined`

    Utilities:

    In addition to the classifiers, there are several utility functions that
    can be used to evaluate the properties of a specific classifier,
    or for automatic selection of a classifier and
    number of classes.

    * :func:`pysal.viz.mapclassify.classifiers.gadf`
    * :class:`pysal.viz.mapclassify.classifiers.K_classifiers`

    """

    def __init__(self, y):
        y = np.asarray(y).flatten()
        self.name = "Map Classifier"
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

    def _update(self, data, *args, **kwargs):
        """
        The only thing that *should* happen in this function is
        1. input sanitization for pandas
        2. classification/reclassification.

        Using their __init__ methods, all classifiers can re-classify given
        different input parameters or additional data.

        If you've got a cleverer updating equation than the intial estimation
        equation, remove the call to self.__init__ below and replace it with
        the updating function.
        """
        if data is not None:
            data = np.asarray(data).flatten()
            data = np.append(data.flatten(), self.y)
        else:
            data = self.y
        self.__init__(data, *args, **kwargs)

    @classmethod
    def make(cls, *args, **kwargs):
        """
        Configure and create a classifier that will consume data and produce
        classifications, given the configuration options specified by this
        function.

        Note that this like a *partial application* of the relevant class
        constructor. `make` creates a function that returns classifications; it
        does not actually do the classification.

        If you want to classify data directly, use the appropriate class
        constructor, like Quantiles, Max_Breaks, etc.

        If you *have* a classifier object, but want to find which bins new data
        falls into, use find_bin.

        Parameters
        ----------
        *args           : required positional arguments
                          all positional arguments required by the classifier,
                          excluding the input data.
        rolling         : bool
                          a boolean configuring the outputted classifier to use
                          a rolling classifier rather than a new classifier for
                          each input. If rolling, this adds the current data to
                          all of the previous data in the classifier, and
                          rebalances the bins, like a running median
                          computation.
        return_object   : bool
                          a boolean configuring the outputted classifier to
                          return the classifier object or not
        return_bins     : bool
                          a boolean configuring the outputted classifier to
                          return the bins/breaks or not
        return_counts   : bool
                          a boolean configuring the outputted classifier to
                          return the histogram of objects falling into each bin
                          or not

        Returns
        -------
        A function that consumes data and returns their bins (and object,
        bins/breaks, or counts, if requested).

        Note
        ----
        This is most useful when you want to run a classifier many times
        with a given configuration, such as when classifying many columns of an
        array or dataframe using the same configuration.

        Examples
        --------
        >>> import pysal.lib as ps
        >>> import pysal.viz.mapclassify as mc
        >>> import geopandas as gpd
        >>> df = gpd.read_file(ps.examples.get_path('columbus.dbf'))
        >>> classifier = mc.Quantiles.make(k=9)
        >>> cl = df[['HOVAL', 'CRIME', 'INC']].apply(classifier)
        >>> cl["HOVAL"].values[:10]
        array([8, 7, 2, 4, 1, 3, 8, 5, 7, 8])
        >>> cl["CRIME"].values[:10]
        array([0, 1, 3, 4, 6, 2, 0, 5, 3, 4])
        >>> cl["INC"].values[:10]
        array([7, 8, 5, 0, 3, 5, 0, 3, 6, 4])
        >>> import pandas as pd; from numpy import linspace as lsp
        >>> data = [lsp(3,8,num=10), lsp(10, 0, num=10), lsp(-5, 15, num=10)]
        >>> data = pd.DataFrame(data).T
        >>> data
                  0          1          2
        0  3.000000  10.000000  -5.000000
        1  3.555556   8.888889  -2.777778
        2  4.111111   7.777778  -0.555556
        3  4.666667   6.666667   1.666667
        4  5.222222   5.555556   3.888889
        5  5.777778   4.444444   6.111111
        6  6.333333   3.333333   8.333333
        7  6.888889   2.222222  10.555556
        8  7.444444   1.111111  12.777778
        9  8.000000   0.000000  15.000000
        >>> data.apply(mc.Quantiles.make(rolling=True))
           0  1  2
        0  0  4  0
        1  0  4  0
        2  1  4  0
        3  1  3  0
        4  2  2  1
        5  2  1  2
        6  3  0  4
        7  3  0  4
        8  4  0  4
        9  4  0  4
        >>> dbf = ps.io.open(ps.examples.get_path('baltim.dbf'))
        >>> data = dbf.by_col_array('PRICE', 'LOTSZ', 'SQFT')
        >>> my_bins = [1, 10, 20, 40, 80]
        >>> cl = [mc.User_Defined.make(bins=my_bins)(a) for a in data.T]
        >>> len(cl)
        3
        >>> cl[0][:10]
        array([4, 5, 5, 5, 4, 4, 5, 4, 4, 5])
        """

        # only flag overrides return flag
        to_annotate = copy.deepcopy(kwargs)
        return_object = kwargs.pop("return_object", False)
        return_bins = kwargs.pop("return_bins", False)
        return_counts = kwargs.pop("return_counts", False)

        rolling = kwargs.pop("rolling", False)
        if rolling:
            #  just initialize a fake classifier
            data = list(range(10))
            cls_instance = cls(data, *args, **kwargs)
            #  and empty it, since we'll be using the update
            cls_instance.y = np.array([])
        else:
            cls_instance = None

        #  wrap init in a closure to make a consumer.
        #  Qc Na: "Objects/Closures are poor man's Closures/Objects"
        def classifier(data, cls_instance=cls_instance):
            if rolling:
                cls_instance.update(data, inplace=True, **kwargs)
                yb = cls_instance.find_bin(data)
            else:
                cls_instance = cls(data, *args, **kwargs)
                yb = cls_instance.yb
            outs = [yb, None, None, None]
            outs[1] = cls_instance if return_object else None
            outs[2] = cls_instance.bins if return_bins else None
            outs[3] = cls_instance.counts if return_counts else None
            outs = [a for a in outs if a is not None]
            if len(outs) == 1:
                return outs[0]
            else:
                return outs

        #  for debugging/jic, keep around the kwargs.
        #  in future, we might want to make this a thin class, so that we can
        #  set a custom repr. Call the class `Binner` or something, that's a
        #  pre-configured Classifier that just consumes data, bins it, &
        #  possibly updates the bins.
        classifier._options = to_annotate
        return classifier

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y       :   array
                    (n,1) array of data to classify
        inplace :   bool
                    whether to conduct the update in place or to return a copy
                    estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new

    def __str__(self):
        st = self._table_string()
        return st

    def __repr__(self):
        return self._table_string()

    def __call__(self, *args, **kwargs):
        """
        This will allow the classifier to be called like it's a function.

        Whether or not we want to make this be "find_bin" or "update" is a
        design decision.

        I like this as find_bin, since a classifier's job should be to classify
        the data given to it using the rules estimated from the `_classify()`
        function.
        """
        return self.find_bin(*args)

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

    def find_bin(self, x):
        """
        Sort input or inputs according to the current bin estimate

        Parameters
        ----------
        x       :   array or numeric
                    a value or array of values to fit within the estimated
                    bins

        Returns
        -------
        a bin index or array of bin indices that classify the input into one of
        the classifiers' bins.

        Note that this differs from similar functionality in
        numpy.digitize(x, classi.bins, right=True).

        This will always provide the closest bin, so data "outside" the classifier,
        above and below the max/min breaks, will be classified into the nearest bin.

        numpy.digitize returns k+1 for data greater than the greatest bin, but retains 0
        for data below the lowest bin.
        """
        x = np.asarray(x).flatten()
        right = np.digitize(x, self.bins, right=True)
        if right.max() == len(self.bins):
            right[right == len(self.bins)] = len(self.bins) - 1
        return right

    def plot(
        self,
        gdf,
        border_color="lightgrey",
        border_width=0.10,
        title=None,
        legend=False,
        cmap="YlGnBu",
        axis_on=True,
        legend_kwds={"loc": "lower right"},
        file_name=None,
        dpi=600,
        ax=None,
        legend_width=12,
        legend_decimal=3,
    ):
        """
        Plot Mapclassiifer
        NOTE: Requires matplotlib, and implicitly requires geopandas
        dataframe as input.

        Parameters
        ---------
        gdf           : geopandas geodataframe
                        Contains the geometry column for the choropleth map
        border_color  : string, optional
                        matplotlib color string to use for polygon border
                        (Default: lightgrey)
        border_width  : float, optional
                        width of polygon boarder
                        (Default: 0.10)
        title         : string, optional
                        Title of map
                        (Default: None)
        cmap          : string, optional
                        matplotlib color string for color map to fill polygons
                        (Default: YlGn)
        axis_on       : boolean, optional
                        Show coordinate axes (default True)
                        (Default: True)
        legend_kwds   : dict, optional
                        options for ax.legend()
                        (Default: {"loc": "lower right"})
        file_name     : string, optional
                        Name of file to save figure to.
                        (Default: None)
        dpi           : int, optional
                        Dots per inch for saved figure
                        (Default: 600)
        ax            : matplotlib axis, optional
                        axis on which to plot the choropleth.
                        (Default: None, so plots on the current figure)
        Returns
        -------
        f,ax        : tuple
                      matplotlib figure, axis on which the plot is made.


        Examples
        --------

        >>> import pysal.lib as lp
        >>> import geopandas
        >>> gdf = geopandas.read_file(lp.examples.get_path("columbus.shp"))
        >>> q5 = pysal.viz.mapclassify.Quantiles(gdf.crime)
        >>> q5.plot(gdf)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Mapclassify.plot depends on matplotlib.pyplot, and this was"
                "not able to be imported. \nInstall matplotlib to"
                "plot spatial classifier."
            )
        if ax is None:
            f = plt.figure()
            ax = plt.gca()
        else:
            f = plt.gcf()

        labels = [self.bins[ybi] for ybi in self.yb]
        ax = gdf.assign(_cl=labels).plot(
            column="_cl",
            ax=ax,
            cmap=cmap,
            edgecolor=border_color,
            linewidth=border_width,
            categorical=True,
            legend=legend,
            legend_kwds=legend_kwds,
        )
        ax_legend = ax.get_legend()
        if ax_legend:
            fmt = ".%df" % legend_decimal
            fmt = "%" + fmt
            largest = max([len(fmt % i) for i in self.bins])
            width = largest
            fmt = "%d.%df" % (width, legend_decimal)
            fmt = "%" + fmt
            print(fmt)

            def replace_legend_items(legend, mapping):
                for txt in legend.texts:
                    for k, v in mapping.items():
                        if txt.get_text() == str(k):
                            txt.set_text(v)

            label_map = dict(
                [(i, (fmt % value).rjust(largest)) for i, value in enumerate(labels)]
            )
            print(label_map)
            replace_legend_items(ax_legend, label_map)

        if not axis_on:
            ax.axis("off")
        if title:
            f.suptitle(title)
        if file_name:
            plt.savefig(file_name, dpi=dpi)
        return f, ax


msg = _dep_message("Map_Classifer", "MapClassifier")
Map_Classifier = DeprecationHelper(MapClassifier, message=msg)


class HeadTailBreaks(MapClassifier):
    """
    Head/tail Breaks Map Classification for Heavy-tailed Distributions

    Parameters
    ----------
    y       : array
              (n,1), values to classify

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import numpy as np
    >>> import pysal.viz.mapclassify as mc
    >>> np.random.seed(10)
    >>> cal = mc.load_example()
    >>> htb = mc.HeadTailBreaks(cal)
    >>> htb.k
    3
    >>> htb.counts
    array([50,  7,  1])
    >>> htb.bins
    array([ 125.92810345,  811.26      , 4111.45      ])
    >>> np.random.seed(123456)
    >>> x = np.random.lognormal(3, 1, 1000)
    >>> htb = mc.HeadTailBreaks(x)
    >>> htb.bins
    array([ 32.26204423,  72.50205622, 128.07150107, 190.2899093 ,
           264.82847377, 457.88157946, 576.76046949])
    >>> htb.counts
    array([695, 209,  62,  22,  10,   1,   1])

    Notes
    -----
    Head/tail Breaks is a relatively new classification method developed
    for data with a heavy-tailed distribution.

    Implementation based on contributions by Alessandra Sozzi <alessandra.sozzi@gmail.com>.

    For theoretical details see :cite:`Jiang_2013`.

    """

    def __init__(self, y):
        MapClassifier.__init__(self, y)
        self.name = "HeadTailBreaks"

    def _set_bins(self):

        x = self.y.copy()
        bins = []
        bins = head_tail_breaks(x, bins)
        self.bins = np.array(bins)
        self.k = len(self.bins)


msg = _dep_message("HeadTail_Breaks", "HeadTailBreaks")
HeadTail_Breaks = DeprecationHelper(HeadTailBreaks, message=msg)


class EqualInterval(MapClassifier):
    """
    Equal Interval Classification

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations,
              each value is the id of the class the observation belongs to
              yb[i] = j  for j>=1  if bins[j-1] < y[i] <= bins[j], yb[i] = 0
              otherwise
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> ei = mc.EqualInterval(cal, k = 5)
    >>> ei.k
    5
    >>> ei.counts
    array([57,  0,  0,  0,  1])
    >>> ei.bins
    array([ 822.394, 1644.658, 2466.922, 3289.186, 4111.45 ])

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
        MapClassifier.__init__(self, y)
        self.name = "Equal Interval"

    def _set_bins(self):
        y = self.y
        k = self.k
        max_y = max(y)
        min_y = min(y)
        rg = max_y - min_y
        width = rg * 1.0 / k
        cuts = np.arange(min_y + width, max_y + width, width)
        if len(cuts) > self.k:  # handle overshooting
            cuts = cuts[0:k]
        cuts[-1] = max_y
        bins = cuts.copy()
        self.bins = bins


msg = _dep_message("Equal_Interval", "EqualInterval")
Equal_Interval = DeprecationHelper(EqualInterval, message=msg)


class Percentiles(MapClassifier):
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
             the number of observations falling in each class
             (numpy array k x 1)

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> p = mc.Percentiles(cal)
    >>> p.bins
    array([1.357000e-01, 5.530000e-01, 9.365000e+00, 2.139140e+02,
           2.179948e+03, 4.111450e+03])
    >>> p.counts
    array([ 1,  5, 23, 23,  5,  1])
    >>> p2 = mc.Percentiles(cal, pct = [50, 100])
    >>> p2.bins
    array([   9.365, 4111.45 ])
    >>> p2.counts
    array([29, 29])
    >>> p2.k
    2
    """

    def __init__(self, y, pct=[1, 10, 50, 90, 99, 100]):
        self.pct = pct
        MapClassifier.__init__(self, y)
        self.name = "Percentiles"

    def _set_bins(self):
        y = self.y
        pct = self.pct
        self.bins = np.array([stats.scoreatpercentile(y, p) for p in pct])
        self.k = len(self.bins)

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y   :   array
                    (n,1) array of data to classify
        inplace :   bool
                    whether to conduct the update in place or to return a copy
                    estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"pct": kwargs.pop("pct", self.pct)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


class BoxPlot(MapClassifier):
    """
    BoxPlot Map Classification

    Parameters
    ----------
    y     : array
            attribute to classify
    hinge : float
            multiplier for IQR

    Attributes
    ----------
    yb : array
        (n,1), bin ids for observations
    bins : array
          (n,1), the upper bounds of each class  (monotonic)
    k : int
        the number of classes
    counts : array
             (k,1), the number of observations falling in each class
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

    If q[2]+hinge*IQR > max(y) there will only be 5 classes and no high
    outliers, otherwise, there will be 6 classes and at least one high
    outlier.

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> bp = mc.BoxPlot(cal)
    >>> bp.bins
    array([-5.287625e+01,  2.567500e+00,  9.365000e+00,  3.953000e+01,
            9.497375e+01,  4.111450e+03])
    >>> bp.counts
    array([ 0, 15, 14, 14,  6,  9])
    >>> bp.high_outlier_ids
    array([ 0,  6, 18, 29, 33, 36, 37, 40, 42])
    >>> cal[bp.high_outlier_ids].values
    array([ 329.92,  181.27,  370.5 ,  722.85,  192.05,  110.74, 4111.45,
            317.11,  264.93])
    >>> bx = mc.BoxPlot(np.arange(100))
    >>> bx.bins
    array([-49.5 ,  24.75,  49.5 ,  74.25, 148.5 ])

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
        MapClassifier.__init__(self, y)
        self.name = "Box Plot"

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
        self.k = len(bins)

    def _classify(self):
        MapClassifier._classify(self)
        self.low_outlier_ids = np.nonzero(self.yb == 0)[0]
        self.high_outlier_ids = np.nonzero(self.yb == 5)[0]

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y       :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"hinge": kwargs.pop("hinge", self.hinge)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


msg = _dep_message("Box_Plot", "BoxPlot")
Box_Plot = DeprecationHelper(BoxPlot, message=msg)


class Quantiles(MapClassifier):
    """
    Quantile Map Classification

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations,
              each value is the id of the class the observation belongs to
              yb[i] = j  for j>=1  if bins[j-1] < y[i] <= bins[j], yb[i] = 0
              otherwise
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> q = mc.Quantiles(cal, k = 5)
    >>> q.bins
    array([1.46400e+00, 5.79800e+00, 1.32780e+01, 5.46160e+01, 4.11145e+03])
    >>> q.counts
    array([12, 11, 12, 11, 12])
    """

    def __init__(self, y, k=K):
        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "Quantiles"

    def _set_bins(self):
        y = self.y
        k = self.k
        self.bins = quantile(y, k=k)


class StdMean(MapClassifier):
    """
    Standard Deviation and Mean Map Classification

    Parameters
    ----------
    y         : array
                (n,1), values to classify
    multiples : array
                the multiples of the standard deviation to add/subtract from
                the sample mean to define the bins, default=[-2,-1,1,2]

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> st = mc.StdMean(cal)
    >>> st.k
    5
    >>> st.bins
    array([-967.36235382, -420.71712519,  672.57333208, 1219.21856072,
           4111.45      ])
    >>> st.counts
    array([ 0,  0, 56,  1,  1])
    >>>
    >>> st3 = mc.StdMean(cal, multiples = [-3, -1.5, 1.5, 3])
    >>> st3.bins
    array([-1514.00758246,  -694.03973951,   945.8959464 ,  1765.86378936,
            4111.45      ])
    >>> st3.counts
    array([ 0,  0, 57,  0,  1])

    """

    def __init__(self, y, multiples=[-2, -1, 1, 2]):
        self.multiples = multiples
        MapClassifier.__init__(self, y)
        self.name = "StdMean"

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

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y   :   array
                    (n,1) array of data to classify
        inplace :   bool
                    whether to conduct the update in place or to return a copy
                    estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"multiples": kwargs.pop("multiples", self.multiples)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


msg = _dep_message("Std_Mean", "StdMean")
Std_Mean = DeprecationHelper(StdMean, message=msg)


class MaximumBreaks(MapClassifier):
    """
    Maximum Breaks Map Classification

    Parameters
    ----------
    y  : array
         (n, 1), values to classify

    k  : int
         number of classes required

    mindiff : float
              The minimum difference between class breaks

    Attributes
    ----------
    yb : array
         (n, 1), bin ids for observations
    bins : array
           (k, 1), the upper bounds of each class
    k    : int
           the number of classes
    counts : array
             (k, 1), the number of observations falling in each class (numpy
             array k x 1)

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> mb = mc.MaximumBreaks(cal, k = 5)
    >>> mb.k
    5
    >>> mb.bins
    array([ 146.005,  228.49 ,  546.675, 2417.15 , 4111.45 ])
    >>> mb.counts
    array([50,  2,  4,  1,  1])

    """

    def __init__(self, y, k=5, mindiff=0):
        self.k = k
        self.mindiff = mindiff
        MapClassifier.__init__(self, y)
        self.name = "MaximumBreaks"

    def _set_bins(self):
        xs = self.y.copy()
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
                cp = (xs[id] + xs[id + 1]) / 2.0
                mp.append(cp[0])
        mp.append(xs[-1])
        mp.sort()
        self.bins = np.array(mp)

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y   :   array
                    (n,1) array of data to classify
        inplace :   bool
                    whether to conduct the update in place or to return a copy
                    estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        kwargs.update({"mindiff": kwargs.pop("mindiff", self.mindiff)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


msg = _dep_message("Maximum_Breaks", "MaximumBreaks")
Maximum_Breaks = DeprecationHelper(MaximumBreaks, message=msg)


class NaturalBreaks(MapClassifier):
    """
    Natural Breaks Map Classification

    Parameters
    ----------
    y       : array
              (n,1), values to classify
    k       : int
              number of classes required

    initial : int, default: 10
              Number of initial solutions generated with different centroids. Best of initial results is returned.

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import numpy as np
    >>> import pysal.viz.mapclassify as mc
    >>> np.random.seed(123456)
    >>> cal = mc.load_example()
    >>> nb = mc.NaturalBreaks(cal, k=5)
    >>> nb.k
    5
    >>> nb.counts
    array([49,  3,  4,  1,  1])
    >>> nb.bins
    array([  75.29,  192.05,  370.5 ,  722.85, 4111.45])
    >>> x = np.array([1] * 50)
    >>> x[-1] = 20
    >>> nb = mc.NaturalBreaks(x, k = 5)

    Warning: Not enough unique values in array to form k classes
    Warning: setting k to 2

    >>> nb.bins
    array([ 1, 20])
    >>> nb.counts
    array([49,  1])

    """

    def __init__(self, y, k=K, initial=10):
        self.k = k
        self.init = initial
        MapClassifier.__init__(self, y)
        self.name = "NaturalBreaks"

    def _set_bins(self):

        x = self.y.copy()
        k = self.k
        values = np.array(x)
        uv = np.unique(values)
        uvk = len(uv)
        if uvk < k:
            ms = "Warning: Not enough unique values in array to form k classes"
            Warn(ms, UserWarning)
            Warn("Warning: setting k to %d" % uvk, UserWarning)
            k = uvk
            uv.sort()
            # we set the bins equal to the sorted unique values and ramp k
            # downwards. no need to call kmeans.
            self.bins = uv
            self.k = k
        else:
            res0 = natural_breaks(x, k, init=self.init)
            fit = res0[2]
            self.bins = np.array(res0[-1])
            self.k = len(self.bins)

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


msg = _dep_message("Natural_Breaks", "NaturalBreaks")
Natural_Breaks = DeprecationHelper(NaturalBreaks, message=msg)


class FisherJenks(MapClassifier):
    """
    Fisher Jenks optimal classifier - mean based

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> fj = mc.FisherJenks(cal)
    >>> fj.adcm
    799.24
    >>> fj.bins
    array([  75.29,  192.05,  370.5 ,  722.85, 4111.45])
    >>> fj.counts
    array([49,  3,  4,  1,  1])
    >>>
    """

    def __init__(self, y, k=K):

        nu = len(np.unique(y))
        if nu < k:
            raise ValueError("Fewer unique values than specified classes.")
        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "FisherJenks"

    def _set_bins(self):
        x = self.y.copy()
        self.bins = np.array(_fisher_jenks_means(x, classes=self.k)[1:])


msg = _dep_message("Fisher_Jenks", "FisherJenks")
Fisher_Jenks = DeprecationHelper(FisherJenks, message=msg)


class FisherJenksSampled(MapClassifier):
    """
    Fisher Jenks optimal classifier - mean based using random sample

    Parameters
    ----------
    y      : array
             (n,1), values to classify
    k      : int
             number of classes required
    pct    : float
             The percentage of n that should form the sample
             If pct is specified such that n*pct > 1000, then
             pct = 1000./n, unless truncate is False
    truncate : boolean
               truncate pct in cases where pct * n > 1000., (Default True)

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------

    (Turned off due to timing being different across hardware)


    For theoretical details see :cite:`Rey_2016`.

    """

    def __init__(self, y, k=K, pct=0.10, truncate=True):
        self.k = k
        n = y.size

        if (pct * n > 1000) and truncate:
            pct = 1000.0 / n
        ids = np.random.random_integers(0, n - 1, int(n * pct))
        yr = y[ids]
        yr[-1] = max(y)  # make sure we have the upper bound
        yr[0] = min(y)  # make sure we have the min
        self.original_y = y
        self.pct = pct
        self._truncated = truncate
        self.yr = yr
        self.yr_n = yr.size
        MapClassifier.__init__(self, yr)
        self.yb, self.counts = bin1d(y, self.bins)
        self.name = "Fisher_Jenks_Sampled"
        self.y = y
        self._summary()  # have to recalculate summary stats

    def _set_bins(self):
        fj = Fisher_Jenks(self.y, self.k)
        self.bins = fj.bins

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        kwargs.update({"pct": kwargs.pop("pct", self.pct)})
        kwargs.update({"truncate": kwargs.pop("truncate", self._truncated)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


msg = _dep_message("Fisher_Jenks_Sampled", "FisherJenksSampled")
Fisher_Jenks_Sampled = DeprecationHelper(FisherJenksSampled, message=msg)


class JenksCaspall(MapClassifier):
    """
    Jenks Caspall  Map Classification

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> jc = mc.JenksCaspall(cal, k = 5)
    >>> jc.bins
    array([1.81000e+00, 7.60000e+00, 2.98200e+01, 1.81270e+02, 4.11145e+03])
    >>> jc.counts
    array([14, 13, 14, 10,  7])

    """

    def __init__(self, y, k=K):
        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "JenksCaspall"

    def _set_bins(self):
        x = self.y.copy()
        k = self.k
        # start with quantiles
        q = quantile(x, k)
        solving = True
        xb, cnts = bin1d(x, q)
        # class means
        if x.ndim == 1:
            x.shape = (x.size, 1)
        n, k = x.shape
        xm = [np.median(x[xb == i]) for i in np.unique(xb)]
        xb0 = xb.copy()
        q = xm
        it = 0
        rk = list(range(self.k))
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
        cuts = np.array([max(x[xb == i]) for i in sp.unique(xb)])
        cuts.shape = (len(cuts),)
        self.bins = cuts
        self.iterations = it


msg = _dep_message("Jenks_Caspall", "JenksCaspall")
Jenks_Caspall = DeprecationHelper(JenksCaspall, message=msg)


class JenksCaspallSampled(MapClassifier):
    """
    Jenks Caspall Map Classification using a random sample

    Parameters
    ----------

    y       : array
              (n,1), values to classify
    k       : int
              number of classes required
    pct     : float
              The percentage of n that should form the sample
              If pct is specified such that n*pct > 1000, then pct = 1000./n

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> x = np.random.random(100000)
    >>> jc = mc.JenksCaspall(x)
    >>> jcs = mc.JenksCaspallSampled(x)
    >>> jc.bins
    array([0.1988721 , 0.39624334, 0.59441487, 0.79624357, 0.99999251])
    >>> jcs.bins
    array([0.20998558, 0.42112792, 0.62752937, 0.80543819, 0.99999251])
    >>> jc.counts
    array([19943, 19510, 19547, 20297, 20703])
    >>> jcs.counts
    array([21039, 20908, 20425, 17813, 19815])

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
            pct = 1000.0 / n
        ids = np.random.random_integers(0, n - 1, int(n * pct))
        yr = y[ids]
        yr[0] = max(y)  # make sure we have the upper bound
        self.original_y = y
        self.pct = pct
        self.yr = yr
        self.yr_n = yr.size
        MapClassifier.__init__(self, yr)
        self.yb, self.counts = bin1d(y, self.bins)
        self.name = "JenksCaspallSampled"
        self.y = y
        self._summary()  # have to recalculate summary stats

    def _set_bins(self):
        jc = JenksCaspall(self.y, self.k)
        self.bins = jc.bins
        self.iterations = jc.iterations

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        kwargs.update({"pct": kwargs.pop("pct", self.pct)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


msg = _dep_message("Jenks_Caspall_Sampled", "JenksCaspallSampled")
Jenks_Caspall_Sampled = DeprecationHelper(JenksCaspallSampled, message=msg)


class JenksCaspallForced(MapClassifier):
    """
    Jenks Caspall  Map Classification with forced movements

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> jcf = mc.JenksCaspallForced(cal, k = 5)
    >>> jcf.k
    5
    >>> jcf.bins
    array([[1.34000e+00],
           [5.90000e+00],
           [1.67000e+01],
           [5.06500e+01],
           [4.11145e+03]])
    >>> jcf.counts
    array([12, 12, 13,  9, 12])
    >>> jcf4 = mc.JenksCaspallForced(cal, k = 4)
    >>> jcf4.k
    4
    >>> jcf4.bins
    array([[2.51000e+00],
           [8.70000e+00],
           [3.66800e+01],
           [4.11145e+03]])
    >>> jcf4.counts
    array([15, 14, 14, 15])
    """

    def __init__(self, y, k=K):
        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "JenksCaspallForced"

    def _set_bins(self):
        x = self.y.copy()
        k = self.k
        q = quantile(x, k)
        solving = True
        xb, cnt = bin1d(x, q)
        # class means
        if x.ndim == 1:
            x.shape = (x.size, 1)
        n, tmp = x.shape
        xm = [x[xb == i].mean() for i in np.unique(xb)]
        q = xm
        xbar = np.array([xm[xbi] for xbi in xb])
        xbar.shape = (n, 1)
        ss = x - xbar
        ss *= ss
        ss = sum(ss)
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
        cuts = [max(x[xb == c]) for c in sp.unique(xb)]
        self.bins = np.array(cuts)
        self.iterations = it


msg = _dep_message("Jenks_Caspall_Forced", "JenksCaspallForced")
Jenks_Caspall_Forced = DeprecationHelper(JenksCaspallForced, message=msg)


class UserDefined(MapClassifier):
    """
    User Specified Binning

    Parameters
    ----------
    y    : array
           (n,1), values to classify
    bins : array
           (k,1), upper bounds of classes (have to be monotically increasing)

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> bins = [20, max(cal)]
    >>> bins
    [20, 4111.45]
    >>> ud = mc.UserDefined(cal, bins)
    >>> ud.bins
    array([  20.  , 4111.45])
    >>> ud.counts
    array([37, 21])
    >>> bins = [20, 30]
    >>> ud = mc.UserDefined(cal, bins)
    >>> ud.bins
    array([  20.  ,   30.  , 4111.45])
    >>> ud.counts
    array([37,  4, 17])

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
        MapClassifier.__init__(self, y)
        self.name = "UserDefined"

    def _set_bins(self):
        pass

    def _update(self, y=None, bins=None):
        if y is not None:
            if hasattr(y, "values"):
                y = y.values
            y = np.append(y.flatten(), self.y)
        else:
            y = self.y
        if bins is None:
            bins = self.bins
        self.__init__(y, bins)

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        bins = kwargs.pop("bins", self.bins)
        if inplace:
            self._update(y=y, bins=bins, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, bins, **kwargs)
            return new


msg = _dep_message("User_Defined", "UserDefined")
User_Defined = DeprecationHelper(UserDefined, message=msg)


class MaxP(MapClassifier):
    """
    MaxP Map Classification

    Based on Max-p regionalization algorithm

    Parameters
    ----------
    y       : array
              (n,1), values to classify
    k       : int
              number of classes required
    initial : int
              number of initial solutions to use prior to swapping

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> mp = mc.Max_P_Classifier(cal)
    >>> mp.bins
    array([   8.7 ,   16.7 ,   20.47,  110.74, 4111.45])

    >>> mp.counts
    array([29,  8,  1, 12,  8])
    """

    def __init__(self, y, k=K, initial=1000):
        self.k = k
        self.initial = initial
        MapClassifier.__init__(self, y)
        self.name = "MaxP"

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
            remaining = list(range(n))
            seeds = [
                np.nonzero(di == min(di))[0][0] for di in [np.abs(x - qi) for qi in q]
            ]
            rseeds = np.random.permutation(list(range(k))).tolist()
            [remaining.remove(seed) for seed in seeds]
            self.classes = classes = []
            [classes.append([seed]) for seed in seeds]
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
        while swapping:
            rseeds = np.random.permutation(list(range(k))).tolist()
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
        self.bins = np.array([xs[cl][-1] for cl in classes])

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

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"initial": kwargs.pop("initial", self.initial)})
        if inplace:
            self._update(y, bins, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, bins, **kwargs)
            return new


msg = _dep_message("Max_P_Classifier", "MaxP")
Max_P_Classifier = DeprecationHelper(MaxP, message=msg)


def _fit(y, classes):
    """Calculate the total sum of squares for a vector y classified into
    classes

    Parameters
    ----------
    y : array
        (n,1), variable to be classified

    classes : array
              (k,1), integer values denoting class membership

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
kmethods["FisherJenks"] = FisherJenks
kmethods["NaturalBreaks"] = NaturalBreaks
kmethods["MaximumBreaks"] = MaximumBreaks


def gadf(y, method="Quantiles", maxk=15, pct=0.8):
    """
    Evaluate the Goodness of Absolute Deviation Fit of a Classifier
    Finds the minimum value of k for which gadf>pct

    Parameters
    ----------

    y      : array
             (n, 1) values to be classified
    method : {'Quantiles, 'Fisher_Jenks', 'Maximum_Breaks', 'Natrual_Breaks'}
    maxk   : int
             maximum value of k to evaluate
    pct    : float
             The percentage of GADF to exceed

    Returns
    -------
    k : int
        number of classes
    cl : object
         instance of the classifier at k
    gadf : float
           goodness of absolute deviation fit

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> qgadf = mc.classifiers.gadf(cal)
    >>> qgadf[0]
    15
    >>> qgadf[-1]
    0.3740257590909283

    Quantiles fail to exceed 0.80 before 15 classes. If we lower the bar to
    0.2 we see quintiles as a result

    >>> qgadf2 = mc.classifiers.gadf(cal, pct = 0.2)
    >>> qgadf2[0]
    5
    >>> qgadf2[-1]
    0.21710231966462412
    >>>

    Notes
    -----

    The GADF is defined as:

        .. math::

            GADF = 1 - \sum_c \sum_{i \in c}
                   |y_i - y_{c,med}|  / \sum_i |y_i - y_{med}|

        where :math:`y_{med}` is the global median and :math:`y_{c,med}` is
        the median for class :math:`c`.

    See Also
    --------
    KClassifiers
    """

    y = np.array(y)
    adam = (np.abs(y - np.median(y))).sum()
    for k in range(2, maxk + 1):
        cl = kmethods[method](y, k)
        gadf = 1 - cl.adcm / adam
        if gadf > pct:
            break
    return (k, cl, gadf)


class KClassifiers(object):
    """
    Evaluate all k-classifers and pick optimal based on k and GADF

    Parameters
    ----------
    y      : array
             (n,1), values to be classified
    pct    : float
             The percentage of GADF to exceed

    Attributes
    ----------
    best   :  object
              instance of the optimal MapClassifier
    results : dictionary
              keys are classifier names, values are the MapClassifier
              instances with the best pct for each classifer

    Examples
    --------
    >>> import pysal.viz.mapclassify as mc
    >>> cal = mc.load_example()
    >>> ks = mc.classifiers.KClassifiers(cal)
    >>> ks.best.name
    'FisherJenks'
    >>> ks.best.k
    4
    >>> ks.best.gadf
    0.8481032719908105

    Notes
    -----
    This can be used to suggest a classification scheme.

    See Also
    --------
    gadf

    """

    def __init__(self, y, pct=0.8):
        results = {}
        best = gadf(y, "FisherJenks", maxk=len(y) - 1, pct=pct)
        pct0 = best[0]
        k0 = best[-1]
        keys = list(kmethods.keys())
        keys.remove("FisherJenks")
        results["FisherJenks"] = best
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


msg = _dep_message("K_classifiers", "KClassifiers")
K_classifiers = DeprecationHelper(KClassifiers, message=msg)
