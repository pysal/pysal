"""
Spatial autocorrelation for binary attributes

"""
__author__ = "Sergio J. Rey <srey@asu.edu> , Luc Anselin <luc.anselin@asu.edu>"

from ..weights.spatial_lag import lag_spatial
from .tabular import _univariate_handler
import numpy as np

__all__ = ['Join_Counts']

PERMUTATIONS = 999


class Join_Counts(object):
    """Binary Join Counts


    Parameters
    ----------

    y               : array
                      binary variable measured across n spatial units
    w               : W
                      spatial weights instance
    permutations    : int
                      number of random permutations for calculation of pseudo-p_values

    Attributes
    ----------
    y            : array
                   original variable
    w            : W
                   original w object
    permutations : int
                   number of permutations
    bb           : float
                   number of black-black joins
    ww           : float
                   number of white-white joins
    bw           : float
                   number of black-white joins
    J            : float
                   number of joins
    sim_bb       : array
                   (if permutations>0)
                   vector of bb values for permuted samples
    p_sim_bb     : array
                  (if permutations>0)
                   p-value based on permutations (one-sided)
                   null: spatial randomness
                   alternative: the observed bb is greater than under randomness
    mean_bb      : float
                   average of permuted bb values
    min_bb       : float
                   minimum of permuted bb values
    max_bb       : float
                   maximum of permuted bb values
    sim_bw       : array
                   (if permutations>0)
                   vector of bw values for permuted samples
    p_sim_bw     : array
                   (if permutations>0)
                   p-value based on permutations (one-sided)
                   null: spatial randomness
                   alternative: the observed bw is greater than under randomness
    mean_bw      : float
                   average of permuted bw values
    min_bw       : float
                   minimum of permuted bw values
    max_bw       : float
                   maximum of permuted bw values


    Examples
    --------

    Replicate example from anselin and rey

    >>> import numpy as np
    >>> w = pysal.lat2W(4, 4)
    >>> y = np.ones(16)
    >>> y[0:8] = 0
    >>> np.random.seed(12345)
    >>> jc = pysal.Join_Counts(y, w)
    >>> jc.bb
    10.0
    >>> jc.bw
    4.0
    >>> jc.ww
    10.0
    >>> jc.J
    24.0
    >>> len(jc.sim_bb)
    999
    >>> jc.p_sim_bb
    0.0030000000000000001
    >>> np.mean(jc.sim_bb)
    5.5465465465465469
    >>> np.max(jc.sim_bb)
    10.0
    >>> np.min(jc.sim_bb)
    0.0
    >>> len(jc.sim_bw)
    999
    >>> jc.p_sim_bw
    1.0
    >>> np.mean(jc.sim_bw)
    12.811811811811811
    >>> np.max(jc.sim_bw)
    24.0
    >>> np.min(jc.sim_bw)
    7.0
    >>>
    """
    def __init__(self, y, w, permutations=PERMUTATIONS):
        y = np.asarray(y).flatten()
        w.transformation = 'b'  # ensure we have binary weights
        self.w = w
        self.y = y
        self.permutations = permutations
        self.J = w.s0 / 2.
        self.bb, self.ww, self.bw = self.__calc(self.y)

        if permutations:
            sim = [self.__calc(np.random.permutation(self.y))
                   for i in xrange(permutations)]
            sim_jc = np.array(sim)
            self.sim_bb = sim_jc[:, 0]
            self.min_bb = np.min(self.sim_bb)
            self.mean_bb = np.mean(self.sim_bb)
            self.max_bb = np.max(self.sim_bb)
            self.sim_bw = sim_jc[:, 2]
            self.min_bw = np.min(self.sim_bw)
            self.mean_bw = np.mean(self.sim_bw)
            self.max_bw = np.max(self.sim_bw)
            p_sim_bb = self.__pseudop(self.sim_bb, self.bb)
            p_sim_bw = self.__pseudop(self.sim_bw, self.bw)
            self.p_sim_bb = p_sim_bb
            self.p_sim_bw = p_sim_bw

    def __calc(self, z):
        zl = lag_spatial(self.w, z)
        bb = sum(z * zl) / 2.0
        zw = 1 - z
        zl = lag_spatial(self.w, zw)
        ww = sum(zw * zl) / 2.0
        bw = self.J - (bb + ww)
        return (bb, ww, bw)

    def __pseudop(self, sim, jc):
        above = sim >= jc
        larger = sum(above)
        psim = (larger + 1.) / (self.permutations + 1.)
        return psim

    @property
    def _statistic(self):
        return self.bw
    
    @classmethod
    def by_col(cls, df, cols, w=None, inplace=False, pvalue='sim', outvals=None, **stat_kws):
        """ 
        Function to compute a Join_Count statistic on a dataframe

        Arguments
        ---------
        df          :   pandas.DataFrame
                        a pandas dataframe with a geometry column
        cols        :   string or list of string
                        name or list of names of columns to use to compute the statistic
        w           :   pysal weights object
                        a weights object aligned with the dataframe. If not provided, this
                        is searched for in the dataframe's metadata
        inplace     :   bool
                        a boolean denoting whether to operate on the dataframe inplace or to
                        return a series contaning the results of the computation. If
                        operating inplace, the derived columns will be named
                        'column_join_count'
        pvalue      :   string
                        a string denoting which pvalue should be returned. Refer to the
                        the Join_Count statistic's documentation for available p-values
        outvals     :   list of strings
                        list of arbitrary attributes to return as columns from the 
                        Join_Count statistic
        **stat_kws  :   keyword arguments
                        options to pass to the underlying statistic. For this, see the
                        documentation for the Join_Count statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
        returns a copy of the dataframe with the relevant columns attached.

        See Also
        ---------
        For further documentation, refer to the Join_Count class in pysal.esda
        """
        if outvals is None:
            outvals = []
            outvals.extend(['bb', 'p_sim_bw', 'p_sim_bb'])
            pvalue = ''
        return _univariate_handler(df, cols, w=w, inplace=inplace, pvalue=pvalue, 
                                   outvals=outvals, stat=cls,
                                   swapname='bw', **stat_kws)
