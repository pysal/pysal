"""
Spatial diagnostics module
"""
__author__ = "Luc Anselin luc.anselin@asu.edu, Daniel Arribas-Bel darribas@asu.edu"

from utils import spdot
from scipy.stats.stats import chisqprob
from scipy.stats import norm
import numpy as np
import numpy.linalg as la

__all__ = ['LMtests', 'MoranRes', 'AKtest']


class LMtests:

    """
    Lagrange Multiplier tests. Implemented as presented in Anselin et al.
    (1996) [Anselin1996a]_

    ...

    Attributes
    ----------

    ols         : OLS
                  OLS regression object
    w           : W
                  Spatial weights instance
    tests       : list
                  Lists of strings with the tests desired to be performed.
                  Values may be:

                   * 'all': runs all the options (default)
                   * 'lme': LM error test
                   * 'rlme': Robust LM error test
                   * 'lml' : LM lag test
                   * 'rlml': Robust LM lag test

    Parameters
    ----------

    lme         : tuple
                  (Only if 'lme' or 'all' was in tests). Pair of statistic and
                  p-value for the LM error test.
    lml         : tuple
                  (Only if 'lml' or 'all' was in tests). Pair of statistic and
                  p-value for the LM lag test.
    rlme        : tuple
                  (Only if 'rlme' or 'all' was in tests). Pair of statistic
                  and p-value for the Robust LM error test.
    rlml        : tuple
                  (Only if 'rlml' or 'all' was in tests). Pair of statistic
                  and p-value for the Robust LM lag test.
    sarma       : tuple
                  (Only if 'rlml' or 'all' was in tests). Pair of statistic
                  and p-value for the SARMA test.

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from ols import OLS

    Open the csv file to access the data for analysis

    >>> csv = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')

    Pull out from the csv the files we need ('HOVAL' as dependent as well as
    'INC' and 'CRIME' as independent) and directly transform them into nx1 and
    nx2 arrays, respectively

    >>> y = np.array([csv.by_col('HOVAL')]).T
    >>> x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T

    Create the weights object from existing .gal file

    >>> w = pysal.open(pysal.examples.get_path('columbus.gal'), 'r').read()

    Row-standardize the weight object (not required although desirable in some
    cases)

    >>> w.transform='r'

    Run an OLS regression

    >>> ols = OLS(y, x)

    Run all the LM tests in the residuals. These diagnostics test for the
    presence of remaining spatial autocorrelation in the residuals of an OLS
    model and give indication about the type of spatial model. There are five
    types: presence of a spatial lag model (simple and robust version),
    presence of a spatial error model (simple and robust version) and joint presence
    of both a spatial lag as well as a spatial error model.

    >>> lms = pysal.spreg.diagnostics_sp.LMtests(ols, w)

    LM error test:

    >>> print round(lms.lme[0],4), round(lms.lme[1],4)
    3.0971 0.0784

    LM lag test:

    >>> print round(lms.lml[0],4), round(lms.lml[1],4)
    0.9816 0.3218

    Robust LM error test:

    >>> print round(lms.rlme[0],4), round(lms.rlme[1],4)
    3.2092 0.0732

    Robust LM lag test:

    >>> print round(lms.rlml[0],4), round(lms.rlml[1],4)
    1.0936 0.2957

    LM SARMA test:

    >>> print round(lms.sarma[0],4), round(lms.sarma[1],4)
    4.1907 0.123
    """

    def __init__(self, ols, w, tests=['all']):
        cache = spDcache(ols, w)
        if tests == ['all']:
            tests = ['lme', 'lml', 'rlme', 'rlml', 'sarma']
        if 'lme' in tests:
            self.lme = lmErr(ols, w, cache)
        if 'lml' in tests:
            self.lml = lmLag(ols, w, cache)
        if 'rlme' in tests:
            self.rlme = rlmErr(ols, w, cache)
        if 'rlml' in tests:
            self.rlml = rlmLag(ols, w, cache)
        if 'sarma' in tests:
            self.sarma = lmSarma(ols, w, cache)


class MoranRes:

    """
    Moran's I for spatial autocorrelation in residuals from OLS regression

    ...

    Parameters
    ----------

    ols         : OLS
                  OLS regression object
    w           : W
                  Spatial weights instance
    z           : boolean
                  If set to True computes attributes eI, vI and zI. Due to computational burden of vI, defaults to False.

    Attributes
    ----------
    I           : float
                  Moran's I statistic
    eI          : float
                  Moran's I expectation
    vI          : float
                  Moran's I variance
    zI          : float
                  Moran's I standardized value

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from ols import OLS

    Open the csv file to access the data for analysis

    >>> csv = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')

    Pull out from the csv the files we need ('HOVAL' as dependent as well as
    'INC' and 'CRIME' as independent) and directly transform them into nx1 and
    nx2 arrays, respectively

    >>> y = np.array([csv.by_col('HOVAL')]).T
    >>> x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T

    Create the weights object from existing .gal file

    >>> w = pysal.open(pysal.examples.get_path('columbus.gal'), 'r').read()

    Row-standardize the weight object (not required although desirable in some
    cases)

    >>> w.transform='r'

    Run an OLS regression

    >>> ols = OLS(y, x)

    Run Moran's I test for residual spatial autocorrelation in an OLS model.
    This computes the traditional statistic applying a correction in the
    expectation and variance to account for the fact it comes from residuals
    instead of an independent variable

    >>> m = pysal.spreg.diagnostics_sp.MoranRes(ols, w, z=True)

    Value of the Moran's I statistic:

    >>> print round(m.I,4)
    0.1713

    Value of the Moran's I expectation:

    >>> print round(m.eI,4)
    -0.0345

    Value of the Moran's I variance:

    >>> print round(m.vI,4)
    0.0081

    Value of the Moran's I standardized value. This is
    distributed as a standard Normal(0, 1)

    >>> print round(m.zI,4)
    2.2827

    P-value of the standardized Moran's I value (z):

    >>> print round(m.p_norm,4)
    0.0224
    """

    def __init__(self, ols, w, z=False):
        cache = spDcache(ols, w)
        self.I = get_mI(ols, w, cache)
        if z:
            self.eI = get_eI(ols, w, cache)
            self.vI = get_vI(ols, w, self.eI, cache)
            self.zI, self.p_norm = get_zI(self.I, self.eI, self.vI)


class AKtest:

    """
    Moran's I test of spatial autocorrelation for IV estimation.
    Implemented following the original reference Anselin and Kelejian
    (1997) [Anselin1997]_
    ...

    Parameters
    ----------

    iv          : TSLS
                  Regression object from TSLS class
    w           : W
                  Spatial weights instance
    case        : string
                  Flag for special cases (default to 'nosp'):

                   * 'nosp': Only NO spatial end. reg.
                   * 'gen': General case (spatial lag + end. reg.)

    Attributes
    ----------

    mi          : float
                  Moran's I statistic for IV residuals
    ak          : float
                  Square of corrected Moran's I for residuals::

                  .. math::

                        ak = \dfrac{N \times I^*}{\phi^2}

                  Note: if case='nosp' then it simplifies to the LMerror
    p           : float
                  P-value of the test

    Examples
    --------

    We first need to import the needed modules. Numpy is needed to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis. The TSLS is required to run the model on
    which we will perform the tests.

    >>> import numpy as np
    >>> import pysal
    >>> from twosls import TSLS
    >>> from twosls_sp import GM_Lag

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')

    Before being able to apply the diagnostics, we have to run a model and,
    for that, we need the input variables. Extract the CRIME column (crime
    rates) from the DBF file and make it the dependent variable for the
    regression. Note that PySAL requires this to be an numpy array of shape
    (n, 1) as opposed to the also common shape of (n, ) that other packages
    accept.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T

    In this case, we consider HOVAL (home value) as an endogenous regressor,
    so we acknowledge that by reading it in a different category.

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T

    In order to properly account for the endogeneity, we have to pass in the
    instruments. Let us consider DISCBD (distance to the CBD) is a good one:

    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    Now we are good to run the model. It is an easy one line task.

    >>> reg = TSLS(y, X, yd, q=q)

    Now we are concerned with whether our non-spatial model presents spatial
    autocorrelation in the residuals. To assess this possibility, we can run
    the Anselin-Kelejian test, which is a version of the classical LM error
    test adapted for the case of residuals from an instrumental variables (IV)
    regression. First we need an extra object, the weights matrix, which
    includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are good to run the test. It is a very simple task:

    >>> ak = AKtest(reg, w)

    And explore the information obtained:

    >>> print('AK test: %f\tP-value: %f'%(ak.ak, ak.p))
    AK test: 4.642895      P-value: 0.031182

    The test also accomodates the case when the residuals come from an IV
    regression that includes a spatial lag of the dependent variable. The only
    requirement needed is to modify the ``case`` parameter when we call
    ``AKtest``. First, let us run a spatial lag model:

    >>> reg_lag = GM_Lag(y, X, yd, q=q, w=w)

    And now we can run the AK test and obtain similar information as in the
    non-spatial model.

    >>> ak_sp = AKtest(reg, w, case='gen')
    >>> print('AK test: %f\tP-value: %f'%(ak_sp.ak, ak_sp.p))
    AK test: 1.157593      P-value: 0.281965

    """

    def __init__(self, iv, w, case='nosp'):
        if case == 'gen':
            cache = spDcache(iv, w)
            self.mi, self.ak, self.p = akTest(iv, w, cache)
        elif case == 'nosp':
            cache = spDcache(iv, w)
            self.mi = get_mI(iv, w, cache)
            self.ak, self.p = lmErr(iv, w, cache)
        else:
            print """\n
            Fix the optional argument 'case' to match the requirements:
                * 'gen': General case (spatial lag + end. reg.)
                * 'nosp': No spatial end. reg.
            \n"""


class spDcache:

    """
    Helper class to compute reusable pieces in the spatial diagnostics module
    ...

    Parameters
    ----------

    reg         : OLS_dev, TSLS_dev, STSLS_dev
                  Instance from a regression class
    w           : W
                  Spatial weights instance

    Attributes
    ----------

    j           : array
                  1x1 array with the result from:

                  .. math::

                        J = \dfrac{1}{[(WX\beta)' M (WX\beta) + T \sigma^2]}

    wu          : array
                  nx1 array with spatial lag of the residuals

    utwuDs      : array
                  1x1 array with the result from:

                  .. math::

                        utwuDs = \dfrac{u' W u}{\tilde{\sigma^2}}

    utwyDs      : array
                  1x1 array with the result from:

                  .. math::

                        utwyDs = \dfrac{u' W y}{\tilde{\sigma^2}}


    t           : array
                  1x1 array with the result from :

                  .. math::

                        T = tr[(W' + W) W]

    trA         : float
                  Trace of A as in Cliff & Ord (1981)

    """

    def __init__(self, reg, w):
        self.reg = reg
        self.w = w
        self._cache = {}

    @property
    def j(self):
        if 'j' not in self._cache:
            wxb = self.w.sparse * self.reg.predy
            wxb2 = np.dot(wxb.T, wxb)
            xwxb = spdot(self.reg.x.T, wxb)
            num1 = wxb2 - np.dot(xwxb.T, np.dot(self.reg.xtxi, xwxb))
            num = num1 + (self.t * self.reg.sig2n)
            den = self.reg.n * self.reg.sig2n
            self._cache['j'] = num / den
        return self._cache['j']

    @property
    def wu(self):
        if 'wu' not in self._cache:
            self._cache['wu'] = self.w.sparse * self.reg.u
        return self._cache['wu']

    @property
    def utwuDs(self):
        if 'utwuDs' not in self._cache:
            res = np.dot(self.reg.u.T, self.wu) / self.reg.sig2n
            self._cache['utwuDs'] = res
        return self._cache['utwuDs']

    @property
    def utwyDs(self):
        if 'utwyDs' not in self._cache:
            res = np.dot(self.reg.u.T, self.w.sparse * self.reg.y)
            self._cache['utwyDs'] = res / self.reg.sig2n
        return self._cache['utwyDs']

    @property
    def t(self):
        if 't' not in self._cache:
            prod = (self.w.sparse.T + self.w.sparse) * self.w.sparse
            self._cache['t'] = np.sum(prod.diagonal())
        return self._cache['t']

    @property
    def trA(self):
        if 'trA' not in self._cache:
            xtwx = spdot(self.reg.x.T, spdot(self.w.sparse, self.reg.x))
            mw = np.dot(self.reg.xtxi, xtwx)
            self._cache['trA'] = np.sum(mw.diagonal())
        return self._cache['trA']

    @property
    def AB(self):
        """
        Computes A and B matrices as in Cliff-Ord 1981, p. 203
        """
        if 'AB' not in self._cache:
            U = (self.w.sparse + self.w.sparse.T) / 2.
            z = spdot(U, self.reg.x, array_out=False)
            c1 = spdot(self.reg.x.T, z, array_out=False)
            c2 = spdot(z.T, z, array_out=False)
            G = self.reg.xtxi
            A = spdot(G, c1)
            B = spdot(G, c2)
            self._cache['AB'] = [A, B]
        return self._cache['AB']


def lmErr(reg, w, spDcache):
    """
    LM error test. Implemented as presented in eq. (9) of Anselin et al.
    (1996) [Anselin1996a]_
    ...

    Attributes
    ----------

    reg         : OLS_dev, TSLS_dev, STSLS_dev
                  Instance from a regression class
    w           : W
                  Spatial weights instance
    spDcache    : spDcache
                  Instance of spDcache class

    Returns
    -------

    lme         : tuple
                  Pair of statistic and p-value for the LM error test.

    """
    lm = spDcache.utwuDs ** 2 / spDcache.t
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def lmLag(ols, w, spDcache):
    """
    LM lag test. Implemented as presented in eq. (13) of Anselin et al.
    (1996) [Anselin1996a]_
    ...

    Attributes
    ----------

    ols         : OLS_dev
                  Instance from an OLS_dev regression
    w           : W
                  Spatial weights instance
    spDcache     : spDcache
                  Instance of spDcache class

    Returns
    -------

    lml         : tuple
                  Pair of statistic and p-value for the LM lag test.

    """
    lm = spDcache.utwyDs ** 2 / (ols.n * spDcache.j)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def rlmErr(ols, w, spDcache):
    """
    Robust LM error test. Implemented as presented in eq. (8) of Anselin et
    al. (1996) [Anselin1996a]_

    NOTE: eq. (8) has an errata, the power -1 in the denominator should be inside the square bracket.
    ...

    Attributes
    ----------

    ols         : OLS_dev
                  Instance from an OLS_dev regression
    w           : W
                  Spatial weights instance
    spDcache     : spDcache
                  Instance of spDcache class

    Returns
    -------

    rlme        : tuple
                  Pair of statistic and p-value for the Robust LM error test.

    """
    nj = ols.n * spDcache.j
    num = (spDcache.utwuDs - (spDcache.t * spDcache.utwyDs) / nj) ** 2
    den = spDcache.t * (1. - (spDcache.t / nj))
    lm = num / den
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def rlmLag(ols, w, spDcache):
    """
    Robust LM lag test. Implemented as presented in eq. (12) of Anselin et al.
    (1996) [Anselin1996a]_
    ...

    Attributes
    ----------

    ols             : OLS_dev
                      Instance from an OLS_dev regression
    w               : W
                      Spatial weights instance
    spDcache        : spDcache
                      Instance of spDcache class

    Returns
    -------

    rlml            : tuple
                      Pair of statistic and p-value for the Robust LM lag test.

    """
    lm = (spDcache.utwyDs - spDcache.utwuDs) ** 2 / \
        ((ols.n * spDcache.j) - spDcache.t)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def lmSarma(ols, w, spDcache):
    """
    LM error test. Implemented as presented in eq. (15) of Anselin et al.
    (1996) [Anselin1996a]_
    ...

    Attributes
    ----------

    ols         : OLS_dev
                  Instance from an OLS_dev regression
    w           : W
                  Spatial weights instance
    spDcache     : spDcache
                  Instance of spDcache class

    Returns
    -------

    sarma       : tuple
                  Pair of statistic and p-value for the LM sarma test.

    """

    first = (spDcache.utwyDs - spDcache.utwuDs) ** 2 / \
        (w.n * spDcache.j - spDcache.t)
    secnd = spDcache.utwuDs ** 2 / spDcache.t
    lm = first + secnd
    pval = chisqprob(lm, 2)
    return (lm[0][0], pval[0][0])


def get_mI(reg, w, spDcache):
    """
    Moran's I statistic of spatial autocorrelation as showed in Cliff & Ord
    (1981) [Cliff1981]_, p. 201-203
    ...

    Attributes
    ----------

    reg             : OLS_dev, TSLS_dev, STSLS_dev
                      Instance from a regression class
    w               : W
                      Spatial weights instance
    spDcache        : spDcache
                      Instance of spDcache class

    Returns
    -------

    moran           : float
                      Statistic Moran's I test.

    """
    mi = (w.n * np.dot(reg.u.T, spDcache.wu)) / (w.s0 * reg.utu)
    return mi[0][0]


def get_vI(ols, w, ei, spDcache):
    """
    Moran's I variance coded as in Cliff & Ord 1981 (p. 201-203) and R's spdep
    """
    A = spDcache.AB[0]
    trA2 = np.dot(A, A)
    trA2 = np.sum(trA2.diagonal())

    B = spDcache.AB[1]
    trB = np.sum(B.diagonal()) * 4.
    vi = (w.n ** 2 / (w.s0 ** 2 * (w.n - ols.k) * (w.n - ols.k + 2.))) * \
         (w.s1 + 2. * trA2 - trB -
          ((2. * (spDcache.trA ** 2)) / (w.n - ols.k)))
    return vi


def get_eI(ols, w, spDcache):
    """
    Moran's I expectation using matrix M
    """
    return - (w.n * spDcache.trA) / (w.s0 * (w.n - ols.k))


def get_zI(I, ei, vi):
    """
    Standardized I

    Returns two-sided p-values as provided in the GeoDa family
    """
    z = abs((I - ei) / np.sqrt(vi))
    pval = norm.sf(z) * 2.
    return (z, pval)


def akTest(iv, w, spDcache):
    """
    Computes AK-test for the general case (end. reg. + sp. lag)
    ...

    Parameters
    ----------

    iv          : STSLS_dev
                  Instance from spatial 2SLS regression
    w           : W
                  Spatial weights instance
   spDcache     : spDcache
                  Instance of spDcache class

    Attributes
    ----------
    mi          : float
                  Moran's I statistic for IV residuals
    ak          : float
                  Square of corrected Moran's I for residuals::

                  .. math::
                        ak = \dfrac{N \times I^*}{\phi^2}

    p           : float
                  P-value of the test

    ToDo:
        * Code in as Nancy
        * Compare both
    """
    mi = get_mI(iv, w, spDcache)
    # Phi2
    etwz = spdot(iv.u.T, spdot(w.sparse, iv.z))
    a = np.dot(etwz, np.dot(iv.varb, etwz.T))
    s12 = (w.s0 / w.n) ** 2
    phi2 = (spDcache.t + (4.0 / iv.sig2n) * a) / (s12 * w.n)
    ak = w.n * mi ** 2 / phi2
    pval = chisqprob(ak, 1)
    return (mi, ak[0][0], pval[0][0])


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
