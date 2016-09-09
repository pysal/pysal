
from __future__ import absolute_import, print_function
import numpy as np
import warnings


def _bit_length_26(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return len(bin(x)) - 2


try:
    from scipy.lib._version import NumpyVersion
except ImportError:
    import re
    string_types = basestring 

    class NumpyVersion():
        """Parse and compare numpy version strings.
        Numpy has the following versioning scheme (numbers given are examples; they
        can be >9) in principle):
        - Released version: '1.8.0', '1.8.1', etc.
        - Alpha: '1.8.0a1', '1.8.0a2', etc.
        - Beta: '1.8.0b1', '1.8.0b2', etc.
        - Release candidates: '1.8.0rc1', '1.8.0rc2', etc.
        - Development versions: '1.8.0.dev-f1234afa' (git commit hash appended)
        - Development versions after a1: '1.8.0a1.dev-f1234afa',
                                        '1.8.0b2.dev-f1234afa',
                                        '1.8.1rc1.dev-f1234afa', etc.
        - Development versions (no git hash available): '1.8.0.dev-Unknown'
        Comparing needs to be done against a valid version string or other
        `NumpyVersion` instance.
        Parameters
        ----------
        vstring : str
            Numpy version string (``np.__version__``).
        Notes
        -----
        All dev versions of the same (pre-)release compare equal.
        Examples
        --------
        >>> from scipy.lib._version import NumpyVersion
        >>> if NumpyVersion(np.__version__) < '1.7.0':
        ...     print('skip')
        skip
        >>> NumpyVersion('1.7')  # raises ValueError, add ".0"
        """

        def __init__(self, vstring):
            self.vstring = vstring
            ver_main = re.match(r'\d[.]\d+[.]\d+', vstring)
            if not ver_main:
                raise ValueError("Not a valid numpy version string")

            self.version = ver_main.group()
            self.major, self.minor, self.bugfix = [int(x) for x in
                                                   self.version.split('.')]
            if len(vstring) == ver_main.end():
                self.pre_release = 'final'
            else:
                alpha = re.match(r'a\d', vstring[ver_main.end():])
                beta = re.match(r'b\d', vstring[ver_main.end():])
                rc = re.match(r'rc\d', vstring[ver_main.end():])
                pre_rel = [m for m in [alpha, beta, rc] if m is not None]
                if pre_rel:
                    self.pre_release = pre_rel[0].group()
                else:
                    self.pre_release = ''

            self.is_devversion = bool(re.search(r'.dev-', vstring))

        def _compare_version(self, other):
            """Compare major.minor.bugfix"""
            if self.major == other.major:
                if self.minor == other.minor:
                    if self.bugfix == other.bugfix:
                        vercmp = 0
                    elif self.bugfix > other.bugfix:
                        vercmp = 1
                    else:
                        vercmp = -1
                elif self.minor > other.minor:
                    vercmp = 1
                else:
                    vercmp = -1
            elif self.major > other.major:
                vercmp = 1
            else:
                vercmp = -1

            return vercmp

        def _compare_pre_release(self, other):
            """Compare alpha/beta/rc/final."""
            if self.pre_release == other.pre_release:
                vercmp = 0
            elif self.pre_release == 'final':
                vercmp = 1
            elif other.pre_release == 'final':
                vercmp = -1
            elif self.pre_release > other.pre_release:
                vercmp = 1
            else:
                vercmp = -1

            return vercmp

        def _compare(self, other):
            if not isinstance(other, (string_types, NumpyVersion)):
                raise ValueError("Invalid object to compare with NumpyVersion.")

            if isinstance(other, string_types):
                other = NumpyVersion(other)

            vercmp = self._compare_version(other)
            if vercmp == 0:
                # Same x.y.z version, check for alpha/beta/rc
                vercmp = self._compare_pre_release(other)
                if vercmp == 0:
                    # Same version and same pre-release, check if dev version
                    if self.is_devversion is other.is_devversion:
                        vercmp = 0
                    elif self.is_devversion:
                        vercmp = -1
                    else:
                        vercmp = 1

            return vercmp

        def __lt__(self, other):
            return self._compare(other) < 0

        def __le__(self, other):
            return self._compare(other) <= 0

        def __eq__(self, other):
            return self._compare(other) == 0

        def __ne__(self, other):
            return self._compare(other) != 0

        def __gt__(self, other):
            return self._compare(other) > 0

        def __ge__(self, other):
            return self._compare(other) >= 0

        def __repr(self):
            return "NumpyVersion(%s)" % self.vstring


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2 ** ((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2 ** _bit_length_26(quotient - 1)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match
if NumpyVersion(np.__version__) >= '1.7.1':
    np_matrix_rank = np.linalg.matrix_rank
else:
    def np_matrix_rank(M, tol=None):
        """
        Return matrix rank of array using SVD method
        Rank of the array is the number of SVD singular values of the array that are
        greater than `tol`.
        Parameters
        ----------
        M : {(M,), (M, N)} array_like
            array of <=2 dimensions
        tol : {None, float}, optional
        threshold below which SVD values are considered zero. If `tol` is
        None, and ``S`` is an array with singular values for `M`, and
        ``eps`` is the epsilon value for datatype of ``S``, then `tol` is
        set to ``S.max() * max(M.shape) * eps``.
        Notes
        -----
        The default threshold to detect rank deficiency is a test on the magnitude
        of the singular values of `M`.  By default, we identify singular values less
        than ``S.max() * max(M.shape) * eps`` as indicating rank deficiency (with
        the symbols defined above). This is the algorithm MATLAB uses [1].  It also
        appears in *Numerical recipes* in the discussion of SVD solutions for linear
        least squares [2].
        This default threshold is designed to detect rank deficiency accounting for
        the numerical errors of the SVD computation.  Imagine that there is a column
        in `M` that is an exact (in floating point) linear combination of other
        columns in `M`. Computing the SVD on `M` will not produce a singular value
        exactly equal to 0 in general: any difference of the smallest SVD value from
        0 will be caused by numerical imprecision in the calculation of the SVD.
        Our threshold for small SVD values takes this numerical imprecision into
        account, and the default threshold will detect such numerical rank
        deficiency.  The threshold may declare a matrix `M` rank deficient even if
        the linear combination of some columns of `M` is not exactly equal to
        another column of `M` but only numerically very close to another column of
        `M`.
        We chose our default threshold because it is in wide use.  Other thresholds
        are possible.  For example, elsewhere in the 2007 edition of *Numerical
        recipes* there is an alternative threshold of ``S.max() *
        np.finfo(M.dtype).eps / 2. * np.sqrt(m + n + 1.)``. The authors describe
        this threshold as being based on "expected roundoff error" (p 71).
        The thresholds above deal with floating point roundoff error in the
        calculation of the SVD.  However, you may have more information about the
        sources of error in `M` that would make you consider other tolerance values
        to detect *effective* rank deficiency.  The most useful measure of the
        tolerance depends on the operations you intend to use on your matrix.  For
        example, if your data come from uncertain measurements with uncertainties
        greater than floating point epsilon, choosing a tolerance near that
        uncertainty may be preferable.  The tolerance may be absolute if the
        uncertainties are absolute rather than relative.
        References
        ----------
        .. [1] MATLAB reference documention, "Rank"
            http://www.mathworks.com/help/techdoc/ref/rank.html
        .. [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,
            "Numerical Recipes (3rd edition)", Cambridge University Press, 2007,
            page 795.
        Examples
        --------
        >>> from numpy.linalg import matrix_rank
        >>> matrix_rank(np.eye(4)) # Full rank matrix
        4
        >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
        >>> matrix_rank(I)
        3
        >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
        1
        >>> matrix_rank(np.zeros((4,)))
        0
        """
        M = np.asarray(M)
        if M.ndim > 2:
            raise TypeError('array should have 2 or fewer dimensions')
        if M.ndim < 2:
            return int(not all(M == 0))
        S = np.linalg.svd(M, compute_uv=False)
        if tol is None:
            tol = S.max() * max(M.shape) * np.finfo(S.dtype).eps
        return np.sum(S > tol)



class CacheWriteWarning(UserWarning):
    pass

class CachedAttribute(object):

    def __init__(self, func, cachename=None, resetlist=None):
        self.fget = func
        self.name = func.__name__
        self.cachename = cachename or '_cache'
        self.resetlist = resetlist or ()

    def __get__(self, obj, type=None):
        if obj is None:
            return self.fget
        # Get the cache or set a default one if needed
        _cachename = self.cachename
        _cache = getattr(obj, _cachename, None)
        if _cache is None:
            setattr(obj, _cachename, resettable_cache())
            _cache = getattr(obj, _cachename)
        # Get the name of the attribute to set and cache
        name = self.name
        _cachedval = _cache.get(name, None)
        # print("[_cachedval=%s]" % _cachedval)
        if _cachedval is None:
            # Call the "fget" function
            _cachedval = self.fget(obj)
            # Set the attribute in obj
            # print("Setting %s in cache to %s" % (name, _cachedval))
            try:
                _cache[name] = _cachedval
            except KeyError:
                setattr(_cache, name, _cachedval)
            # Update the reset list if needed (and possible)
            resetlist = self.resetlist
            if resetlist is not ():
                try:
                    _cache._resetdict[name] = self.resetlist
                except AttributeError:
                    pass
        # else:
        # print("Reading %s from cache (%s)" % (name, _cachedval))
        return _cachedval

    def __set__(self, obj, value):
        errmsg = "The attribute '%s' cannot be overwritten" % self.name
        warnings.warn(errmsg, CacheWriteWarning)


class _cache_readonly(object):
    """
    Decorator for CachedAttribute
    """

    def __init__(self, cachename=None, resetlist=None):
        self.func = None
        self.cachename = cachename
        self.resetlist = resetlist or None

    def __call__(self, func):
        return CachedAttribute(func,
                               cachename=self.cachename,
                               resetlist=self.resetlist)
cache_readonly = _cache_readonly()


