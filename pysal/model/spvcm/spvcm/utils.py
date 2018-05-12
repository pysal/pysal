from __future__ import division
from scipy import sparse as spar
import numpy as np
from numpy import linalg as nla
from scipy.sparse import linalg as spla
import pysal as ps
import scipy.linalg as scla
from warnings import warn as Warn

__all__ = ['grid_det']
PUBLIC_DICT_ATTS = [k for k in dir(dict) if not k.startswith('_')]

##########################
# GENERAL PURPOSE UTILS  #
##########################

def no_op(*_, **__):
    """
    This is a no-op. It takes any arguments,
    keyword or positional, and returns none
    """
    return

def thru_op(*args, **kws):
    """
    This is a thru-op. It returns everything passed to it.
    """
    if len(args) == 1 and kws == dict():
        return args[0]
    elif len(args) == 1 and kws != dict():
        return args[0], kws
    elif len(args) == 0:
        return kws
    else:
        return args, kws

##########################
# BUILD EXAMPLE DATASETS #
##########################

def south(df=False):
    """
    Sets up the data for the US southern counties example.

    Returns
    -------
    dictionary or (dictionary, dataframe), where the dictionary is keyed on:

    X           : Data from southern counties, columns "GI89", "BLK90", "HR90"
    Y           : Outcome variate, "DNL90"
    Z           : upper-level variate, the state average "FH90"
    W           : queen weights matrix between counties
    M           : queen matrix between states
    membership  : membership vector relating counties to their states

    and the dataframe contains the raw dataset
    """
    import pysal as ps

    data = ps.pdio.read_files(ps.examples.get_path('south.shp'))
    data = data[data.STATE_NAME != 'District of Columbia']
    X = data[['GI89', 'BLK90', 'HR90']].values
    N = X.shape[0]
    Z = data.groupby('STATE_NAME')['FH90'].mean()
    Z = Z.values.reshape(-1,1)
    J = Z.shape[0]

    Y = data.DNL90.values.reshape(-1,1)

    W2 = ps.queen_from_shapefile(ps.examples.get_path('us48.shp'),
                                 idVariable='STATE_NAME')
    W2 = ps.w_subset(W2, ids=data.STATE_NAME.unique().tolist()) #only keep what's in the data
    W1 = ps.queen_from_shapefile(ps.examples.get_path('south.shp'),
                                 idVariable='FIPS')
    W1 = ps.w_subset(W1, ids=data.FIPS.tolist()) #again, only keep what's in the data

    W1.transform = 'r'
    W2.transform = 'r'

    membership = data.STATE_NAME.apply(lambda x: W2.id_order.index(x)).values

    d = {'X':X, 'Y':Y, 'Z':Z, 'W':W1, 'M':W2, 'membership':membership}
    if df:
        return d, data
    else:
        return d

def baltim(df=False):
    """
    Sets up the baltimore house price example

    Returns
    --------
    dictionary or (dictinoary, dataframe), where the dictionary is keyed:

    X           : Data from baltimore houses, columns "AGE", "LOTSZ", "SQFT"
    Y           : outcomes, log house price
    coordinates : the geographic coordinates of house sales

    dataframe contains the raw data of the baltimore example
    """
    baltim = ps.pdio.read_files(ps.examples.get_path('baltim.shp'))
    coords = baltim[['X', 'Y']].values
    Y = np.log(baltim.PRICE.values).reshape(-1,1)
    Yz = Y - Y.mean()
    X = baltim[['AGE', 'LOTSZ', 'SQFT']].values
    Xz = X-X.mean(axis=0)
    out = {'Y':Yz, 'X':Xz, 'coordinates':coords}
    if df:
        return out, baltim
    else:
        return out

####################
# MATRIX UTILITIES #
####################

def lulogdet(matrix):
    """
    compute the log determinant using a lu decomposition appropriate to input type
    """
    if spar.issparse(matrix):
        LUfunction = lambda x: spla.splu(x).U.diagonal()
    else:
        LUfunction = lambda x: scla.lu_factor(x)[0].diagonal()
    LUdiag = LUfunction(matrix)
    return np.sum(np.log(np.abs(LUdiag)))

def splogdet(matrix):
    """
    compute the log determinant via an appropriate method according to the input.
    """
    redo = False
    if spar.issparse(matrix):
        LU = spla.splu(spar.csc_matrix(matrix))
        ldet = np.sum(np.log(np.abs(LU.U.diagonal())))
    else:
        sgn, ldet = nla.slogdet(matrix)
        if np.isinf(ldet) or sgn is 0:
            Warn('Dense log determinant via numpy.linalg.slogdet() failed!')
            redo = True
        if sgn not in [-1,1]:
            Warn("Drastic loss of precision in numpy.linalg.slogdet()!")
            redo = True
        ldet = sgn*ldet
    if redo:
        Warn("Please pass convert to a sparse weights matrix. Trying sparse determinant...", UserWarning)
        ldet = splogdet(spar.csc_matrix(matrix))
    return ldet

def speye(i, sparse=True):
    """
    constructs a square identity matrix according to i, either sparse or dense
    """
    if sparse:
        return spar.identity(i)
    else:
        return np.identity(i)

spidentity = speye

def speye_like(matrix):
    """
    constructs an identity matrix depending on the input dimension and type
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise UserWarning("Matrix is not square")
    else:
        return speye(matrix.shape[0], sparse=spar.issparse(matrix))

spidentity_like = speye_like

def speigen_range(matrix, retry=True, coerce=True):
    """
    Construct the eigenrange of a potentially sparse matrix.
    """
    if spar.issparse(matrix):
        try:
            emax = spla.eigs(matrix, k=1, which='LR')[0]
        except (spla.ArpackNoConvergence, spla.ArpackError) as e:
            rowsums = np.unique(np.asarray(matrix.sum(axis=1)).flatten())
            if np.allclose(rowsums, np.ones_like(rowsums)):
                emax = np.array([1])
            else:
                Warn('Maximal eigenvalue computation failed to converge'
                     ' and matrix is not row-standardized.')
                raise e
        emin = spla.eigs(matrix, k=1, which='SR')[0]
        if coerce:
            emax = emax.real.astype(float)
            emin = emin.real.astype(float)
    else:
        try:
            eigs = nla.eigvals(matrix)
            emin, emax = eigs.min().astype(float), eigs.max().astype(float)
        except Exception as e:
            Warn('Dense eigenvector computation failed!')
            if retry:
                Warn('Retrying with sparse matrix...')
                spmatrix = spar.csc_matrix(matrix)
                speigen_range(spmatrix)
            else:
                Warn('Bailing...')
                raise e
    return emin, emax

def spinv(M):
    """
    Compute an inverse of a matrix using the appropriate sparse or dense
    function
    """
    if spar.issparse(M):
        return spla.inv(M)
    else:
        return nla.inv(M)

def spsolve(A,b):
    """
    Solve the system Ax=b for x, depending on the type of A. The solution vector is equivalent to A^{-1}b

    If a is sparse, the result will be sparse. Otherwise, the result will be dense.
    """
    if spar.issparse(A):
        return spla.spsolve(A, b)
    elif spar.issparse(b):
        Warn('b is sparse, but A is dense. Solving the dense system.')
        return spsolve(A, b.toarray())
    return scla.solve(A,b)



#########################
# STATISTICAL UTILITIES #
#########################

def chol_mvn(Mu, Sigma):
    """
    Sample from a Multivariate Normal given a mean & Covariance matrix, using
    cholesky decomposition of the covariance. If the cholesky decomp fails due
    to the matrix not being strictly positive definite, then the
    numpy.random.multivariate_normal will be used.

    That is, new values are generated according to :
    New = Mu + cholesky(Sigma) . N(0,1)

    Parameters
    ----------
    Mu      :   np.ndarray (p,1)
                An array containing the means of the multivariate normal being
                sampled
    Sigma   :   np.ndarray (p,p)
                An array containing the covariance between the dimensions of the
                multivariate normal being sampled

    Returns
    -------
    np.ndarray of size (p,1) containing draws from the multivariate normal
    described by MVN(Mu, Sigma)
    """
    try:
        D = scla.cholesky(Sigma, overwrite_a = True)
        e = np.random.normal(0,1,size=Mu.shape)
        kernel = np.dot(D.T, e)
        out = Mu + kernel
    except np.linalg.LinAlgError:
        out = np.random.multivariate_normal(Mu.flatten(), Sigma)
        out = out.reshape(Mu.shape)
    return out

def sma_covariance(param, W, sparse=True):
    """
    This computes a covariance matrix for a SMA-type error specification:

    ( (I + param * W)(I + param * W)^T)

    this always returns a dense array
    """
    half = speye_like(W) + param * W
    whole = half.dot(half.T)
    if sparse:
        return whole
    return whole.toarray()

def sma_precision(param, W, sparse=False):
    """
    This computes a precision matrix for a spatial moving average error specification.
    """
    covariance = sma_covariance(param, W, sparse=sparse)
    if sparse:
        return spinv(covariance)
    return np.linalg.inv(covariance)

def se_covariance(param, W, sparse=False):
    """
    This computes a covariance matrix for a SAR-type error specification:

    ( (I - param * W)^T(I - param * W) )^{-1}

    and always returns a dense matrix.

    This first calls se_precision, and then inverts the results of that call.

    """
    prec = se_precision(param, W, sparse=sparse)
    if sparse:
        return spla.inv(prec)
    return np.linalg.inv(prec)

def se_precision(param, W, sparse=True):
    """
    This computes a precision matrix for a SAR-type error specification.
    """
    half = speye_like(W) - param * W
    prec = half.T.dot(half)
    if sparse:
        return prec
    return prec.toarray()

def ind_covariance(param, W, sparse=False):
    """
    This returns a covariance matrix for a standard diagonal specification:

    I

    and always returns a dense matrix. Thus, it ignores param entirely.
    """
    out = speye(W.shape[0], sparse=sparse)
    if sparse:
        return spar.csc_matrix(out)
    return out

def grid_det(W, parmin=None, parmax=None, parstep=None, grid=None):
    """
    This is a utility function to set up the grid of matrix determinants over a
    range of spatial parameters for a given W.
    """
    if (parmin is None) and (parmax is None):
        parmin, parmax = speigen_range(W)
    if parstep is None:
        parstep = (parmax - parmin) / 1000
    if grid is None:
        grid = np.arange(parmin, parmax, parstep)
    logdets = [splogdet(speye_like(W) - rho * W) for rho in grid]
    grid = np.vstack((grid, np.array(logdets).reshape(grid.shape)))
    return grid
