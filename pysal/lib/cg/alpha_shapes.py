"""
Computation of alpha shape algorithm in 2-D based on original implementation
by Tim Kittel (@timkittel) available at:

    https://github.com/timkittel/alpha-shapes

Author(s):
    Dani Arribas-Bel daniel.arribas.bel@gmail.com
"""

try:
    from numba import jit
    HAS_JIT = True
except ImportError:
    from warnings import warn
    def jit(function=None, **kwargs):
        if function is not None:
            def wrapped(*original_args, **original_kw):
                return function(*original_args, **original_kw)
            return wrapped
        else:
            def partial_inner(func):
                return jit(func)
            return partial_inner
    HAS_JIT = False
import numpy as np
import scipy.spatial as spat

EPS = np.finfo(float).eps

__all__ = ['alpha_shape', 'alpha_shape_auto']

@jit
def nb_dist(x, y):
    '''
    numba implementation of distance between points `x` and `y`
    ...

    Arguments
    ---------
    x       : ndarray
              Coordinates of point `x`
    y       : ndarray
              Coordinates of point `y`

    Returns
    -------
    dist    : float
              Distance between `x` and `y`

    Example
    -------

    >>> x = np.array([0, 0])
    >>> y = np.array([1, 1])
    >>> dist = nb_dist(x, y)
    >>> dist
    1.4142135623730951
    '''
    sum = 0
    for x_i, y_i in zip(x, y):
        sum += (x_i - y_i)**2
    dist = np.sqrt(sum)
    return dist

@jit(nopython=True)
def r_circumcircle_triangle_single(a, b, c):
    '''
    Computation of the circumcircle of a single triangle
    ...

    Source for equations:

    > https://www.mathopenref.com/trianglecircumcircle.html

    [Last accessed July 11th. 2018]

    Arguments
    ---------
    a       : ndarray
              (2,) Array with coordinates of vertex `a` of the triangle
    b       : ndarray
              (2,) Array with coordinates of vertex `b` of the triangle
    c       : ndarray
              (2,) Array with coordinates of vertex `c` of the triangle

    Returns
    -------
    r       : float
              Circumcircle of the triangle

    Example
    -------

    >>> a = np.array([0, 0])
    >>> b = np.array([0.5, 0])
    >>> c = np.array([0.25, 0.25])
    >>> r = r_circumcircle_triangle_single(a, b, c)
    >>> r
    0.2500000000000001
    '''
    ab = nb_dist(a, b)
    bc = nb_dist(b, c)
    ca = nb_dist(c, a)

    num = ab * bc * ca
    den = np.sqrt( (ab + bc + ca) * \
                   (bc + ca - ab) * \
                   (ca + ab - bc) * \
                   (ab + bc - ca) )
    if den == 0:
        return np.array([ab, bc, ca]).max() / 2.0
    else:
        return num / den

@jit(nopython=True)
def r_circumcircle_triangle(a_s, b_s, c_s):
    '''
    Computation of circumcircles for a series of triangles
    ...

    Arguments
    ---------
    a_s     : ndarray
              (N, 2) array with coordinates of vertices `a` of the triangles
    b_s     : ndarray
              (N, 2) array with coordinates of vertices `b` of the triangles
    c_s     : ndarray
              (N, 2) array with coordinates of vertices `c` of the triangles

    Returns
    -------
    radii   : ndarray
              (N,) array with circumcircles for every triangle

    Example
    -------

    >>> a_s = np.array([[0, 0], [2, 1], [3, 2]])
    >>> b_s = np.array([[1, 0], [5, 1], [2, 4]])
    >>> c_s = np.array([[0, 7], [1, 3], [4, 2]])
    >>> rs = r_circumcircle_triangle(a_s, b_s, c_s)
    >>> rs
    array([3.53553391, 2.5       , 1.58113883])
    '''
    len_a = len(a_s)
    r2 = np.zeros( (len_a,) )
    for i in range(len_a):
        r2[i] = r_circumcircle_triangle_single(a_s[i], 
                                               b_s[i], 
                                               c_s[i])
    return r2

@jit
def get_faces(triangle):
    '''
    Extract faces from a single triangle
    ...

    Arguments
    ---------
    triangles       : ndarray
                      (3,) array with the vertex indices for a triangle

    Returns
    -------
    faces           : ndarray
                      (3, 2) array with a row for each face containing the
                      indices of the two points that make up the face

    Example
    -------
    
    >>> triangle = np.array([3, 1, 4], dtype=np.int32)
    >>> faces = get_faces(triangle)
    >>> faces
    array([[3., 1.],
           [1., 4.],
           [4., 3.]])

    '''
    faces = np.zeros((3, 2))
    for i, (i0, i1) in enumerate([(0, 1), (1, 2), (2, 0)]):
        faces[i] = triangle[i0], triangle[i1]
    return faces

@jit
def build_faces(faces, triangles_is, 
        num_triangles, num_faces_single):
    '''
    Build facing triangles

    ...

    Arguments
    ---------
    faces               : ndarray
                          (num_triangles * num_faces_single, 2) array of
                          zeroes in int form
    triangles_is        : ndarray
                          (D, 3) array, where D is the number of Delaunay
                          triangles, with the vertex indices for each
                          triangle
    num_triangles       : int
                          Number of triangles
    num_faces_single    : int
                          Number of faces a triangle has (i.e. 3)

    Returns
    -------
    faces               : ndarray
                          Two dimensional array with a row for every facing
                          segment containing the indices of the coordinate points

    Example
    -------
    >>> import scipy.spatial as spat
    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> triangulation = spat.Delaunay(pts)
    >>> triangulation.simplices
    array([[3, 1, 4],
           [1, 2, 4],
           [2, 1, 0]], dtype=int32)
    >>> num_faces_single = 3
    >>> num_triangles = triangulation.simplices.shape[0]
    >>> num_faces = num_triangles * num_faces_single
    >>> faces = np.zeros((num_faces, 2), dtype=np.int_)
    >>> mask = np.ones((num_faces,), dtype=np.bool_)
    >>> faces = build_faces(faces, triangulation.simplices, num_triangles, num_faces_single)
    >>> faces
    array([[3, 1],
           [1, 4],
           [4, 3],
           [1, 2],
           [2, 4],
           [4, 1],
           [2, 1],
           [1, 0],
           [0, 2]])

    '''
    for i in range(num_triangles):
        from_i = num_faces_single * i
        to_i = num_faces_single * (i+1)
        faces[from_i: to_i] = get_faces(triangles_is[i])
    return faces

@jit
def nb_mask_faces(mask, faces):
    '''
    Run over each row in `faces`, if the face in the following row is the
    same, then mark both as False on `mask` 
    ...

    Arguments
    ---------
    mask    : ndarray
              One-dimensional boolean array set to True with as many
              observations as rows in `faces`
    faces   : ndarray
              Sorted sequence of faces for all triangles (ie. triangles split
              by each segment)

    Returns
    -------
    masked  : ndarray
              Sequence of outward-facing faces

    Example
    -------
    >>> import numpy as np
    >>> faces = np.array([[0, 1], [0, 2], [1, 2], [1, 2], [1, 3], [1, 4], [1, 4], [2, 4], [3, 4]])
    >>> mask = np.ones((faces.shape[0], ), dtype=np.bool_)
    >>> masked = nb_mask_faces(mask, faces)
    >>> masked
    array([[0, 1],
           [0, 2],
           [1, 3],
           [2, 4],
           [3, 4]])
    '''
    for k in range(faces.shape[0]-1):
        if mask[k]:
            if np.all(faces[k] == faces[k+1]):
                mask[k] = False
                mask[k+1] = False
    return faces[mask]

def get_single_faces(triangles_is):
    '''
    Extract outward facing edges from collection of triangles
    ...

    Arguments
    ---------
    triangles_is    : ndarray
                      (D, 3) array, where D is the number of Delaunay triangles,
                      with the vertex indices for each triangle

    Returns
    -------
    single_faces    : ndarray

    Example
    -------
    >>> import scipy.spatial as spat
    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> alpha = 0.33
    >>> triangulation = spat.Delaunay(pts)
    >>> triangulation.simplices
    array([[3, 1, 4],
           [1, 2, 4],
           [2, 1, 0]], dtype=int32)
    >>> get_single_faces(triangulation.simplices)
    array([[0, 1],
           [0, 2],
           [1, 3],
           [2, 4],
           [3, 4]])

    '''
    num_faces_single = 3
    num_triangles = triangles_is.shape[0]
    num_faces = num_triangles * num_faces_single
    faces = np.zeros((num_faces, 2), dtype=np.int_)
    mask = np.ones((num_faces,), dtype=np.bool_)

    faces = build_faces(faces, triangles_is, 
                        num_triangles, num_faces_single)

    orderlist = ["x{}".format(i) for i in range(faces.shape[1])]
    dtype_list = [(el, faces.dtype.str) for el in orderlist]
    # Arranging each face so smallest vertex is first
    faces.sort(axis=1)                  
    # Arranging faces in ascending way
    faces.view(dtype_list).sort(axis=0)
    # Masking
    single_faces = nb_mask_faces(mask, faces)
    return single_faces

def alpha_geoms(alpha, triangles, radii, xys):
    '''
    Generate alpha-shape polygon(s) from `alpha` value, vertices of `triangles`,
    the `radii` for all points, and the points themselves
    ...

    Arguments
    ---------
    alpha       : float
                  Alpha value to delineate the alpha-shape
    triangles   : ndarray
                  (D, 3) array, where D is the number of Delaunay triangles,
                  with the vertex indices for each triangle
    radii       : ndarray
                  (N,) array with circumcircles for every triangle
    xys         : ndarray
                  (N, 2) array with one point per row and coordinates structured
                  as X and Y

    Returns
    -------
    geoms       : GeoSeries
                  Polygon(s) resulting from the alpha shape algorithm. The
                  GeoSeries object remains so even if only a single polygon is
                  returned. There is no CRS included in the object.

    Example
    -------
    >>> import scipy.spatial as spat
    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> alpha = 0.33
    >>> triangulation = spat.Delaunay(pts)
    >>> triangles = pts[triangulation.simplices]
    >>> triangles
    array([[[6, 7],
            [3, 5],
            [9, 3]],
    <BLANKLINE>
           [[3, 5],
            [4, 1],
            [9, 3]],
    <BLANKLINE>
           [[4, 1],
            [3, 5],
            [0, 1]]])
    >>> a_pts = triangles[:, 0, :]
    >>> b_pts = triangles[:, 1, :]
    >>> c_pts = triangles[:, 2, :]
    >>> radii = r_circumcircle_triangle(a_pts, b_pts, c_pts)
    >>> geoms = alpha_geoms(alpha, triangulation.simplices, radii, pts)
    >>> geoms
    0    POLYGON ((0 1, 3 5, 4 1, 0 1))
    dtype: object
    '''
    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize
    except ImportError:
        raise ImportError("Shapely is a required package to use alpha_shapes")

    try:
        from geopandas import GeoSeries
    except ImportError:
        raise ImportError("Geopandas is a required package to use alpha_shapes")

    triangles_reduced = triangles[radii < 1/alpha]
    outer_triangulation = get_single_faces(triangles_reduced)
    face_pts = xys[outer_triangulation]
    geoms = GeoSeries(list(polygonize(list(map(LineString, 
                                               face_pts)))))
    return geoms

def alpha_shape(xys, alpha):
    '''
    Alpha-shape delineation (Edelsbrunner, Kirkpatrick &
    Seidel, 1983) from a collection of points
    ...

    Arguments
    ---------
    xys     : ndarray
              (N, 2) array with one point per row and coordinates structured as X
              and Y
    alpha   : float
              Alpha value to delineate the alpha-shape

    Returns
    -------
    shapes  : GeoSeries
              Polygon(s) resulting from the alpha shape algorithm. The
              GeoSeries object remains so even if only a single polygon is
              returned. There is no CRS included in the object.

    Example
    -------

    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> alpha = 0.1
    >>> poly = alpha_shape(pts, alpha)
    >>> poly
    0    POLYGON ((0 1, 3 5, 6 7, 9 3, 4 1, 0 1))
    dtype: object
    >>> poly.centroid
    0    POINT (4.690476190476191 3.452380952380953)
    dtype: object

    References
    ----------

    Edelsbrunner, H., Kirkpatrick, D., & Seidel, R. (1983). On the shape of
        a set of points in the plane. IEEE Transactions on information theory, 
        29(4), 551-559.
    '''
    if not HAS_JIT:
        warn("Numba not imported, so alpha shape construction may be slower than expected.")
    triangulation = spat.Delaunay(xys)
    triangles = xys[triangulation.simplices]
    a_pts = triangles[:, 0, :]
    b_pts = triangles[:, 1, :]
    c_pts = triangles[:, 2, :]
    radii = r_circumcircle_triangle(a_pts, b_pts, c_pts)
    del triangles, a_pts, b_pts, c_pts
    geoms = alpha_geoms(alpha, triangulation.simplices, radii, xys)
    return geoms

def alpha_shape_auto(xys, step=1, verbose=False):
    '''
    Computation of alpha-shape delineation with automated selection of alpha.
    ...

    This method uses the algorithm proposed by  Edelsbrunner, Kirkpatrick &
    Seidel (1983) to return the tightest polygon that contains all points in
    `xys`. The algorithm ranks every point based on its radious and iterates
    over each point, checking whether the maximum alpha that would keep the
    point and all the other ones in the set with smaller radii results in a
    single polygon. If that is the case, it moves to the next point;
    otherwise, it retains the previous alpha value and returns the polygon
    as `shapely` geometry.

    Arguments
    ---------
    xys     : ndarray
              Nx2 array with one point per row and coordinates structured as X
              and Y
    step    : int
              [Optional. Default=1]
              Number of points in `xys` to jump ahead after checking whether the
              largest possible alpha that includes the point and all the
              other ones with smaller radii
    verbose : Boolean
              [Optional. Default=False] If True, it prints alpha values being
              tried at every step.


    Returns
    -------
    poly    : shapely.Polygon
              Tightest alpha-shape polygon containing all points in `xys`

    Example
    -------

    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> poly = alpha_shape_auto(pts)
    >>> poly.bounds
    (0.0, 1.0, 9.0, 7.0)
    >>> poly.centroid.x, poly.centroid.y
    (4.690476190476191, 3.4523809523809526)

    References
    ----------

    Edelsbrunner, H., Kirkpatrick, D., & Seidel, R. (1983). On the shape of
        a set of points in the plane. IEEE Transactions on information theory, 
        29(4), 551-559.
    '''
    if not HAS_JIT:
        warn("Numba not imported, so alpha shape construction may be slower than expected.")
    triangulation = spat.Delaunay(xys)
    triangles = xys[triangulation.simplices]
    a_pts = triangles[:, 0, :]
    b_pts = triangles[:, 1, :]
    c_pts = triangles[:, 2, :]
    radii = r_circumcircle_triangle(a_pts, b_pts, c_pts)
    radii[np.isnan(radii)] = 0 # "Line" triangles to be kept for sure
    del triangles, a_pts, b_pts, c_pts
    radii_sorted_i = radii.argsort()
    triangles = triangulation.simplices[radii_sorted_i][::-1]
    radii = radii[radii_sorted_i][::-1]
    geoms_prev = alpha_geoms((1/radii.max())-EPS, triangles, radii, xys)
    xys_bb = np.array([*xys.min(axis=0), *xys.max(axis=0)])
    if verbose:
        print('Step set to %i'%step)
    for i in range(0, len(radii), step):
        radi = radii[i]
        alpha = (1 / radi) - EPS
        if verbose:
            print('%.2f%% | Trying a = %f'\
		  %((i+1)/radii.shape[0], alpha))
        geoms = alpha_geoms(alpha, triangles, radii, xys)
        if (geoms.shape[0] != 1) or not (np.all(xys_bb == geoms.total_bounds)):
            break
        else:
            geoms_prev = geoms
    return geoms_prev[0] # Return a shapely polygon

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time
    import geopandas as gpd
    plt.close('all')
    xys = np.random.random((1000, 2))
    t0 = time.time()
    geoms = alpha_shape_auto(xys, 1)
    t1 = time.time()
    print('%.2f Seconds to run algorithm'%(t1-t0))
    f, ax = plt.subplots(1)
    gpd.GeoDataFrame({'geometry':[geoms]}).plot(ax=ax, color='orange', alpha=0.5)
    ax.scatter(xys[:, 0], xys[:, 1], s=0.1)
    plt.show()

