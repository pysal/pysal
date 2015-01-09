.. _weights:

.. testsetup:: *

    import pysal
    import numpy as np

***************
Spatial Weights
***************

.. contents::

Introduction
============

Spatial weights are central components of many areas of spatial analysis. In
general terms, for a spatial data set composed of n locations (points, areal
units, network edges, etc.), the spatial weights matrix expresses the potential
for interaction between observations at each pair i,j of locations. There is a rich
variety of ways to specify the structure of these weights, and
PySAL supports the creation, manipulation and analysis of spatial weights
matrices across three different general types:

 * Contiguity Based Weights
 * Distance Based Weights
 * Kernel Weights

These different types of weights are implemented as instances of the PySAL weights class 
:class:`~pysal.weights.W`. 

In what follows, we provide a high level overview of spatial weights in PySAL, starting with the three different types of weights, followed by
a closer look at the properties of the W class and some related functions. [#]_

PySAL Spatial Weight Types
==========================
PySAL weights are handled in objects of the :class:`pysal.weights.W`. The
conceptual idea of spatial weights is that of a nxn matrix in which the
diagonal elements (:math:`w_{ii}`) are set to zero by definition and the rest of
the cells (:math:`w_{ij}`) capture the potential of interaction. However, these
matrices tend to be fairly sparse (i.e. many cells contain zeros) and hence a
full nxn array would not be an efficient representation. PySAL employs a
different way of storing that is structured in two main dictionaries [#]_ :
neighbors which, for each observation (key) contains a list of the other ones 
(value) with potential for interaction (:math:`w_{ij} \neq 0`); and weights, 
which contains the weight values for each of those observations (in the same 
order). This way, large datasets can be stored when keeping the full matrix 
would not be possible because of memory constraints. In addition to the sparse
representation via the weights and neighbors dictionaries, a PySAL W object
also has an attribute called sparse, which is a `scipy.sparse
<http://docs.scipy.org/doc/scipy/reference/sparse.html>`_ CSR
representation of the spatial weights. (See :ref:`wsp` for an alternative
PySAL weights object.) 

.. _contiguity:

Contiguity Based Weights
------------------------

To illustrate the general weights object, we start with a simple contiguity
matrix constructed for a 5 by 5 lattice (composed of 25 spatial units):

.. doctest::

    >>> import pysal
    >>> w = pysal.lat2W(5, 5)

The w object has a number of attributes:

.. doctest::

    >>> w.n
    25
    >>> w.pct_nonzero
    0.128
    >>> w.weights[0]
    [1.0, 1.0]
    >>> w.neighbors[0]
    [5, 1]
    >>> w.neighbors[5]
    [0, 10, 6]
    >>> w.histogram
    [(2, 4), (3, 12), (4, 9)]

n is the number of spatial units, so conceptually we could be thinking that the
weights are stored in a 25x25 matrix. The second attribute
(pct_nonzero) shows the sparseness of the matrix. The key
attributes used to store contiguity relations in W are the neighbors and
weights attributes. In the example above we see that the observation
with id 0 (Python is zero-offset) has two neighbors with ids [5, 1] each of
which have equal weights of 1.0.

The histogram attribute is a set of tuples indicating the cardinality of the
neighbor relations. In this case we have a regular lattice, so there are 4 units that have 2
neighbors (corner cells), 12 units with 3 neighbors (edge cells), and 9 units
with 4 neighbors (internal cells).

In the above example, the default criterion for contiguity on the lattice was
that of the rook which takes as neighbors any pair of cells that share an edge.
Alternatively, we could have used the queen criterion to include the vertices
of the lattice to define contiguities:

.. doctest::

	>>> wq = pysal.lat2W(rook = False)
	>>> wq.neighbors[0]
	[5, 1, 6]
	>>> 

The bishop criterion, which designates pairs of cells as neighbors if they share
only a vertex, is yet a third alternative for contiguity weights. A bishop matrix
can be computed as the :ref:`difference` between the rook and queen cases.

The lat2W function is particularly useful in setting up simulation experiments
requiring a regular grid. For empirical research, a common use case is to have
a shapefile, which is a nontopological vector data structure, and a need
to carry out some form of spatial analysis that requires spatial weights. Since
topology is not stored in the underlying file there is a need to construct
the spatial weights prior to carrying out the analysis. In PySAL spatial
weights can be obtained directly from shapefiles:

.. doctest::

    >>> w = pysal.rook_from_shapefile("../pysal/examples/columbus.shp")
    >>> w.n
    49
    >>> print "%.4f"%w.pct_nonzero
    0.0833
    >>> w.histogram
    [(2, 7), (3, 10), (4, 17), (5, 8), (6, 3), (7, 3), (8, 0), (9, 1)]

If queen, rather than rook, contiguity is required then the following would work:

.. doctest::

    >>> w = pysal.queen_from_shapefile("../pysal/examples/columbus.shp")
    >>> print "%.4f"%w.pct_nonzero
    0.0983
    >>> w.histogram
    [(2, 5), (3, 9), (4, 12), (5, 5), (6, 9), (7, 3), (8, 4), (9, 1), (10, 1)]
    


Distance Based Weights
----------------------

In addition to using contiguity to define  neighbor relations, more general
functions of the distance separating observations can be used to specify the
weights.

Please note that distance calculations are coded for a flat surface, so you
will need to have your shapefile projected in advance for the output to be
correct.

k-nearest neighbor weights
--------------------------

The neighbors for a given observations can be defined using a k-nearest neighbor criterion.
For example we could use the the centroids of our
5x5 lattice as point locations to measure the distances. First, we import numpy to 
create the coordinates as a 25x2 numpy array named data (numpy arrays are the only
form of input supported at this point):

.. doctest::

    >>> import numpy as np
    >>> x,y=np.indices((5,5))
    >>> x.shape=(25,1)
    >>> y.shape=(25,1)
    >>> data=np.hstack([x,y])
    
    
then define the knn set as:

.. doctest::

    >>> wknn3 = pysal.knnW(data, k = 3)
    >>> wknn3.neighbors[0]
    [1, 5, 6]
    >>> wknn3.s0
    75.0
    >>> w4 = pysal.knnW(data, k = 4)
    >>> set(w4.neighbors[0]) == set([1, 5, 6, 2])
    True
    >>> w4.s0
    100.0
    >>> w4.weights[0]
    [1.0, 1.0, 1.0, 1.0]

Alternatively, we can use a utility function to build a knn W straight from a
shapefile:

.. doctest::
    
    >>> wknn5 = pysal.knnW_from_shapefile(pysal.examples.get_path('columbus.shp'), k=5)
    >>> wknn5.neighbors[0]
    [2, 1, 3, 7, 4]

Distance band weights
---------------------

Knn weights ensure that all observations have the same number of neighbors.  [#]_
An alternative distance based set of weights relies on distance bands or
thresholds to define the neighbor set for each spatial unit as those other units
falling within a threshold distance of the focal unit:

.. doctest::

    >>> wthresh = pysal.threshold_binaryW_from_array(data, 2)
    >>> set(wthresh.neighbors[0]) == set([1, 2, 5, 6, 10])
    True
    >>> set(wthresh.neighbors[1]) == set( [0, 2, 5, 6, 7, 11, 3])
    True
    >>> wthresh.weights[0]
    [1, 1, 1, 1, 1]
    >>> wthresh.weights[1]
    [1, 1, 1, 1, 1, 1, 1]
    >>> 

As can be seen in the above example, the number of neighbors is likely to vary
across observations with distance band weights in contrast to what holds for
knn weights.

Distance band weights can be generated for shapefiles as well as arrays of points. [#]_ First, the 
minimum nearest neighbor distance should be determined so that each unit is assured of at least one 
neighbor:

.. doctest::

    >>> thresh = pysal.min_threshold_dist_from_shapefile("../pysal/examples/columbus.shp")
    >>> thresh
    0.61886415807685413

with this threshold in hand, the distance band weights are obtained as:

.. doctest::

    >>> wt = pysal.threshold_binaryW_from_shapefile("../pysal/examples/columbus.shp", thresh)
    >>> wt.min_neighbors
    1
    >>> wt.histogram
    [(1, 4), (2, 8), (3, 6), (4, 2), (5, 5), (6, 8), (7, 6), (8, 2), (9, 6), (10, 1), (11, 1)]
    >>> set(wt.neighbors[0]) == set([1,2])
    True
    >>> set(wt.neighbors[1]) == set([3,0])
    True

Distance band weights can also be specified to take on continuous values rather
than binary, with the values set to the inverse distance separating each pair
within a given threshold distance. We illustrate this with a small set of 6
points:

.. doctest::

    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> wid = pysal.threshold_continuousW_from_array(points,14.2)
    >>> wid.weights[0]
    [0.10000000000000001, 0.089442719099991588]

If we change the distance decay exponent to -2.0 the result is so called gravity weights:

.. doctest::

    >>> wid2 = pysal.threshold_continuousW_from_array(points,14.2,alpha = -2.0)
    >>> wid2.weights[0]
    [0.01, 0.0079999999999999984]


Kernel Weights
--------------

A combination of distance based thresholds together with  continuously valued
weights is supported through kernel weights:

.. doctest::

    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw = pysal.Kernel(points)
    >>> kw.weights[0]
    [1.0, 0.500000049999995, 0.4409830615267465]
    >>> kw.neighbors[0]
    [0, 1, 3]
    >>> kw.bandwidth
    array([[ 20.000002],
           [ 20.000002],
           [ 20.000002],
           [ 20.000002],
           [ 20.000002],
           [ 20.000002]])


The bandwidth attribute plays the role of the distance threshold with kernel
weights, while the form of the kernel function determines the distance decay
in the derived continuous weights (the following are available:
'triangular','uniform','quadratic','epanechnikov','quartic','bisquare','gaussian').
In the above example, the bandwidth is set to the default value and fixed
across the observations.  The user could specify a different value for a fixed
bandwidth:

.. doctest::

    >>> kw15 = pysal.Kernel(points,bandwidth = 15.0)
    >>> kw15[0]
    {0: 1.0, 1: 0.33333333333333337, 3: 0.2546440075000701}
    >>> kw15.neighbors[0]
    [0, 1, 3]
    >>> kw15.bandwidth
    array([[ 15.],
           [ 15.],
           [ 15.],
           [ 15.],
           [ 15.],
           [ 15.]])

which results in fewer neighbors for the first unit.  Adaptive bandwidths (i.e., different bandwidths
for each unit) can also be user specified:

.. doctest::

    >>> bw = [25.0,15.0,25.0,16.0,14.5,25.0]
    >>> kwa = pysal.Kernel(points,bandwidth = bw)
    >>> kwa.weights[0]
    [1.0, 0.6, 0.552786404500042, 0.10557280900008403]
    >>> kwa.neighbors[0]
    [0, 1, 3, 4]
    >>> kwa.bandwidth
    array([[ 25. ],
           [ 15. ],
           [ 25. ],
           [ 16. ],
           [ 14.5],
           [ 25. ]])

Alternatively the adaptive bandwidths could be defined endogenously:

.. doctest::

    >>> kwea = pysal.Kernel(points,fixed = False)
    >>> kwea.weights[0]
    [1.0, 0.10557289844279438, 9.99999900663795e-08]
    >>> kwea.neighbors[0]
    [0, 1, 3]
    >>> kwea.bandwidth
    array([[ 11.18034101],
           [ 11.18034101],
           [ 20.000002  ],
           [ 11.18034101],
           [ 14.14213704],
           [ 18.02775818]])

Finally, the kernel function could be changed (with endogenous adaptive bandwidths):

.. doctest::

    >>> kweag = pysal.Kernel(points,fixed = False,function = 'gaussian')
    >>> kweag.weights[0]
    [0.3989422804014327, 0.2674190291577696, 0.2419707487162134]
    >>> kweag.bandwidth
    array([[ 11.18034101],
           [ 11.18034101],
           [ 20.000002  ],
           [ 11.18034101],
           [ 14.14213704],
           [ 18.02775818]])


More details on kernel weights can be found in 
:class:`~pysal.weights.Distance.Kernel`. 


A Closer look at W
==================

Although the three different types of spatial weights illustrated above cover a wide array of approaches
towards specifying spatial relations, they all share common attributes from the base W class in PySAL. Here 
we take a closer look at some of the more useful properties of this class.

Attributes of W
-----------------------------
W objects come with a whole bunch of useful attributes that may help you when
dealing with spatial weights matrices. To see a list of all of them, same as
with any other Python object, type:

    >>> import pysal
    >>> help(pysal.W)

If you want to be more specific and learn, for example, about the attribute
`s0`, then type:

    >>> help(pysal.W.s0)
    Help on property:

        float
            
        .. math::
                
            s0 = \sum_i \sum_j w_{i,j}

Weight Transformations
----------------------

Often there is a need to apply a transformation to the spatial weights, such as in the case of row standardization.
Here each value in the row of the spatial weights matrix is rescaled to sum to one:

.. math::
   
     ws_{i,j} = w_{i,j}/ \sum_j w_{i,j}

This and other weights transformations in PySAL are supported by the transform property of the W class. To see this 
let's build a simple contiguity object for the Columbus data set:

.. doctest::

    >>> w = pysal.rook_from_shapefile("../pysal/examples/columbus.shp")
    >>> w.weights[0]
    [1.0, 1.0]

We can row standardize this by setting the transform property:

.. doctest::

    >>> w.transform = 'r'
    >>> w.weights[0]
    [0.5, 0.5]

Supported transformations are the following:
    
    * '`b`': binary.
    * '`r`': row standardization.
    * '`v`': variance stabilizing.

If the original weights (unstandardized) are required, the transform property can be reset:

.. doctest::

    >>> w.transform = 'o'
    >>> w.weights[0]
    [1.0, 1.0]
 
Behind the scenes the transform property is updating all other characteristics of the spatial weights that are a function of the
values and these standardization operations, freeing the user from having to keep these other attributes updated. To determine the current
value of the transformation, simply query this attribute:

.. doctest::

    >>> w.transform
    'O'

More details on other transformations that are supported in W can be found in
:class:`pysal.weights.W`. 



W related functions
===================

Generating a full array
-----------------------
As the underlying data structure of the weights in W is based on a sparse representation, there may be a need to work with the full numpy array.
This is supported through the full method of W:

.. doctest::

    >>> wf = w.full()
    >>> len(wf)
    2

The first element of the return from w.full is the numpy array:
    
.. doctest::

    >>> wf[0].shape
    (49, 49)

while the second element contains the ids for the row (column) ordering of the array:

.. doctest::

    >>> wf[1][0:5]
    [0, 1, 2, 3, 4]

If only the array is required, a simple Python slice can be used:

.. doctest::

    >>> wf = w.full()[0]
    

Shimbel Matrices
----------------
The Shimbel matrix for a set of n objects contains the shortest path distance
separating each pair of units.  This has wide use in spatial analysis for
solving different types of clustering and optimization problems. Using the
function `shimbel` with a `W` instance as an argument generates this
information:

.. doctest::

    >>> w = pysal.lat2W(3,3)
    >>> ws = pysal.shimbel(w)
    >>> ws[0]
    [-1, 1, 2, 1, 2, 3, 2, 3, 4]

Thus we see that observation 0 (the northwest cell of our 3x3 lattice) is a first order neighbor to observations 1 and 3, second order
neighbor to observations 2, 4, and 6, a third order neighbor to 5, and 7, and a fourth order neighbor to observation 8 (the extreme southeast 
cell in the lattice). The position of the -1 simply denotes the focal unit.

Higher Order Contiguity Weights
-------------------------------

Closely related to the shortest path distances is the concept of a spatial weight based on a particular order of contiguity. For example, we could
define the second order contiguity relations using:

.. doctest::

    >>> w2 = pysal.higher_order(w, 2)
    >>> w2.neighbors[0]
    [4, 6, 2]

or a fourth order set of weights:

.. doctest::

    >>> w4 = pysal.higher_order(w, 4)
    WARNING: there are 5 disconnected observations
    Island ids:  [1, 3, 4, 5, 7]
    >>> w4.neighbors[0]
    [8]

In both cases a new instance of the W class is returned with the weights and neighbors defined using the particular order of contiguity.

Spatial Lag
-----------

The final function related to spatial weights that we illustrate here is used to construct a new variable called the spatial lag. The spatial
lag is a function of the attribute values observed at neighboring locations. For example, if we continue with our regular 3x3 lattice and
create an attribute variable y:

.. doctest::

    >>> import numpy as np
    >>> y = np.arange(w.n)
    >>> y
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])

then the spatial lag can be constructed with:
    
.. doctest::

    >>> yl = pysal.lag_spatial(w,y)
    >>> yl
    array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])

Mathematically, the spatial lag is a weighted sum of neighboring attribute values

.. math::
    
    yl_i = \sum_j w_{i,j} y_j

In the example above, the weights were binary, based on the rook criterion. If we row standardize our W object first
and then recalculate the lag, it takes the form of a weighted average of the neighboring attribute values:

.. doctest::

    >>> w.transform = 'r'
    >>> ylr = pysal.lag_spatial(w,y)
    >>> ylr
    array([ 2.        ,  2.        ,  3.        ,  3.33333333,  4.        ,
            4.66666667,  5.        ,  6.        ,  6.        ])

.. _id_order:

One important consideration in calculating the spatial lag is that the ordering
of the values in y aligns with the underlying order in W.  In cases where the
source for your attribute data is different from the one to construct your
weights you may need to reorder your y values accordingly.  To check if this is
the case you can find the order in W as follows:

.. doctest::

    >>> w.id_order
    [0, 1, 2, 3, 4, 5, 6, 7, 8]

In this case the lag_spatial function assumes that the first value in the y
attribute corresponds to unit 0 in the lattice (northwest cell), while the last
value in y would correspond to unit 8 (southeast cell). In other words, for the
value of the spatial lag to be valid the number of elements in y must match w.n
and the orderings must be aligned. 

Fortunately, for the common use case where both the attribute and weights information come from a
shapefile (and its dbf), PySAL handles the alignment automatically: [#]_

.. doctest::

    >>> w = pysal.rook_from_shapefile("../pysal/examples/columbus.shp")
    >>> f = pysal.open("../pysal/examples/columbus.dbf")
    >>> f.header
    ['AREA', 'PERIMETER', 'COLUMBUS_', 'COLUMBUS_I', 'POLYID', 'NEIG', 'HOVAL', 'INC', 'CRIME', 'OPEN', 'PLUMB', 'DISCBD', 'X', 'Y', 'NSA', 'NSB', 'EW', 'CP', 'THOUS', 'NEIGNO']
    >>> y = np.array(f.by_col['INC'])
    >>> w.transform = 'r'
    >>> y
    array([ 19.531   ,  21.232   ,  15.956   ,   4.477   ,  11.252   ,
            16.028999,   8.438   ,  11.337   ,  17.586   ,  13.598   ,
             7.467   ,  10.048   ,   9.549   ,   9.963   ,   9.873   ,
             7.625   ,   9.798   ,  13.185   ,  11.618   ,  31.07    ,
            10.655   ,  11.709   ,  21.155001,  14.236   ,   8.461   ,
             8.085   ,  10.822   ,   7.856   ,   8.681   ,  13.906   ,
            16.940001,  18.941999,   9.918   ,  14.948   ,  12.814   ,
            18.739   ,  17.017   ,  11.107   ,  18.476999,  29.833   ,
            22.207001,  25.872999,  13.38    ,  16.961   ,  14.135   ,
            18.323999,  18.950001,  11.813   ,  18.796   ])
    >>> yl = pysal.lag_spatial(w,y)
    >>> yl
    array([ 18.594     ,  13.32133333,  14.123     ,  14.94425   ,
            11.817857  ,  14.419     ,  10.283     ,   8.3364    ,
            11.7576665 ,  19.48466667,  10.0655    ,   9.1882    ,
             9.483     ,  10.07716667,  11.231     ,  10.46185714,
            21.94100033,  10.8605    ,  12.46133333,  15.39877778,
            14.36333333,  15.0838    ,  19.93666633,  10.90833333,
             9.7       ,  11.403     ,  15.13825   ,  10.448     ,
            11.81      ,  12.64725   ,  16.8435    ,  26.0662505 ,
            15.6405    ,  18.05175   ,  15.3824    ,  18.9123996 ,
            12.2418    ,  12.76675   ,  18.5314995 ,  22.79225025,
            22.575     ,  16.8435    ,  14.2066    ,  14.20075   ,
            15.2515    ,  18.6079995 ,  26.0200005 ,  15.818     ,  14.303     ])
    
    >>> w.id_order
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]

Non-Zero Diagonal
-----------------
The typical weights matrix has zeros along the main diagonal. This has the
practical result of excluding the self from any computation.  However, this is
not always the desired situation, and so PySAL offers a function that adds
values to the main diagonal of a W object.

As an example, we can build a basic rook weights matrix, which has zeros on
the diagonal, then insert ones along the diagonal: 

.. doctest::

    >>> w = pysal.lat2W(5, 5, id_type='string')
    >>> w['id0']
    {'id5': 1.0, 'id1': 1.0}
    >>> w_const = pysal.weights.insert_diagonal(w)
    >>> w_const['id0']
    {'id5': 1.0, 'id0': 1.0, 'id1': 1.0}

The default is to add ones to the diagonal, but the function allows any values to
be added.


WSets
=====

PySAL offers set-like manipulation of spatial weights matrices. While a W is
more complex than a set, the two objects have a number of commonalities
allowing for traditional set operations to have similar functionality on a W.
Conceptually, we treat each neighbor pair as an element of a set, and then
return the appropriate pairs based on the operation invoked (e.g. union,
intersection, etc.).  A key distinction between a set and a W is that a W
must keep track of the universe of possible pairs, even those that do not
result in a neighbor relationship.  

PySAL follows the naming conventions for Python sets, but adds optional flags
allowing the user to control the shape of the weights object returned.  At
this time, all the functions discussed in this section return a binary W no
matter the weights objects passed in.

Union
-----

The union of two weights objects returns a binary weights object, W, that
includes all neighbor pairs that exist in either weights object.  This
function can be used to simply join together two weights objects, say one for
Arizona counties and another for California counties.  It can also be used 
to join two weights objects that overlap as in the example below. 

.. doctest::

    >>> w1 = pysal.lat2W(4,4)
    >>> w2 = pysal.lat2W(6,4)
    >>> w = pysal.w_union(w1, w2)
    >>> w1[0] == w[0]
    True
    >>> w1.neighbors[15]
    [11, 14]
    >>> w2.neighbors[15]
    [11, 14, 19]
    >>> w.neighbors[15]
    [19, 11, 14]

Intersection
------------

The intersection of two weights objects returns a binary weights object, W,
that includes only those neighbor pairs that exist in both weights objects.
Unlike the union case, where all pairs in either matrix are returned, the
intersection only returns a subset of the pairs.  This leaves open the
question of the shape of the weights matrix to return.  For example, you have
one weights matrix of census tracts for City A and second matrix of tracts for
Utility Company B's service area, and want to find the W for the tracts that
overlap.  Depending on the research question, you may want the returned W to
have the same dimensions as City A's weights matrix, the same as the utility
company's weights matrix, a new dimensionality based on all the census tracts
in either matrix or with the dimensionality of just those tracts in the
overlapping area. All of these options are available via the w_shape parameter
and the order that the matrices are passed to the function.  The following
example uses the all case:

.. doctest::

    >>> w1 = pysal.lat2W(4,4)
    >>> w2 = pysal.lat2W(6,4)
    >>> w = pysal.w_intersection(w1, w2, 'all')
    WARNING: there are 8 disconnected observations
    Island ids:  [16, 17, 18, 19, 20, 21, 22, 23]
    >>> w1[0] == w[0]
    True
    >>> w1.neighbors[15]
    [11, 14]
    >>> w2.neighbors[15]
    [11, 14, 19]
    >>> w.neighbors[15]
    [11, 14]
    >>> w2.neighbors[16]
    [12, 20, 17]
    >>> w.neighbors[16]
    []

.. _difference:

Difference
----------

The difference of two weights objects returns a binary weights object, W, that
includes only neighbor pairs from the first object that are not in the second.
Similar to the intersection function, the user must select the shape of the
weights object returned using the w_shape parameter.  The user must also
consider the constrained parameter which controls whether the observations and
the neighbor pairs are differenced or just the neighbor pairs are differenced.
If you were to apply the difference function to our city and utility company
example from the intersection section above, you must decide whether or not
pairs that exist along the border of the regions should be considered
different or not.  It boils down to whether the tracts should be differenced
first and then the differenced pairs identified (constrained=True), or if the
differenced pairs should be identified based on the sets of pairs in the
original weights matrices (constrained=False).  In the example below we
difference weights matrices from regions with partial overlap.

.. doctest::

    >>> w1 = pysal.lat2W(6,4)
    >>> w2 = pysal.lat2W(4,4)
    >>> w1.neighbors[15]
    [11, 14, 19]
    >>> w2.neighbors[15]
    [11, 14]
    >>> w = pysal.w_difference(w1, w2, w_shape = 'w1', constrained = False)
    WARNING: there are 12 disconnected observations
    Island ids:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    >>> w.neighbors[15]
    [19]
    >>> w.neighbors[19]
    [15, 18, 23]
    >>> w = pysal.w_difference(w1, w2, w_shape = 'min', constrained = False)
    >>> 15 in w.neighbors
    False
    >>> w.neighbors[19]
    [18, 23]
    >>> w = pysal.w_difference(w1, w2, w_shape = 'w1', constrained = True)
    WARNING: there are 16 disconnected observations
    Island ids:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    >>> w.neighbors[15]
    []
    >>> w.neighbors[19]
    [18, 23]
    >>> w = pysal.w_difference(w1, w2, w_shape = 'min', constrained = True)
    >>> 15 in w.neighbors
    False
    >>> w.neighbors[19]
    [18, 23]

The difference function can be used to construct a bishop
:ref:`contiguity weights matrix <contiguity>` 
by differencing a queen and rook matrix.

.. doctest::

        >>> wr = pysal.lat2W(5,5)
        >>> wq = pysal.lat2W(5,5,rook = False)
        >>> wb = pysal.w_difference(wq, wr,constrained = False)
        >>> wb.neighbors[0]
        [6]


Symmetric Difference
--------------------

Symmetric difference of two weights objects returns a binary weights object,
W, that includes only neighbor pairs that are not shared by either matrix.
This function offers options similar to those in the difference function
described above.

.. doctest::

    >>> w1 = pysal.lat2W(6, 4)
    >>> w2 = pysal.lat2W(2, 4)
    >>> w_lower = pysal.w_difference(w1, w2, w_shape = 'min', constrained = True)
    >>> w_upper = pysal.lat2W(4, 4)
    >>> w = pysal.w_symmetric_difference(w_lower, w_upper, 'all', False)
    >>> w_lower.id_order
    [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    >>> w_upper.id_order
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    >>> w.id_order
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    >>> w.neighbors[11]
    [7]
    >>> w = pysal.w_symmetric_difference(w_lower, w_upper, 'min', False)
    WARNING: there are 8 disconnected observations
    Island ids:  [0, 1, 2, 3, 4, 5, 6, 7]
    >>> 11 in w.neighbors
    False
    >>> w.id_order
    [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
    >>> w = pysal.w_symmetric_difference(w_lower, w_upper, 'all', True)
    WARNING: there are 16 disconnected observations
    Island ids:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    >>> w.neighbors[11]
    []
    >>> w = pysal.w_symmetric_difference(w_lower, w_upper, 'min', True)
    WARNING: there are 8 disconnected observations
    Island ids:  [0, 1, 2, 3, 4, 5, 6, 7]
    >>> 11 in w.neighbors
    False

Subset
------

Subset of a weights object returns a binary weights object, W, that includes
only those observations provided by the user.  It also can be used to add
islands to a previously existing weights object.

.. doctest::

    >>> w1 = pysal.lat2W(6, 4)
    >>> w1.id_order
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    >>> ids = range(16)
    >>> w = pysal.w_subset(w1, ids)
    >>> w.id_order
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    >>> w1[0] == w[0]
    True
    >>> w1.neighbors[15]
    [11, 14, 19]
    >>> w.neighbors[15]
    [11, 14]


.. _wsp:

WSP
===
A thin PySAL weights object is available to users with extremely large weights
matrices, on the order of 2 million or more observations, or users interested
in holding many large weights matrices in RAM simultaneously. The
:class:`pysal.weights.WSP` is a thin weights object that does not include the
neighbors and weights dictionaries, but does contain the scipy.sparse form of
the weights.  For many PySAL functions the W and WSP objects can be used
interchangeably.  

A WSP object can be constructed from a `Matrix Market
<http://math.nist.gov/MatrixMarket/>`_ file (see :ref:`mtx` for more info on
reading and writing mtx files in PySAL):

.. doctest::

    >>> mtx = pysal.open("../pysal/examples/wmat.mtx", 'r')
    >>> wsp = mtx.read(sparse=True)

or built directly from a scipy.sparse object:

.. doctest::

    >>> import scipy.sparse
    >>> rows = [0, 1, 1, 2, 2, 3]
    >>> cols = [1, 0, 2, 1, 3, 3]
    >>> weights =  [1, 0.75, 0.25, 0.9, 0.1, 1]
    >>> sparse = scipy.sparse.csr_matrix((weights, (rows, cols)), shape=(4,4))
    >>> w = pysal.weights.WSP(sparse)

The WSP object has subset of the attributes of a W object; for example:

.. doctest::

    >>> w.n
    4
    >>> w.s0
    4.0
    >>> w.trcWtW_WW
    6.3949999999999996

The following functionality is available to convert from a W to a WSP:

.. doctest::

    >>> w = pysal.weights.lat2W(5,5)
    >>> w.s0
    80.0
    >>> wsp = pysal.weights.WSP(w.sparse)
    >>> wsp.s0
    80.0

and from a WSP to W:

.. doctest::

    >>> sp = pysal.weights.lat2SW(5, 5)
    >>> wsp = pysal.weights.WSP(sp)
    >>> wsp.s0
    80
    >>> w = pysal.weights.WSP2W(wsp)
    >>> w.s0
    80
    

Further Information 
====================

For further details see the :doc:`Weights  API <../../library/weights/index>`.



.. rubric:: Footnotes

.. [#] Although this tutorial provides an introduction to the functionality of the PySAL weights class, it is not exhaustive. Complete documentation for the class and associated functions can be found by accessing the help from within a Python interpreter. 
.. [#] The dictionaries for the weights and value attributes in W are read-only.
.. [#] Ties at the k-nn distance band are randomly broken to ensure each observation has exactly k neighbors.
.. [#] If the shapefile contains geographical coordinates these distance calculations will be misleading and the user should first project their coordinates using a GIS.
.. [#] The ordering exploits the one-to-one relation between a record in the DBF file and the shape in the shapefile.
	
	
