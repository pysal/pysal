"""
Weights.
"""
__author__ = "Sergio J. Rey <srey@asu.edu> "

import pysal
import math
import numpy as np
import scipy.sparse
from os.path import basename as BASENAME
from pysal.weights import util

__all__ = ['W', 'WSP']


class W(object):
    """
    Spatial weights.

    Parameters
    ----------
    neighbors       : dictionary
                      key is region ID, value is a list of neighbor IDS
                      Example:  {'a':['b'],'b':['a','c'],'c':['b']}
    weights : dictionary
                      key is region ID, value is a list of edge weights
                      If not supplied all edge weights are assumed to have a weight of 1.
                      Example: {'a':[0.5],'b':[0.5,1.5],'c':[1.5]}
    id_order : list
                      An ordered list of ids, defines the order of
                      observations when iterating over W if not set,
                      lexicographical ordering is used to iterate and the
                      id_order_set property will return False.  This can be
                      set after creation by setting the 'id_order' property.
    silent_island_warning   : boolean
                            By default PySAL will print a warning if the
                            dataset contains any disconnected observations or
                            islands. To silence this warning set this
                            parameter to True.
    ids : list
                      values to use for keys of the neighbors and weights dicts

    Attributes
    ----------

    asymmetries         : list
                          of
    cardinalities       : dictionary
                          of
    diagW2              : array
                          of
    diagWtW             : array
                          of
    diagWtW_WW          : array
                          of
    histogram           : dictionary
                          of
    id2i                : dictionary
                          of
    id_order            : list
                          of
    id_order_set        : boolean
                          True if
    islands             : list
                          of

    max_neighbors       : int
                          maximum number of neighbors

    mean_neighbors      : int
                          mean number of neighbors

    min_neighbors       : int
                          minimum neighbor count
    n                   : int
                          of

    neighbor_offsets    : list
                          ids of neighbors to a region in id_order
    nonzero             : int
                          Number of non-zero entries
    pct_nonzero         : float
                          Percentage of nonzero neighbor counts
    s0                  : float
                          of
    s1                  : float
                          of
    s2                  : float
                          of
    s2array             : array
                          of
    sd                  : float
                          of
    sparse              : sparse_matrix
                          SciPy sparse matrix instance
    trcW2               : float
                          of
    trcWtW              : float
                          of
    trcWtW_WW           : float
                          of
    transform           : string
                          of

    Examples
    --------
    >>> from pysal import W, lat2W
    >>> neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
    >>> weights = {0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1], 4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1], 8: [1, 1]}
    >>> w = W(neighbors, weights)
    >>> "%.3f"%w.pct_nonzero
    '29.630'

    Read from external gal file

    >>> import pysal
    >>> w = pysal.open(pysal.examples.get_path("stl.gal")).read()
    >>> w.n
    78
    >>> "%.3f"%w.pct_nonzero
    '6.542'

    Set weights implicitly

    >>> neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
    >>> w = W(neighbors)
    >>> round(w.pct_nonzero,3)
    29.63
    >>> w = lat2W(100, 100)
    >>> w.trcW2
    39600.0
    >>> w.trcWtW
    39600.0
    >>> w.transform='r'
    >>> round(w.trcW2, 3)
    2530.722
    >>> round(w.trcWtW, 3)
    2533.667

    Cardinality Histogram

    >>> w=pysal.rook_from_shapefile(pysal.examples.get_path("sacramentot2.shp"))
    >>> w.histogram
    [(1, 1), (2, 6), (3, 33), (4, 103), (5, 114), (6, 73), (7, 35), (8, 17), (9, 9), (10, 4), (11, 4), (12, 3), (13, 0), (14, 1)]

    Disconnected observations (islands)

    >>> w = pysal.W({1:[0],0:[1],2:[], 3:[]})
    WARNING: there are 2 disconnected observations
    Island ids:  [2, 3]

    """

    def __init__(self, neighbors, weights=None, id_order=None,
        silent_island_warning=False, ids=None):
        self.silent_island_warning = silent_island_warning
        self.transformations = {}
        self.neighbors = neighbors
        if not weights:
            weights = {}
            for key in neighbors:
                weights[key] = [1.] * len(neighbors[key])
        self.weights = weights
        self.transformations['O'] = self.weights.copy()  # original weights
        self.transform = 'O'
        if id_order is None:
            self._id_order = self.neighbors.keys()
            self._id_order.sort()
            self._id_order_set = False
        else:
            self._id_order = id_order
            self._id_order_set = True
        self._reset()
        self._n = len(self.weights)
        if self.islands and not self.silent_island_warning:
            ni = len(self.islands)
            if ni == 1:
                print "WARNING: there is one disconnected observation (no neighbors)"
                print "Island id: ", self.islands
            else:
                print "WARNING: there are %d disconnected observations" % ni
                print "Island ids: ", self.islands

    def _reset(self):
        """Reset properties.

        """
        self._cache = {}

    @property
    def sparse(self):
        """Sparse matrix object.

        For any matrix manipulations required for w, w.sparse should be
        used. This is based on scipy.sparse.

        """
        if 'sparse' not in self._cache:
            self._sparse = self._build_sparse()
            self._cache['sparse'] = self._sparse
        return self._sparse

    def _build_sparse(self):
        """Construct the sparse attribute.

        """

        row = []
        col = []
        data = []
        id2i = self.id2i
        for i, neigh_list in self.neighbor_offsets.iteritems():
            card = self.cardinalities[i]
            row.extend([id2i[i]] * card)
            col.extend(neigh_list)
            data.extend(self.weights[i])
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        s = scipy.sparse.csr_matrix((data, (row, col)), shape=(self.n, self.n))
        return s

    @property
    def id2i(self):
        """Dictionary where the key is an ID and the value is that ID's
        index in W.id_order.

        """
        if 'id2i' not in self._cache:
            self._id2i = {}
            for i, id_i in enumerate(self._id_order):
                self._id2i[id_i] = i
            self._id2i = self._id2i
            self._cache['id2i'] = self._id2i
        return self._id2i

    @property
    def n(self):
        """Number of units.

        """
        if "n" not in self._cache:
            self._n = len(self.neighbors)
            self._cache['n'] = self._n
        return self._n

    @property
    def s0(self):
        """s0 is defined as

        .. math::

               s0=\sum_i \sum_j w_{i,j}

        """
        if 's0' not in self._cache:
            self._s0 = self.sparse.sum()
            self._cache['s0'] = self._s0
        return self._s0

    @property
    def s1(self):
        """s1 is defined as

        .. math::

               s1=1/2 \sum_i \sum_j (w_{i,j} + w_{j,i})^2

        """
        if 's1' not in self._cache:
            t = self.sparse.transpose()
            t = t + self.sparse
            t2 = t.multiply(t)  # element-wise square
            self._s1 = t2.sum() / 2.
            self._cache['s1'] = self._s1
        return self._s1

    @property
    def s2array(self):
        """Individual elements comprising s2.

        See Also
        --------
        s2

        """
        if 's2array' not in self._cache:
            s = self.sparse
            self._s2array = np.array(s.sum(1) + s.sum(0).transpose()) ** 2
            self._cache['s2array'] = self._s2array
        return self._s2array

    @property
    def s2(self):
        """s2 is defined as

        .. math::

                s2=\sum_j (\sum_i w_{i,j} + \sum_i w_{j,i})^2

        """
        if 's2' not in self._cache:
            self._s2 = self.s2array.sum()
            self._cache['s2'] = self._s2
        return self._s2

    @property
    def trcW2(self):
        """Trace of :math:`WW`.

        See Also
        --------
        diagW2

        """
        if 'trcW2' not in self._cache:
            self._trcW2 = self.diagW2.sum()
            self._cache['trcw2'] = self._trcW2
        return self._trcW2

    @property
    def diagW2(self):
        """Diagonal of :math:`WW`.

        See Also
        --------
        trcW2

        """
        if 'diagw2' not in self._cache:
            self._diagW2 = (self.sparse * self.sparse).diagonal()
            self._cache['diagW2'] = self._diagW2
        return self._diagW2

    @property
    def diagWtW(self):
        """Diagonal of :math:`W^{'}W`.

        See Also
        --------
        trcWtW

        """
        if 'diagWtW' not in self._cache:
            self._diagWtW = (self.sparse.transpose() * self.sparse).diagonal()
            self._cache['diagWtW'] = self._diagWtW
        return self._diagWtW

    @property
    def trcWtW(self):
        """Trace of :math:`W^{'}W`.

        See Also
        --------
        diagWtW

        """
        if 'trcWtW' not in self._cache:
            self._trcWtW = self.diagWtW.sum()
            self._cache['trcWtW'] = self._trcWtW
        return self._trcWtW

    @property
    def diagWtW_WW(self):
        """Diagonal of :math:`W^{'}W + WW`.

        """
        if 'diagWtW_WW' not in self._cache:
            wt = self.sparse.transpose()
            w = self.sparse
            self._diagWtW_WW = (wt * w + w * w).diagonal()
            self._cache['diagWtW_WW'] = self._diagWtW_WW
        return self._diagWtW_WW

    @property
    def trcWtW_WW(self):
        """Trace of :math:`W^{'}W + WW`.

        """
        if 'trcWtW_WW' not in self._cache:
            self._trcWtW_WW = self.diagWtW_WW.sum()
            self._cache['trcWtW_WW'] = self._trcWtW_WW
        return self._trcWtW_WW

    @property
    def pct_nonzero(self):
        """Percentage of nonzero weights.

        """
        if 'pct_nonzero' not in self._cache:
            self._pct_nonzero = 100. * self.sparse.nnz / (1. * self._n ** 2)
            self._cache['pct_nonzero'] = self._pct_nonzero
        return self._pct_nonzero

    @property
    def cardinalities(self):
        """Number of neighbors for each observation.

        """
        if 'cardinalities' not in self._cache:
            c = {}
            for i in self._id_order:
                c[i] = len(self.neighbors[i])
            self._cardinalities = c
            self._cache['cardinalities'] = self._cardinalities
        return self._cardinalities

    @property
    def max_neighbors(self):
        """Largest number of neighbors.

        """
        if 'max_neighbors' not in self._cache:
            self._max_neighbors = max(self.cardinalities.values())
            self._cache['max_neighbors'] = self._max_neighbors
        return self._max_neighbors

    @property
    def mean_neighbors(self):
        """Average number of neighbors.

        """
        if 'mean_neighbors' not in self._cache:
            self._mean_neighbors = np.mean(self.cardinalities.values())
            self._cache['mean_neighbors'] = self._mean_neighbors
        return self._mean_neighbors

    @property
    def min_neighbors(self):
        """Minimum number of neighbors.

        """
        if 'min_neighbors' not in self._cache:
            self._min_neighbors = min(self.cardinalities.values())
            self._cache['min_neighbors'] = self._min_neighbors
        return self._min_neighbors

    @property
    def nonzero(self):
        """Number of nonzero weights.

        """
        if 'nonzero' not in self._cache:
            self._nonzero = self.sparse.nnz
            self._cache['nonzero'] = self._nonzero
        return self._nonzero

    @property
    def sd(self):
        """Standard deviation of number of neighbors.

        """
        if 'sd' not in self._cache:
            self._sd = np.std(self.cardinalities.values())
            self._cache['sd'] = self._sd
        return self._sd

    @property
    def asymmetries(self):
        """List of id pairs with asymmetric weights.

        """
        if 'asymmetries' not in self._cache:
            self._asymmetries = self.asymmetry()
            self._cache['asymmetries'] = self._asymmetries
        return self._asymmetries

    @property
    def islands(self):
        """List of ids without any neighbors.

        """
        if 'islands' not in self._cache:
            self._islands = [i for i,
                             c in self.cardinalities.items() if c == 0]
            self._cache['islands'] = self._islands
        return self._islands

    @property
    def histogram(self):
        """Cardinality histogram as a dictionary where key is the id and
        value is the number of neighbors for that unit.

        """
        if 'histogram' not in self._cache:
            ct, bin = np.histogram(self.cardinalities.values(),
                                   range(self.min_neighbors, self.max_neighbors + 2))
            self._histogram = zip(bin, ct)
            self._cache['histogram'] = self._histogram
        return self._histogram

    def __getitem__(self, key):
        """Allow a dictionary like interaction with the weights class.

        Examples
        --------
        >>> from pysal import rook_from_shapefile as rfs
        >>> w = rfs(pysal.examples.get_path("10740.shp"))
        WARNING: there is one disconnected observation (no neighbors)
        Island id:  [163]
        >>> w[163]
        {}
        >>> w[0]
        {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        """
        return dict(zip(self.neighbors[key], self.weights[key]))

    def __iter__(self):
        """
        Support iteration over weights.

        Examples
        --------
        >>> import pysal
        >>> w=pysal.lat2W(3,3)
        >>> for i,wi in enumerate(w):
        ...     print i,wi
        ...
        0 (0, {1: 1.0, 3: 1.0})
        1 (1, {0: 1.0, 2: 1.0, 4: 1.0})
        2 (2, {1: 1.0, 5: 1.0})
        3 (3, {0: 1.0, 4: 1.0, 6: 1.0})
        4 (4, {1: 1.0, 3: 1.0, 5: 1.0, 7: 1.0})
        5 (5, {8: 1.0, 2: 1.0, 4: 1.0})
        6 (6, {3: 1.0, 7: 1.0})
        7 (7, {8: 1.0, 4: 1.0, 6: 1.0})
        8 (8, {5: 1.0, 7: 1.0})
        >>>
        """
        for i in self._id_order:
            yield i, dict(zip(self.neighbors[i], self.weights[i]))

    def remap_ids(self, new_ids):
        '''
        In place modification throughout `W` of id values from `w.id_order` to
        `new_ids` in all

        ...

        Arguments
        ---------

        new_ids     :   list
                        /ndarray
                        Aligned list of new ids to be inserted. Note that first
                        element of new_ids will replace first element of
                        w.id_order, second element of new_ids replaces second
                        element of w.id_order and so on.

        Example
        -------

        >>> import pysal as ps
        >>> w = ps.lat2W(3, 3)
        >>> w.id_order
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> w.neighbors[0]
        [3, 1]
        >>> new_ids = ['id%i'%id for id in w.id_order]
        >>> _ = w.remap_ids(new_ids)
        >>> w.id_order
        ['id0', 'id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8']
        >>> w.neighbors['id0']
        ['id3', 'id1']
        '''
        old_ids = self._id_order
        if len(old_ids) != len(new_ids):
            raise Exception("W.remap_ids: length of `old_ids` does not match \
            that of new_ids")
        if len(set(new_ids)) != len(new_ids):
            raise Exception("W.remap_ids: list `new_ids` contains duplicates")
        else:
            new_neighbors = {}
            new_weights = {}
            old_transformations = self.transformations['O'].copy()
            new_transformations = {}
            for o,n in zip(old_ids, new_ids):
                o_neighbors = self.neighbors[o]
                o_weights = self.weights[o]
                n_neighbors = [ new_ids[old_ids.index(j)] for j in o_neighbors]
                new_neighbors[n] = n_neighbors
                new_weights[n] = o_weights[:]
                new_transformations[n] = old_transformations[o]
            self.neighbors = new_neighbors
            self.weights = new_weights
            self.transformations["O"] = new_transformations

            id_order = [ self._id_order.index(o) for o in old_ids]
            for i,id_ in enumerate(id_order):
                self.id_order[id_] = new_ids[i]

            self._reset()

    def __set_id_order(self, ordered_ids):
        """
        Set the iteration order in w.

        W can be iterated over. On construction the iteration order is set to
        the lexicographic order of the keys in the w.weights dictionary. If a specific order
        is required it can be set with this method.

        Parameters
        ----------

        ordered_ids : sequence
                      identifiers for observations in specified order

        Notes
        -----

        ordered_ids is checked against the ids implied by the keys in
        w.weights. If they are not equivalent sets an exception is raised and
        the iteration order is not changed.

        Examples
        --------

        >>> import pysal
        >>> w=pysal.lat2W(3,3)
        >>> for i,wi in enumerate(w):
        ...     print i,wi
        ...
        0 (0, {1: 1.0, 3: 1.0})
        1 (1, {0: 1.0, 2: 1.0, 4: 1.0})
        2 (2, {1: 1.0, 5: 1.0})
        3 (3, {0: 1.0, 4: 1.0, 6: 1.0})
        4 (4, {1: 1.0, 3: 1.0, 5: 1.0, 7: 1.0})
        5 (5, {8: 1.0, 2: 1.0, 4: 1.0})
        6 (6, {3: 1.0, 7: 1.0})
        7 (7, {8: 1.0, 4: 1.0, 6: 1.0})
        8 (8, {5: 1.0, 7: 1.0})
        >>> w.id_order
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> w.id_order=range(8,-1,-1)
        >>> w.id_order
        [8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> for i,w_i in enumerate(w):
        ...     print i,w_i
        ...
        0 (8, {5: 1.0, 7: 1.0})
        1 (7, {8: 1.0, 4: 1.0, 6: 1.0})
        2 (6, {3: 1.0, 7: 1.0})
        3 (5, {8: 1.0, 2: 1.0, 4: 1.0})
        4 (4, {1: 1.0, 3: 1.0, 5: 1.0, 7: 1.0})
        5 (3, {0: 1.0, 4: 1.0, 6: 1.0})
        6 (2, {1: 1.0, 5: 1.0})
        7 (1, {0: 1.0, 2: 1.0, 4: 1.0})
        8 (0, {1: 1.0, 3: 1.0})
        >>>

        """

        if set(self._id_order) == set(ordered_ids):
            self._id_order = ordered_ids
            self._id_order_set = True
            self._reset()
        else:
            raise Exception('ordered_ids do not align with W ids')

    def __get_id_order(self):
        """Returns the ids for the observations in the order in which they
        would be encountered if iterating over the weights.

        """
        return self._id_order

    id_order = property(__get_id_order, __set_id_order)

    @property
    def id_order_set(self):
        """
        Returns True if user has set id_order, False if not.

        Examples
        --------
        >>> from pysal import lat2W
        >>> w=lat2W()
        >>> w.id_order_set
        True
        """
        return self._id_order_set

    @property
    def neighbor_offsets(self):
        """
        Given the current id_order, neighbor_offsets[id] is the offsets of the
        id's neighbors in id_order.

        Returns
        -------
        list
                offsets of the id's neighbors in id_order

        Examples
        --------
        >>> from pysal import W
        >>> neighbors={'c': ['b'], 'b': ['c', 'a'], 'a': ['b']}
        >>> weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
        >>> w=W(neighbors,weights)
        >>> w.id_order = ['a','b','c']
        >>> w.neighbor_offsets['b']
        [2, 0]
        >>> w.id_order = ['b','a','c']
        >>> w.neighbor_offsets['b']
        [2, 1]
        """

        if "neighbors_0" not in self._cache:
            self.__neighbors_0 = {}
            id2i = self.id2i
            for j, neigh_list in self.neighbors.iteritems():
                self.__neighbors_0[j] = [id2i[neigh] for neigh in neigh_list]
            self._cache['neighbors_0'] = self.__neighbors_0
        return self.__neighbors_0

    def get_transform(self):
        """
        Getter for transform property.

        Returns
        -------
        transformation : string (or none)

        Examples
        --------
        >>> from pysal import lat2W
        >>> w=lat2W()
        >>> w.weights[0]
        [1.0, 1.0]
        >>> w.transform
        'O'
        >>> w.transform='r'
        >>> w.weights[0]
        [0.5, 0.5]
        >>> w.transform='b'
        >>> w.weights[0]
        [1.0, 1.0]
        >>>

        """

        return self._transform

    def set_transform(self, value="B"):
        """
        Transformations of weights.

        Notes
        -----

        Transformations are applied only to the value of the weights at
        instantiation. Chaining of transformations cannot be done on a W
        instance.

        Parameters
        ----------
        transform   :   string
                        not case sensitive)

        .. table::

            ================   ======================================================
            transform string   value
            ================   ======================================================
            B                  Binary
            R                  Row-standardization (global sum=n)
            D                  Double-standardization (global sum=1)
            V                  Variance stabilizing
            O                  Restore original transformation (from instantiation)
            ================   ======================================================

        Examples
        --------
        >>> from pysal import lat2W
        >>> w=lat2W()
        >>> w.weights[0]
        [1.0, 1.0]
        >>> w.transform
        'O'
        >>> w.transform='r'
        >>> w.weights[0]
        [0.5, 0.5]
        >>> w.transform='b'
        >>> w.weights[0]
        [1.0, 1.0]
        >>>
        """
        value = value.upper()
        self._transform = value
        if value in self.transformations:
            self.weights = self.transformations[value]
            self._reset()
        else:
            if value == "R":
                # row standardized weights
                weights = {}
                self.weights = self.transformations['O']
                for i in self.weights:
                    wijs = self.weights[i]
                    row_sum = sum(wijs) * 1.0
                    if row_sum == 0.0:
                        if not self.silent_island_warning:
                            print 'WARNING: ', i, ' is an island (no neighbors)'
                    weights[i] = [wij / row_sum for wij in wijs]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "D":
                # doubly-standardized weights
                # update current chars before doing global sum
                self._reset()
                s0 = self.s0
                ws = 1.0 / s0
                weights = {}
                self.weights = self.transformations['O']
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i] = [wij * ws for wij in wijs]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "B":
                # binary transformation
                weights = {}
                self.weights = self.transformations['O']
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i] = [1.0 for wij in wijs]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "V":
                # variance stabilizing
                weights = {}
                q = {}
                k = self.cardinalities
                s = {}
                Q = 0.0
                self.weights = self.transformations['O']
                for i in self.weights:
                    wijs = self.weights[i]
                    q[i] = math.sqrt(sum([wij * wij for wij in wijs]))
                    s[i] = [wij / q[i] for wij in wijs]
                    Q += sum([si for si in s[i]])
                nQ = self.n / Q
                for i in self.weights:
                    weights[i] = [w * nQ for w in s[i]]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "O":
                # put weights back to original transformation
                weights = {}
                original = self.transformations[value]
                self.weights = original
                self._reset()
            else:
                print 'unsupported weights transformation'

    transform = property(get_transform, set_transform)

    def asymmetry(self, intrinsic=True):
        """
        Asymmetry check.

        Parameters
        ----------
        intrinsic   :   boolean
                        default=True

                intrinsic symmetry:
                      :math:`w_{i,j} == w_{j,i}`

                if intrisic is False:
                    symmetry is defined as :math:`i \in N_j \ AND \ j \in N_i` where
                    :math:`N_j` is the set of neighbors for j.

        Returns
        -------
        asymmetries : list
                      empty if no asymmetries are found
                      if asymmetries, then a list of (i,j) tuples is returned

        Examples
        --------

        >>> from pysal import lat2W
        >>> w=lat2W(3,3)
        >>> w.asymmetry()
        []
        >>> w.transform='r'
        >>> w.asymmetry()
        [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 1), (2, 5), (3, 0), (3, 4), (3, 6), (4, 1), (4, 3), (4, 5), (4, 7), (5, 2), (5, 4), (5, 8), (6, 3), (6, 7), (7, 4), (7, 6), (7, 8), (8, 5), (8, 7)]
        >>> result = w.asymmetry(intrinsic=False)
        >>> result
        []
        >>> neighbors={0:[1,2,3], 1:[1,2,3], 2:[0,1], 3:[0,1]}
        >>> weights={0:[1,1,1], 1:[1,1,1], 2:[1,1], 3:[1,1]}
        >>> w=W(neighbors,weights)
        >>> w.asymmetry()
        [(0, 1), (1, 0)]
        """

        if intrinsic:
            wd = self.sparse.transpose() - self.sparse
        else:
            transform = self.transform
            self.transform = 'b'
            wd = self.sparse.transpose() - self.sparse
            self.transform = transform

        ids = np.nonzero(wd)
        if len(ids[0]) == 0:
            return []
        else:
            ijs = zip(ids[0], ids[1])
            ijs.sort()
            return ijs

    def full(self):
        """
        Generate a full numpy array.

        Returns
        -------

        implicit : tuple
                   first element being the full numpy array and second element
                   keys being the ids associated with each row in the array.

        Examples
        --------
        >>> from pysal import W
        >>> neighbors={'first':['second'],'second':['first','third'],'third':['second']}
        >>> weights={'first':[1],'second':[1,1],'third':[1]}
        >>> w=W(neighbors,weights)
        >>> wf,ids=w.full()
        >>> wf
        array([[ 0.,  1.,  0.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  0.]])
        >>> ids
        ['first', 'second', 'third']

        See also
        --------
        full

        """
        return util.full(self)

    def towsp(self):
        '''
        Generate a WSP object.

        Returns
        -------

        implicit : pysal.WSP
                   Thin W class

        Examples
        --------
        >>> import pysal as ps
        >>> from pysal import W
        >>> neighbors={'first':['second'],'second':['first','third'],'third':['second']}
        >>> weights={'first':[1],'second':[1,1],'third':[1]}
        >>> w=W(neighbors,weights)
        >>> wsp=w.towsp()
        >>> isinstance(wsp, ps.weights.weights.WSP)
        True
        >>> wsp.n
        3
        >>> wsp.s0
        4

        See also
        --------
        WSP

        '''
        return WSP(self.sparse, self._id_order)

    def set_shapefile(self, shapefile, idVariable=None, full=False):
        """
        Adding meta data for writing headers of gal and gwt files.

        Parameters
        ----------

        shapefile :     string
                        shapefile name used to construct weights

        idVariable :    string
                        name of attribute in shapefile to associate with ids in the weights

        full :          boolean
                        True - write out entire path for shapefile, False
                        (default) only base of shapefile without extension

        """

        if full:
            self._shpName = shapefile
        else:
            self._shpName = BASENAME(shapefile).split(".")[0]

        self._varName = idVariable


class WSP(object):

    """
    Thin W class for spreg.

    Parameters
    ----------

    sparse   : sparse_matrix
               NxN object from scipy.sparse

    id_order : list
               An ordered list of ids, assumed to match the ordering in
               sparse.

    Attributes
    ----------

    n           : int
                  description
    s0          : float
                  description
    trcWtW_WW   : float
                  description

    Examples
    --------

    From GAL information

    >>> import scipy.sparse
    >>> import pysal
    >>> rows = [0, 1, 1, 2, 2, 3]
    >>> cols = [1, 0, 2, 1, 3, 3]
    >>> weights =  [1, 0.75, 0.25, 0.9, 0.1, 1]
    >>> sparse = scipy.sparse.csr_matrix((weights, (rows, cols)), shape=(4,4))
    >>> w = pysal.weights.WSP(sparse)
    >>> w.s0
    4.0
    >>> w.trcWtW_WW
    6.3949999999999996
    >>> w.n
    4

    """

    def __init__(self, sparse, id_order=None):
        if not scipy.sparse.issparse(sparse):
            raise ValueError("must pass a scipy sparse object")
        rows, cols = sparse.shape
        if rows != cols:
            raise ValueError("Weights object must be square")
        self.sparse = sparse.tocsr()
        self.n = sparse.shape[0]
        if id_order:
            if len(id_order) != self.n:
                raise ValueError(
                    "Number of values in id_order must match shape of sparse")
        self.id_order = id_order
        self._cache = {}

    @property
    def s0(self):
        """s0 is defined as:

        .. math::

               s0=\sum_i \sum_j w_{i,j}

        """
        if 's0' not in self._cache:
            self._s0 = self.sparse.sum()
            self._cache['s0'] = self._s0
        return self._s0

    @property
    def trcWtW_WW(self):
        """Trace of :math:`W^{'}W + WW`.

        """
        if 'trcWtW_WW' not in self._cache:
            self._trcWtW_WW = self.diagWtW_WW.sum()
            self._cache['trcWtW_WW'] = self._trcWtW_WW
        return self._trcWtW_WW

    @property
    def diagWtW_WW(self):
        """Diagonal of :math:`W^{'}W + WW`.

        """
        if 'diagWtW_WW' not in self._cache:
            wt = self.sparse.transpose()
            w = self.sparse
            self._diagWtW_WW = (wt * w + w * w).diagonal()
            self._cache['diagWtW_WW'] = self._diagWtW_WW
        return self._diagWtW_WW
