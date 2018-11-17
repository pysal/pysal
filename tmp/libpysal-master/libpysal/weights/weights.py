
"""
Weights.
"""
__author__ = "Sergio J. Rey <srey@asu.edu> "

import copy
from os.path import basename as BASENAME
import math
import warnings
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import connected_components
#from .util import full, WSP2W resolve import cycle by
#forcing these into methods
from . import adjtools
from ..io.fileio import FileIO as popen

__all__ = ['W', 'WSP']

class W(object):
    """
    Spatial weights class.

    Parameters
    ----------
    neighbors            : dictionary
                           Key is region ID, value is a list of neighbor IDS.
                           Example:  {'a':['b'],'b':['a','c'],'c':['b']}
    weights              : dictionary
                           Key is region ID, value is a list of edge weights.
                           If not supplied all edge weights are assumed to have a weight of 1.
                           Example: {'a':[0.5],'b':[0.5,1.5],'c':[1.5]}
    id_order             : list
                           An ordered list of ids, defines the order of
                           observations when iterating over W if not set,
                           lexicographical ordering is used to iterate and the
                           id_order_set property will return False. This can be
                           set after creation by setting the 'id_order' property.
    silent_island_warning: boolean
                           By default libpysal will print a warning if the
                           dataset contains any disconnected observations or
                           islands. To silence this warning set this
                           parameter to True.
    silent_connected_components   : boolean
                            By default PySAL will print a warning if the
                            dataset contains any disconnected components in the
                            adjacency matrix. These are disconnected *groups*
                            of islands. To silence this warning set this
                            parameter to True.
    ids                  : list
                           Values to use for keys of the neighbors and weights dicts.

    Attributes (NOTE: these are described by their docstrings. to view, use the `help` function)
    ----------

    asymmetries
    cardinalities
    component_labels
    diagW2
    diagWtW
    diagWtW_WW
    histogram
    id2i
    id_order
    id_order_set
    islands
    max_neighbors
    mean_neighbors
    min_neighbors
    n
    n_components
    neighbor_offsets
    nonzero
    pct_nonzero
    s0
    s1
    s2
    s2array
    sd
    sparse
    trcW2
    trcWtW
    trcWtW_WW
    transform

    Examples
    --------
    >>> from libpysal.weights.weights import W
    >>> neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
    >>> weights = {0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1], 4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1], 8: [1, 1]}
    >>> w = W(neighbors, weights)
    >>> "%.3f"%w.pct_nonzero
    '29.630'

    Read from external gal file

    >>> import libpysal
    >>> w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
    >>> w.n
    78
    >>> "%.3f"%w.pct_nonzero
    '6.542'

    Set weights implicitly

    >>> neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
    >>> w = W(neighbors)
    >>> round(w.pct_nonzero,3)
    29.63
    >>> from libpysal.weights import lat2W
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
    >>> w.histogram
    [(2, 4), (3, 392), (4, 9604)]

    Disconnected observations (islands)

    >>> from libpysal.weights import W
    >>> w = W({1:[0],0:[1],2:[], 3:[]})

    WARNING: there are 2 disconnected observations
    Island ids:  [2, 3]

    """

    def __init__(self, neighbors, weights=None, id_order=None,
                 silence_warnings=False, ids=None):
        self.silent_island_warning = silence_warnings
        self.silent_connected_components = silence_warnings
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
            self._id_order = list(self.neighbors.keys())
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
                warnings.warn("There is one disconnected observation"
                              " (no neighbors).\nIsland id: {}"
                              .format(str(self.islands[0])), 
                              stacklevel=2)
            else:
                warnings.warn("There are %d disconnected observations" % ni + ' \n '
                              " Island ids: %s" % ', '.join(str(island) for island in self.islands))
        if self.n_components > 1 and not self.islands and not self.silent_connected_components:
            warnings.warn("The weights matrix is not fully connected. There are %d components" % self.n_components)

    def _reset(self):
        """Reset properties.

        """
        self._cache = {}

    @classmethod
    def from_file(cls, path='', format=None, **kwargs):
        f = popen(dataPath=path, mode='r', dataFormat=format)
        w = f.read(**kwargs)
        f.close()
        return w

    @classmethod
    def from_shapefile(cls, *args, **kwargs):
        # we could also just "do the right thing," but I think it'd make sense to
        # try and get people to use `Rook.from_shapefile(shapefile)` rather than
        # W.from_shapefile(shapefile, type=`rook`), otherwise we'd need to build
        # a type dispatch table. Generic W should be for stuff we don't know
        # anything about. 
        raise NotImplementedError('Use type-specific constructors, like Rook,'
                                  ' Queen, DistanceBand, or Kernel')

    @classmethod
    def from_WSP(cls, WSP, silence_warnings=True):
        return WSP2W(WSP, silence_warnings=silence_warnings)

    @classmethod
    def from_adjlist(cls, adjlist, focal_col='focal', 
                     neighbor_col='neighbor', weight_col=None):
        """
        Return an adjacency list representation of a weights object. 

        Parameters
        ----------
        adjlist         :   pandas DataFrame
                            adjacency list with a minimum of two columns
        focal_col       :   string
                            name of the column with the "source" node ids
        neighbor_col    :   string
                            name of the column with the "destination" node ids
        weight_col      :   string
                            name of the column with the weight information. If not provided and
                            the dataframe has no column named "weight" then all weights
                            are assumed to be 1.
        """
        if weight_col is None:
           weight_col = 'weight' 
        try_weightcol = getattr(adjlist, weight_col) 
        if try_weightcol is None:
            adjlist = adjlist.copy(deep=True)
            adjlist['weight'] = 1
        all_ids = set(adjlist[focal_col].tolist()) 
        all_ids |= set(adjlist[neighbor_col].tolist())
        grouper = adjlist.groupby(focal_col)
        neighbors = grouper[neighbor_col].apply(list).to_dict()
        weights = grouper[weight_col].apply(list).to_dict()
        neighbors.update({k:[] for k in 
                          all_ids.difference(list(neighbors.keys()))})
        weights.update({k:[] for k in 
                        all_ids.difference(list(weights.keys()))})
        return cls(neighbors=neighbors, weights=weights)

    def to_adjlist(self, remove_symmetric=False, 
                   focal_col='focal', neighbor_col='neighbor', weight_col='weight'):
        """
        Compute an adjacency list representation of a weights object.

        Parameters
        ----------
        remove_symmetric    :   bool
                            whether or not to remove ``symmetric'' entries. If the W is symmetric,
                            a standard ``directed'' adjacency list will contain both the forward and
                            backward links by default because adjacency lists are a directed
                            graph representation. If this is True, a W created from this adjacency list
                            **MAY NOT BE THE SAME** as the original W. If you would like to 
                            consider (1,2) and (2,1) as distinct links, leave this as "False". 
        focal_col       :   string
                            name of the column in which to store "source" node ids.
        neighbor_col    :   string
                            name of the column in which to store "destination" node ids.
        weight_col      :   string
                            name of the column in which to store weight information. 
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('pandas must be installed to use this method')
        adjlist = pd.DataFrame(((idx, n,w) for idx, neighb in self 
                                           for n,w in list(neighb.items())),
                               columns = ('focal', 'neighbor', 'weight'))
        return adjtools.filter_adjlist(adjlist) if remove_symmetric else adjlist

    def to_networkx(self):
        """
        Convert a weights object to a networkx graph

        Parameters
        ----------
        None

        Returns
        -------
        a networkx graph representation of the W object
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required to use this function.")
        G = nx.DiGraph() if len(self.asymmetries)>0 else nx.Graph()
        return nx.from_scipy_sparse_matrix(self.sparse, create_using=G)

    @classmethod
    def from_networkx(cls, graph, weight_col='weight'):
        """
        Convert a networkx graph to a PySAL W object.

        Parameters
        ----------
        graph       :   networkx graph
                        the graph to convert to a W
        weight_col  :   string
                        if the graph is labeled, this should be the
                        name of the field to use as the weight for
                        the W.
        Returns
        --------
        a pysal.weights.W object containing the same graph
        as the networkx graph
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required to use this function.")
        sparse_matrix = nx.to_scipy_sparse_matrix(graph)
        return WSP(sparse_matrix).to_W()
        neighbors = dict()
        weights = dict()
        for focal in graph.nodes():
            links = graph[focal]
            neighbors.update({focal:[]})
            weights.update({focal:[]})
            for neighbor, weight in list(links.items()):
                   neighbors[focal].append(neighbor)
                   if weight == {}:
                       weights[focal].append(1)
                   else:
                       weights[focal].append(weight[weight_col])
        return cls(neighbors=neighbors, weights=weights)

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

    @property
    def n_components(self):
        """Store whether the adjacency matrix is fully connected.
        """
        if 'n_components' not in self._cache:
            self._n_components, self._component_labels = connected_components(self.sparse)
            self._cache['n_components'] = self._n_components
            self._cache['component_labels'] = self._component_labels
        return self._n_components

    @property
    def component_labels(self):
        """Store the graph component in which each observation falls.
        """
        if 'component_labels' not in self._cache:
            self._n_components, self._component_labels = connected_components(self.sparse)
            self._cache['n_components'] = self._n_components
            self._cache['component_labels'] = self._component_labels
        return self._component_labels

    def _build_sparse(self):
        """Construct the sparse attribute.

        """

        row = []
        col = []
        data = []
        id2i = self.id2i
        for i, neigh_list in list(self.neighbor_offsets.items()):
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
            self._mean_neighbors = np.mean(list(self.cardinalities.values()))
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
            self._sd = np.std(list(self.cardinalities.values()))
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
                             c in list(self.cardinalities.items()) if c == 0]
            self._cache['islands'] = self._islands
        return self._islands

    @property
    def histogram(self):
        """Cardinality histogram as a dictionary where key is the id and
        value is the number of neighbors for that unit.

        """
        if 'histogram' not in self._cache:
            ct, bin = np.histogram(list(self.cardinalities.values()),
                                   list(range(self.min_neighbors, self.max_neighbors + 2)))
            self._histogram = list(zip(bin, ct))
            self._cache['histogram'] = self._histogram
        return self._histogram

    def __getitem__(self, key):
        """Allow a dictionary like interaction with the weights class.

        Examples
        --------
        >>> from libpysal.weights import lat2W
        >>> w = lat2W()

        >>> w[0] == dict({1: 1.0, 5: 1.0})
        True
        """
        return dict(list(zip(self.neighbors[key], self.weights[key])))

    def __iter__(self):
        """
        Support iteration over weights.

        Examples
        --------
        >>> from libpysal.weights import lat2W
        >>> w=lat2W(3,3)
        >>> for i,wi in enumerate(w):
        ...     print(i,wi[0])
        ...
        0 0
        1 1
        2 2
        3 3
        4 4
        5 5
        6 6
        7 7
        8 8
        >>>
        """
        for i in self._id_order:
            yield i, dict(list(zip(self.neighbors[i], self.weights[i])))

    def remap_ids(self, new_ids):
        '''
        In place modification throughout `W` of id values from `w.id_order` to
        `new_ids` in all

        ...

        Parameters
        ----------

        new_ids     :   list
                        /ndarray
                        Aligned list of new ids to be inserted. Note that first
                        element of new_ids will replace first element of
                        w.id_order, second element of new_ids replaces second
                        element of w.id_order and so on.

        Examples
        --------

        >>> from libpysal.weights import lat2W
        >>> w = lat2W(3, 3)
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

        >>> from libpysal.weights import lat2W
        >>> w=lat2W(3,3)
        >>> for i,wi in enumerate(w):
        ...     print(i, wi[0])
        ...
        0 0
        1 1
        2 2
        3 3
        4 4
        5 5
        6 6
        7 7
        8 8
        >>> w.id_order
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> w.id_order=range(8,-1,-1)
        >>> list(w.id_order)
        [8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> for i,w_i in enumerate(w):
        ...     print(i,w_i[0])
        ...
        0 8
        1 7
        2 6
        3 5
        4 4
        5 3
        6 2
        7 1
        8 0
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
        >>> from libpysal.weights import lat2W
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
        >>> from libpysal.weights import W
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
            for j, neigh_list in list(self.neighbors.items()):
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
        >>> from libpysal.weights import lat2W
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

           :widths: auto

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
        >>> from libpysal.weights import lat2W
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
                            print(('WARNING: ', i, ' is an island (no neighbors)'))
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
                raise Exception('unsupported weights transformation')

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

        >>> from libpysal.weights import lat2W
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
            ijs = list(zip(ids[0], ids[1]))
            ijs.sort()
            return ijs

    def symmetrize(self, inplace=False):
        """
        Construct a symmetric KNN weight.

        This ensures that the neighbors of each focal observation
        consider the focal observation itself as a neighbor. 

        This returns a generic W object, since the object is no
        longer guaranteed to have k neighbors for each observation.
        """
        if not inplace:
            neighbors = copy.deepcopy(self.neighbors)
            weights = copy.deepcopy(self.weights)
            out_W = W(neighbors, weights)
            out_W.symmetrize(inplace=True)
            return out_W
        else:
            for focal, fneighbs in list(self.neighbors.items()):
                for j, neighbor in enumerate(fneighbs):
                    neighb_neighbors = self.neighbors[neighbor]
                    if focal not in neighb_neighbors:
                        self.neighbors[neighbor].append(focal)
                        self.weights[neighbor].append(self.weights[focal][j])
            self._cache = dict()
            return

    def full(self):
        """
        Generate a full numpy array.

        Parameters
        ----------
        self        : W
                   spatial weights object

        Returns
        -------
        (fullw, keys) : tuple
                        first element being the full numpy array and second element
                        keys being the ids associated with each row in the array.

        Examples
        --------
        >>> from libpysal.weights import W, full
        >>> neighbors = {'first':['second'],'second':['first','third'],'third':['second']}
        >>> weights = {'first':[1],'second':[1,1],'third':[1]}
        >>> w = W(neighbors, weights)
        >>> wf, ids = full(w)
        >>> wf
        array([[0., 1., 0.],
               [1., 0., 1.],
               [0., 1., 0.]])

        >>> ids
        ['first', 'second', 'third']
        """
        wfull = np.zeros([self.n, self.n], dtype=float)
        keys = list(self.neighbors.keys())
        if self.id_order:
            keys = self.id_order
        for i, key in enumerate(keys):
            n_i = self.neighbors[key]
            w_i = self.weights[key]
            for j, wij in zip(n_i, w_i):
                c = keys.index(j)
                wfull[i, c] = wij
        return (wfull, keys)


    def to_WSP(self):
        '''
        Generate a WSP object.

        Returns
        -------

        implicit : libpysal.weights.WSP
                   Thin W class

        Examples
        --------
        >>> from libpysal.weights import W, WSP
        >>> neighbors={'first':['second'],'second':['first','third'],'third':['second']}
        >>> weights={'first':[1],'second':[1,1],'third':[1]}
        >>> w=W(neighbors,weights)
        >>> wsp=w.to_WSP()
        >>> isinstance(wsp, WSP)
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

    def plot(self, gdf, indexed_on=None, ax=None, color='k',
             node_kws=None, edge_kws=None):
        """
        Plot spatial weights objects.
        NOTE: Requires matplotlib, and implicitly requires geopandas 
        dataframe as input.

        Parameters
        ---------
        gdf         : geopandas geodataframe 
                      the original shapes whose topological relations are 
                      modelled in W. 
        indexed_on  : str 
                      column of gdf which the weights object uses as an index.
                      (Default: None, so the geodataframe's index is used)
        ax          : matplotlib axis
                      axis on which to plot the weights. 
                      (Default: None, so plots on the current figure)
        color       : string
                      matplotlib color string, will color both nodes and edges
                      the same by default. 
        node_kws    : keyword argument dictionary
                      dictionary of keyword arguments to send to pyplot.scatter,
                      which provide fine-grained control over the aesthetics
                      of the nodes in the plot
        edge_kws    : keyword argument dictionary
                      dictionary of keyword arguments to send to pyplot.plot,
                      which provide fine-grained control over the aesthetics
                      of the edges in the plot

        Returns
        -------
        f,ax : matplotlib figure,axis on which the plot is made. 

        NOTE: if you'd like to overlay the actual shapes from the 
              geodataframe, call gdf.plot(ax=ax) after this. To plot underneath,
              adjust the z-order of the geopandas plot: gdf.plot(ax=ax,zorder=0)

        Examples
        --------

        >>> from libpysal.weights.contiguity import Queen
        >>> import libpysal as lp
        >>> import geopandas
        >>> gdf = geopandas.read_file(lp.examples.get_path("columbus.shp"))
        >>> weights = Queen.from_dataframe(gdf)
        >>> tmp = weights.plot(gdf, color='firebrickred', node_kws=dict(marker='*', color='k'))
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("W.plot depends on matplotlib.pyplot, and this was"
                              "not able to be imported. \nInstall matplotlib to"
                              "plot spatial weights.")
        if ax is None:
            f = plt.figure()
            ax = plt.gca()
        else:
            f = plt.gcf()
        if node_kws is not None:
            if 'color' not in node_kws:
                node_kws['color'] = color
        else:
            node_kws=dict(color=color)
        if edge_kws is not None:
            if 'color' not in edge_kws:
                edge_kws['color'] = color
        else:
            edge_kws=dict(color=color) 

        for idx, neighbors in self:
            if idx in self.islands:
                continue
            if indexed_on is not None:
                neighbors = gdf[gdf[indexed_on].isin(neighbors)].index.tolist()
                idx = gdf[gdf[indexed_on] == idx].index.tolist()[0]
            centroids = gdf.loc[neighbors].centroid.apply(lambda p: (p.x, p.y))
            centroids = np.vstack(centroids.values)
            focal = np.hstack(gdf.loc[idx].geometry.centroid.xy)
            seen = set()
            for nidx, neighbor in zip(neighbors, centroids):
                if (idx,nidx) in seen:
                    continue
                ax.plot(*list(zip(focal, neighbor)), marker=None,
                        **edge_kws)
                seen.update((idx,nidx))
                seen.update((nidx,idx))
        ax.scatter(gdf.centroid.apply(lambda p: p.x),
                   gdf.centroid.apply(lambda p: p.y),
                   **node_kws)
        return f,ax


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
    >>> from libpysal.weights import WSP
    >>> rows = [0, 1, 1, 2, 2, 3]
    >>> cols = [1, 0, 2, 1, 3, 3]
    >>> weights =  [1, 0.75, 0.25, 0.9, 0.1, 1]
    >>> sparse = scipy.sparse.csr_matrix((weights, (rows, cols)), shape=(4,4))
    >>> w = WSP(sparse)
    >>> w.s0
    4.0
    >>> w.trcWtW_WW
    6.395
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

    @classmethod
    def from_W(cls, W):
        """
        Constructs a WSP object from the W's sparse matrix

        Parameters
        ----------
        W       :   libpysal.weights.W
                    a pysal weights object with a sparse form and ids

        Returns
        -------
        a WSP instance
        """
        return cls(W.sparse, id_order=W.id_order)

    def to_W(self, silence_warnings=False):

        """
        Convert a pysal WSP object (thin weights matrix) to a pysal W object.

        Parameters
        ----------
        self                     : WSP
                                  PySAL sparse weights object
        silence_warnings         : boolean
                                  Switch to turn off (default on) print statements
                                  for every observation with islands

        Returns
        -------
        w       : W
                  PySAL weights object

        Examples
        --------
        >>> from libpysal.weights import lat2SW, WSP, WSP2W

        Build a 10x10 scipy.sparse matrix for a rectangular 2x5 region of cells
        (rook contiguity), then construct a libpysal sparse weights object (self).

        >>> sp = lat2SW(2, 5)
        >>> self = WSP(sp)
        >>> self.n
        10
        >>> print(self.sparse[0].todense())
        [[0 1 0 0 0 1 0 0 0 0]]

        Convert this sparse weights object to a standard PySAL weights object.

        >>> w = WSP2W(self)
        >>> w.n
        10
        >>> print(w.full()[0][0])
        [0. 1. 0. 0. 0. 1. 0. 0. 0. 0.]

        """
        self.sparse
        indices = self.sparse.indices
        data = self.sparse.data
        indptr = self.sparse.indptr
        id_order = self.id_order
        if id_order:
            # replace indices with user IDs
            indices = [id_order[i] for i in indices]
        else:
            id_order = list(range(self.n))
        neighbors, weights = {}, {}
        start = indptr[0]
        for i in range(self.n):
            oid = id_order[i]
            end = indptr[i + 1]
            neighbors[oid] = indices[start:end]
            weights[oid] = data[start:end]
            start = end
        ids = copy.copy(self.id_order)
        w = W(neighbors, weights, ids,
                    silence_warnings=silence_warnings)
        w._sparse = copy.deepcopy(self.sparse)
        w._cache['sparse'] = w._sparse
        return w
