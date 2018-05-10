import collections
import functools
import itertools
import random
import types

import scipy.sparse.csgraph as csg
from sklearn.metrics.pairwise import distance_metrics
from scipy.sparse.dok import dok_matrix
import numpy as np
import networkx as nx
import libpysal.api as ps_api
import pulp

Move = collections.namedtuple("move", "area old_region new_region")
"A named tuple representing a move from `old_region` to `new_region`."  # sphinx


def array_from_dict_values(dct, sorted_keys=None, flat_output=False,
                           dtype=np.float):
    """
    Return values of the dictionary passed as `dct` argument as an numpy array.
    The values in the returned array are sorted by the keys of `dct`.

    Parameters
    ----------
    dct : dict

    sorted_keys : iterable, optional
        If passed, then the elements of the returned array will be sorted by
        this argument. Thus, this argument can be passed to suppress the
        sorting, or for getting a subset of the dictionary's values or to get
        repeated values.
    flat_output : bool, default: False
        If True, the returned array will be one-dimensional.
        If False, the returned array will be two-dimensional with one row per
        key in `dct`.
    dtype : default: np.float64
        The `dtype` of the returned array.

    Returns
    -------
    array : :class:`numpy.ndarray`

    Examples
    --------
    >>> dict_flat = {0: 0, 1: 10}
    >>> dict_it = {0: [0], 1: [10]}
    >>> desired_flat = np.array([0, 10])
    >>> desired_2d = np.array([[0],
    ...                        [10]])
    >>> flat_flat = array_from_dict_values(dict_flat, flat_output=True)
    >>> (flat_flat == desired_flat).all()
    True
    >>> flat_2d = array_from_dict_values(dict_flat)
    >>> (flat_2d == desired_2d).all()
    True
    >>> it_flat = array_from_dict_values(dict_it, flat_output=True)
    >>> (it_flat == desired_flat).all()
    True
    >>> it_2d = array_from_dict_values(dict_it)
    >>> (it_2d == desired_2d).all()
    True
    """
    if sorted_keys is None:
        sorted_keys = sorted(dct)
    iterable_values = isinstance(dct[sorted_keys[0]], collections.Iterable)
    if iterable_values:
        it = itertools.chain.from_iterable(dct[key] for key in sorted_keys)
    else:
        it = (dct[key] for key in sorted_keys)

    flat_arr = np.fromiter(it, dtype=dtype)
    if flat_output:
        return flat_arr
    return flat_arr.reshape((len(dct), -1))


def scipy_sparse_matrix_from_dict(neighbors):
    """
    Parameters
    ----------
    neighbors : dict
        Each key represents an area. The corresponding value contains the
        area's neighbors.

    Returns
    -------
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix representing the areas' contiguity relation.

    Examples
    --------
    >>> neighbors = {0: {1, 3}, 1: {0, 2, 4}, 2: {1, 5},
    ...              3: {0, 4}, 4: {1, 3, 5}, 5: {2, 4}}
    >>> obtained = scipy_sparse_matrix_from_dict(neighbors)
    >>> desired = np.array([[0, 1, 0, 1, 0, 0],
    ...                     [1, 0, 1, 0, 1, 0],
    ...                     [0, 1, 0, 0, 0, 1],
    ...                     [1, 0, 0, 0, 1, 0],
    ...                     [0, 1, 0, 1, 0, 1],
    ...                     [0, 0, 1, 0, 1, 0]])
    >>> (obtained.todense() == desired).all()
    True
    >>> neighbors = {"left": {"middle"},
    ...              "middle": {"left", "right"},
    ...              "right": {"middle"}}
    >>> obtained = scipy_sparse_matrix_from_dict(neighbors)
    >>> desired = np.array([[0, 1, 0],
    ...                     [1, 0, 1],
    ...                     [0, 1, 0]])
    >>> (obtained.todense() == desired).all()
    True
    """
    n_areas = len(neighbors)
    name_to_int = {area_name: i
                   for i, area_name in enumerate(sorted(neighbors))}
    adj = dok_matrix((n_areas, n_areas))
    for i in neighbors:
        for j in neighbors[i]:
            adj[name_to_int[i], name_to_int[j]] = 1
    return adj.tocsr()


def scipy_sparse_matrix_from_w(w):
    """

    Parameters
    ----------
    w : :class:`libpysal.weights.weights.W`
        A W object representing the areas' contiguity relation.

    Returns
    -------
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix representing the areas' contiguity relation.

    Examples
    --------
    >>> import libpysal as ps
    >>> neighbor_dict = {0: {1}, 1: {0, 2}, 2: {1}}
    >>> w = ps.weights.W(neighbor_dict)
    >>> obtained = scipy_sparse_matrix_from_w(w)
    >>> desired = np.array([[0, 1, 0],
    ...                     [1, 0, 1],
    ...                     [0, 1, 0]])
    >>> (obtained.todense() == desired).all()
    True
    """
    return w.sparse


def dict_from_graph_attr(graph, attr, array_values=False):
    """
    Parameters
    ----------
    graph : networkx.Graph

    attr : str, iterable, or dict
        If str, then it specifies the an attribute of the graph's nodes.
        If iterable of strings, then multiple attributes of the graph's nodes
        are specified.
        If dict, then each key is a node and each value the corresponding
        attribute value. (This format is also this function's return format.)
    array_values : bool, default: False
        If True, then each value is transformed into a :class:`numpy.ndarray`.

    Returns
    -------
    result_dict : dict
        Each key is a node in the graph.
        If `array_values` is False, then each value is a list of attribute
        values corresponding to the key node.
        If `array_values` is True, then each value this list of attribute
        values is turned into a :class:`numpy.ndarray`. That requires the
        values to be shape-compatible for stacking.

    Examples
    --------
    >>> import networkx as nx
    >>> edges = [(0, 1), (1, 2),          # 0 | 1 | 2
    ...          (0, 3), (1, 4), (2, 5),  # ---------
    ...          (3, 4), (4,5)]           # 3 | 4 | 5
    >>> graph = nx.Graph(edges)
    >>> data_dict = {node: 10*node for node in graph}
    >>> nx.set_node_attributes(graph, "test_data", data_dict)
    >>> desired = {key: [value] for key, value in data_dict.items()}
    >>> dict_from_graph_attr(graph, "test_data") == desired
    True
    >>> dict_from_graph_attr(graph, ["test_data"]) == desired
    True
    """
    if isinstance(attr, dict):
        return attr
    if isinstance(attr, str):
        attr = [attr]
    data_dict = {node: [] for node in graph.nodes()}
    for a in attr:
        for node, value in nx.get_node_attributes(graph, a).items():
            data_dict[node].append(value)
    if array_values:
        for node in data_dict:
            data_dict[node] = np.array(data_dict[node])
    return data_dict


def array_from_graph(graph, attr):
    """

    Parameters
    ----------
    graph : networkx.Graph

    attr : str or iterable
        If str, then it specifies the an attribute of the graph's nodes.
        If iterable of strings, then multiple attributes of the graph's nodes
        are specified.

    Returns
    -------
    array : :class:`numpy.ndarray`
        Array with one row for each node in `graph`.

    Examples
    --------
    >>> import networkx as nx
    >>> edges = [(0, 1), (1, 2),          # 0 | 1 | 2
    ...          (0, 3), (1, 4), (2, 5),  # ---------
    ...          (3, 4), (4,5)]           # 3 | 4 | 5
    >>> graph = nx.Graph(edges)
    >>> data_dict = {node: 10*node for node in graph}
    >>> nx.set_node_attributes(graph, "test_data", data_dict)
    >>> desired = np.array([[0],
    ...                     [10],
    ...                     [20],
    ...                     [30],
    ...                     [40],
    ...                     [50]])
    >>> (array_from_graph(graph, "test_data") == desired).all()
    True
    >>> (array_from_graph(graph, ["test_data"]) == desired).all()
    True
    >>> (array_from_graph(graph, ["test_data", "test_data"]) ==
    ...  np.hstack((desired, desired))).all()
    True
    """
    dct = dict_from_graph_attr(graph, attr)
    return array_from_dict_values(dct)


def array_from_graph_or_dict(graph, attr):
        if isinstance(attr, (str, collections.Iterable)):
            return array_from_graph(graph, attr)
        elif isinstance(attr, collections.Mapping):
            return array_from_dict_values(attr)
        else:
            raise ValueError("The `attr` argument must be a string, a list of "
                             "strings or a dictionary.")


def array_from_region_list(region_list):
    """
    Parameters
    ----------
    region_list : `list`
        Each list element is an iterable of a region's areas.

    Returns
    -------
    labels : :class:`numpy.ndarray`
        Each element specifies the region of the corresponding area.

    Examples
    --------
    >>> import numpy as np
    >>> obtained = array_from_region_list([{0, 1, 2, 5}, {3, 4}])
    >>> desired = np.array([ 0, 0, 0, 1, 1, 0])
    >>> (obtained == desired).all()
    True
    """
    n_areas = sum(len(region) for region in region_list)
    labels = np.zeros((n_areas))
    for region_idx, region in enumerate(region_list):
        for area in region:
            labels[area] = region_idx
    return labels


def array_from_df_col(df, attr):
    """
    Extract one or more columns from a DataFrame as numpy array.

    Parameters
    ----------
    df : Union[DataFrame, GeoDataFrame]

    attr : Union[str, Sequence[str]]
        The columns' names to extract.

    Returns
    -------
    col : :class:`numpy.ndarray`
        The specified column(s) of the array.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"col1": [1, 2, 3],
    ...                    "col2": [7, 8, 9]})
    >>> (array_from_df_col(df, "col1") == np.array([[1],
    ...                                         [2],
    ...                                         [3]])).all()
    True
    >>> (array_from_df_col(df, ["col1"]) == np.array([[1],
    ...                                           [2],
    ...                                           [3]])).all()
    True
    >>> (array_from_df_col(df, ["col1", "col2"]) == np.array([[1, 7],
    ...                                                   [2, 8],
    ...                                                   [3, 9]])).all()
    True
    """
    value_error = ValueError("The attr argument has to be of one of the "
                             "following types: str or a sequence of strings.")
    if isinstance(attr, str):
        attr = [attr]
    elif isinstance(attr, collections.Sequence):
        if not all(isinstance(el, str) for el in attr):
            raise value_error
    else:
        raise value_error
    return np.array(df[attr])


def w_from_gdf(gdf, contiguity):
    """
    Get a `W` object from a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame

    contiguity : {"rook", "queen"}

    Returns
    -------
    weights : `W`
        The contiguity information contained in the `gdf` argument in the form
        of a W object.
    """
    if not isinstance(contiguity, str) or \
            contiguity.lower() not in ["rook", "queen"]:
        raise ValueError("The contiguity argument must be either None "
                         "or one of the following strings: "
                         '"rook" or"queen".')
    if contiguity.lower() == "rook":
        weights = ps_api.Rook.from_dataframe(gdf)
    else:  # contiguity.lower() == "queen"
        weights = ps_api.Queen.from_dataframe(gdf)
    return weights


def dataframe_to_dict(df, cols):
    """
    Parameters
    ----------
    df : Union[:class:`pandas.DataFrame`, :class:`geopandas.GeoDataFrame`]

    cols : Union[`str`,  `list`]
        If `str`, then it is the name of a column of `df`.
        If `list`, then it is a list of strings. Each string is the name of a
        column of `df`.

    Returns
    -------
    result : dict
        The keys are the elements of the DataFrame's index.
        Each value is a :class:`numpy.ndarray` holding the corresponding values
        in the columns specified by `cols`.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"data": [100, 120, 115]})
    >>> result = dataframe_to_dict(df, "data")
    >>> result == {0: 100, 1: 120, 2: 115}
    True
    >>> import numpy as np
    >>> df = pd.DataFrame({"data": [100, 120],
    ...                    "other": [1, 2]})
    >>> actual = dataframe_to_dict(df, ["data", "other"])
    >>> desired = {0: np.array([100, 1]), 1: np.array([120, 2])}
    >>> all(np.array_equal(actual[i], desired[i]) for i in desired)
    True
    """
    return dict(zip(df.index, np.array(df[cols])))


def find_sublist_containing(el, lst, index=False):
    """

    Parameters
    ----------
    el :
        The element to search for in the sublists of `lst`.
    lst : collections.Sequence
        A sequence of sequences or sets.
    index : bool, default: False
        If False (default), the subsequence or subset containing `el` is
        returned.
        If True, the index of the subsequence or subset in `lst` is returned.

    Returns
    -------
    result : collections.Sequence, collections.Set or int
        See the `index` argument for more information.

    Raises
    ------
    exc : LookupError
        If `el` is not in any of the elements of `lst`.

    Examples
    --------
    >>> lst = [{0, 1}, {2}]
    >>> find_sublist_containing(0, lst, index=False) == {0, 1}
    True
    >>> find_sublist_containing(0, lst, index=True) == 0
    True
    >>> find_sublist_containing(2, lst, index=False) == {2}
    True
    >>> find_sublist_containing(2, lst, index=True) == 1
    True
    """
    for idx, sublst in enumerate(lst):
        if el in sublst:
            return idx if index else sublst
    raise LookupError(
            "{} not found in any of the sublists of {}".format(el, lst))


def get_metric_function(metric=None):
    """
    Parameters
    ----------
    metric : str or function or None, default: None
        Using None is equivalent to using "euclidean".

        If str, then this string specifies the distance metric (from
        scikit-learn) to use for calculating the objective function.
        Possible values are:

        * "cityblock" for sklearn.metrics.pairwise.manhattan_distances
        * "cosine" for sklearn.metrics.pairwise.cosine_distances
        * "euclidean" for sklearn.metrics.pairwise.euclidean_distances
        * "l1" for sklearn.metrics.pairwise.manhattan_distances
        * "l2" for sklearn.metrics.pairwise.euclidean_distances
        * "manhattan" for sklearn.metrics.pairwise.manhattan_distances

        If function, then this function should take two arguments and return a
        scalar value. Furthermore, the following conditions must be fulfilled:

        1. d(a, b) >= 0, for all a and b
        2. d(a, b) == 0, if and only if a = b, positive definiteness
        3. d(a, b) == d(b, a), symmetry
        4. d(a, c) <= d(a, b) + d(b, c), the triangle inequality

    Returns
    -------
    metric_func : function
        If the `metric` argument is a function, it is returned.
        If the `metric` argument is a string, then the corresponding distance
        metric function from `sklearn.metrics.pairwise` is returned.
    """
    if metric is None:
        metric = "manhattan"

    if isinstance(metric, str):
        try:
            return distance_metrics()[metric]
        except KeyError:
            raise ValueError(
                "{} is not a known metric. Please use rather one of the "
                "following metrics: {}".format(metric,
                                               tuple(name for name in
                                                     distance_metrics().keys()
                                                     if name != "precomputed"))
            )
    elif callable(metric):
        return metric
    else:
        raise ValueError("A {} was passed as `metric` argument. "
                         "Please pass a string or a function "
                         "instead.".format(type(metric)))


class MissingMetric(RuntimeError):
    """Raised when a distance metric is required but was not set."""


def raise_distance_metric_not_set(x, y):
    raise MissingMetric("distance metric not set!")


def make_move(moving_area, new_label, labels):
    """
    Modify the `labels` argument in place (no return value!) such that the
    area `moving_area` has the new region label `new_label`.

    Parameters
    ----------
    moving_area :
        The area to be moved (assigned to a new region).
    new_label : `int`
        The new region label of area `moving_area`.
    labels : :class:`numpy.ndarray`
        Each element is a region label of the area corresponding array index.

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array([0, 0, 0, 0, 1, 1])
    >>> make_move(3, 1, labels)
    >>> (labels == np.array([0, 0, 0, 1, 1, 1])).all()
    True
    """
    labels[moving_area] = new_label


def distribute_regions_among_components(component_labels, n_regions):
    r"""
    Parameters
    ----------
    component_labels : list
        Each element specifies to which connected component an area belongs.
        An example would be [0, 0, 1, 0, 0, 1] for the following two islands:
        
        ::
        
          island one        island two
          .-------.         .---.
          | 0 | 1 |         | 2 |
          | - - - |         | - |
          | 3 | 4 |         | 5 |
          `-------´         `---´

    n_regions : int

    Returns
    -------
    result_dict : Dict[int, int]
        Each key is a label of a connected component. Each value specifies into
        how many regions the component is to be clustered.
    """
    # copy list to avoid manipulating callers list instance
    component_labels = list(component_labels)
    n_regions_to_distribute = n_regions
    components = set(component_labels)
    if len(components) == 1:
        return {0: n_regions}
    result_dict = {}
    # make sure each connected component has at least one region assigned to it
    for comp in components:
        component_labels.remove(comp)
        result_dict[comp] = 1
        n_regions_to_distribute -= 1
    # distribute the rest of the regions to random components with bigger
    # components being likely to get more regions assigned to them
    while n_regions_to_distribute > 0:
        position = random.randrange(len(component_labels))
        picked_comp = component_labels.pop(position)
        result_dict[picked_comp] += 1
        n_regions_to_distribute -= 1
    return result_dict


def generate_initial_sol(adj, n_regions):
    """
    Generate a random initial clustering.

    Parameters
    ----------
    adj : :class:`scipy.sparse.csr_matrix`

    n_regions : int

    Yields
    ------
    region_labels : :class:`numpy.ndarray`
        An array with -1 for areas which are not part of the yielded
        component and an integer >= 0 specifying the region of areas within the
        yielded component.
    """
    # check args
    n_areas = adj.shape[0]
    if n_areas == 0:
        raise ValueError("There must be at least one area.")
    if n_areas < n_regions:
        raise ValueError("The number of regions ({}) must be "
                         "less than or equal to the number of areas "
                         "({}).".format(n_regions, n_areas))
    if n_regions == 1:
        yield {area: 0 for area in range(n_areas)}
        return

    n_comps, comp_labels = csg.connected_components(adj)
    if n_comps > n_regions:
            raise ValueError("The number of regions ({}) must not be "
                             "less than the number of connected components "
                             "({}).".format(n_regions, n_comps))
    n_regions_per_comp = distribute_regions_among_components(comp_labels,
                                                             n_regions)

    print("n_regions_per_comp", n_regions_per_comp)
    regions_built = 0
    for comp_label, n_regions_in_comp in n_regions_per_comp.items():
        print("comp_label", comp_label)
        print("n_regions_in_comp", n_regions_in_comp)
        region_labels = -np.ones(len(comp_labels), dtype=np.int32)
        in_comp = comp_labels == comp_label
        comp_adj = adj[in_comp]
        comp_adj = comp_adj[:, in_comp]
        region_labels_comp = _randomly_divide_connected_graph(
                comp_adj, n_regions_in_comp) + regions_built
        regions_built += n_regions_in_comp
        print("Regions in comp:", set(region_labels_comp))
        region_labels[in_comp] = region_labels_comp
        yield region_labels


def _randomly_divide_connected_graph(adj, n_regions):
    """
    Divide the provided connected graph into `n_regions` regions.

    Parameters
    ----------
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix.
    n_regions : int
        The desired number of clusters. Must be > 0 and <= number of nodes.

    Returns
    -------
    labels : :class:`numpy.ndarray`
        Each element (an integer in {0, ..., `n_regions` - 1}) specifies the
        region an area (defined by the index in the array) belongs to.

    Examples
    --------
    >>> from scipy.sparse import diags
    >>> n_nodes = 10
    >>> adj_diagonal = [1] * (n_nodes-1)
    >>> # 10x10 adjacency matrix representing the path 0-1-2-...-9-10
    >>> adj = diags([adj_diagonal, adj_diagonal], offsets=[-1, 1])
    >>> n_regions_desired = 4
    >>> labels = _randomly_divide_connected_graph(adj, n_regions_desired)
    >>> n_regions_obtained = len(set(labels))
    >>> n_regions_desired == n_regions_obtained
    True
    """
    if not n_regions > 0:
        msg = "n_regions is {} but must be positive.".format(n_regions)
        raise ValueError(msg)
    n_areas = adj.shape[0]
    if not n_regions <= n_areas:
        msg = "n_regions is {} but must less than or equal to " + \
              "the number of nodes which is {}".format(n_regions, n_areas)
        raise ValueError(msg)
    mst = csg.minimum_spanning_tree(adj)
    for _ in range(n_regions - 1):
        # try different links to cut and pick the one leading to the most
        # balanced solution
        best_link = None
        max_region_size = float("inf")
        for __ in range(5):
            mst_copy = mst.copy()
            nonzero_i, nonzero_j = mst_copy.nonzero()
            random_position = random.randrange(len(nonzero_i))
            i, j = nonzero_i[random_position], nonzero_j[random_position]
            mst_copy[i, j] = 0
            mst_copy.eliminate_zeros()
            labels = csg.connected_components(mst_copy, directed=False)[1]
            max_size = max(np.unique(labels, return_counts=True)[1])
            if max_size < max_region_size:
                best_link = (i, j)
                max_region_size = max_size
        mst[best_link[0], best_link[1]] = 0
        mst.eliminate_zeros()
    return csg.connected_components(mst)[1]


def copy_func(f):
    """
    Return a copy of a function. This is useful e.g. to create aliases (whose
    docstrings can be changed without affecting the original function).
    The implementation is taken from https://stackoverflow.com/a/13503277.

    Parameters
    ----------
    f : function

    Returns
    -------
    g : function
        Copy of `f`.
    """
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def assert_feasible(solution, adj, n_regions=None):
    """
    Parameters
    ----------
    solution : :class:`numpy.ndarray`
        Array of region labels.
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix representing the contiguity relation.
    n_regions : `int` or `None`
        An `int` represents the desired number of regions.
        If `None`, then the number of regions is not checked.

    Raises
    ------
    exc : `ValueError`
        A `ValueError` is raised if clustering is not spatially contiguous.
        Given the `n_regions` argument is not `None`, a `ValueError` is raised
        also if the number of regions is not equal to the `n_regions` argument.
    """
    if n_regions is not None:
        if len(set(solution)) != n_regions:
            raise ValueError("The number of regions is {} but "
                             "should be {}".format(len(solution), n_regions))
    for region_label in set(solution):
        _, comp_labels = csg.connected_components(adj)
        # check whether equal region_label implies equal comp_label
        comp_labels_in_region = comp_labels[solution == region_label]
        if not all_elements_equal(comp_labels_in_region):
            raise ValueError("Region {} is not spatially "
                             "contiguous.".format(region_label))


def all_elements_equal(array):
    return np.max(array) == np.min(array)


def separate_components(adj, labels):
    """
    Take a labels array and yield modifications of it (one modified array per
    connected component). The modified array will be unchanged at those indices
    belonging to the current connected component. Thus it will have integers
    >= 0 there. At all other indices the Yielded array will be -1.

    Parameters
    ----------
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix representing the contiguity relation.
    labels : :class:`numpy.ndarray`

    Yields
    ------
    comp_dict : :class:`numpy.ndarray`
        Each yielded dict represents one connected component of the graph
        specified by the `adj` argument. In a yielded dict, each key is an area
        and each value is the corresponding region-ID.

    Examples
    --------
    >>> edges_island1 = [(0, 1), (1, 2),          # 0 | 1 | 2
    ...                  (0, 3), (1, 4), (2, 5),  # ---------
    ...                  (3, 4), (4,5)]           # 3 | 4 | 5
    >>>
    >>> edges_island2 = [(6, 7),                  # 6 | 7
    ...                  (6, 8), (7, 9),          # -----
    ...                  (8, 9)]                  # 8 | 9
    >>>
    >>> graph = nx.Graph(edges_island1 + edges_island2)
    >>> adj = nx.to_scipy_sparse_matrix(graph)
    >>>
    >>> # island 1: island divided into regions 0, 1, and 2
    >>> sol_island1 = [area%3 for area in range(6)]
    >>> # island 2: all areas are in region 3
    >>> sol_island2 = [3 for area in range(6, 10)]
    >>> labels = np.array(sol_island1 + sol_island2)
    >>>
    >>> yielded = list(separate_components(adj, labels))
    >>> yielded.sort(key=lambda arr: arr[0], reverse=True)
    >>> (yielded[0] == np.array([0, 1, 2, 0, 1, 2, -1, -1, -1, -1])).all()
    True
    >>> (yielded[1] == np.array([-1, -1, -1, -1, -1, -1, 3, 3, 3, 3])).all()
    True
    """
    n_comps, comp_labels = csg.connected_components(adj)
    for comp in set(comp_labels):
        region_labels = -np.ones(len(comp_labels), dtype=np.int32)
        in_comp = comp_labels == comp
        region_labels[in_comp] = labels[in_comp]
        yield region_labels


def random_element_from(lst):
    random_position = random.randrange(len(lst))
    return lst[random_position]


def pop_randomly_from(lst):
    random_position = random.randrange(len(lst))
    return lst.pop(random_position)


def count(arr, el):
    """
    Parameters
    ----------
    arr : :class:`numpy.ndarray`

    el : object

    Returns
    -------
    result : :class:`numpy.ndarray`
        The number of occurences of `el` in `arr`.

    Examples
    --------
    >>> arr = np.array([0, 0, 0, 1, 1])
    >>> count(arr, 0)
    3
    >>> count(arr, 1)
    2
    >>> count(arr, 2)
    0
    """
    unique, counts = np.unique(arr, return_counts=True)
    idx = np.where(unique == el)[0]
    if len(idx) > 0:
        return int(counts[idx])
    return 0


def check_solver(solver):
    if not isinstance(solver, str) \
            or solver.lower() not in ["cbc", "cplex", "glpk", "gurobi"]:
        raise ValueError("The solver argument must be one of the following"
                         ' strings: "cbc", "cplex", "glpk", or "gurobi".')


def get_solver_instance(solver_string):
    solver = {"cbc": pulp.solvers.COIN_CMD,
              "cplex": pulp.solvers.CPLEX,
              "glpk": pulp.solvers.GLPK,
              "gurobi": pulp.solvers.GUROBI}[solver_string.lower()]
    solver_instance = solver()
    return solver_instance
