import collections
import numbers
from math import floor, log10

from geopandas import GeoDataFrame
import numpy as np
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum

from region.csgraph_utils import neighbors
from region.util import array_from_df_col, array_from_dict_values,\
    array_from_graph_or_dict, check_solver, copy_func, get_metric_function,\
    get_solver_instance, raise_distance_metric_not_set,\
    scipy_sparse_matrix_from_dict, scipy_sparse_matrix_from_w, w_from_gdf


class MaxPRegionsExact:
    """
    A class for solving the max-p-regions problem by transforming it into a
    mixed-integer-programming problem (MIP) as described in [DAR2012]_.

    Attributes
    ----------
    labels_ : dict
        Each key is an area and each value the region it has been assigned to.
    """
    def __init__(self):
        self.labels_ = None
        self.solver = None
        self.metric = raise_distance_metric_not_set

    def fit_from_scipy_sparse_matrix(self, adj, attr, spatially_extensive_attr,
                                     threshold, solver="cbc",
                                     metric="euclidean"):
        """
        Solve the max-p-regions problem as MIP as described in [DAR2012]_.

        The resulting region labels are assigned to the instance's
        :attr:`labels_` attribute.

        Parameters
        ----------
        adj : class:`scipy.sparse.csr_matrix`
            Adjacency matrix representing the areas' contiguity relation.
        attr : :class:`numpy.ndarray`
            Array (number of areas x number of attributes) of areas' attributes
            relevant to clustering.
        spatially_extensive_attr : :class:`numpy.ndarray`
            Array (number of areas x number of attributes) of areas' attributes
            relevant to ensuring the threshold condition.
        threshold : numbers.Real or :class:`numpy.ndarray`
            The lower bound for a region's sum of spatially extensive
            attributes. The argument's type is numbers.Real if there is only
            one spatially extensive attribute per area, otherwise it is a
            one-dimensional array with as many entries as there are spatially
            extensive attributes per area.
        solver : {"cbc", "cplex", "glpk", "gurobi"}, default: "cbc"
            The solver to use. Unless the default solver is used, the user has
            to make sure that the specified solver is installed.

            * "cbc" - the Cbc (Coin-or branch and cut) solver
            * "cplex" - the CPLEX solver
            * "glpk" - the GLPK (GNU Linear Programming Kit) solver
            * "gurobi" - the Gurobi Optimizer

        metric : str or function, default: "euclidean"
            See the `metric` argument in
            :func:`region.util.get_metric_function`.
        """
        self.metric = get_metric_function(metric)
        check_solver(solver)

        prob = LpProblem("Max-p-Regions", LpMinimize)

        # Parameters of the optimization problem
        n_areas = adj.shape[0]
        I = list(range(n_areas))  # index for areas
        II = [(i, j)
              for i in I
              for j in I]
        II_upper_triangle = [(i, j) for i, j in II if i < j]
        # index of potential regions, called k in [DAR2012]_:
        K = range(n_areas)
        # index of contiguity order, called c in [DAR2012]_:
        O = range(n_areas)
        d = {(i, j): self.metric(attr[i].reshape(1, -1),
                                 attr[j].reshape(1, -1))
             for i, j in II_upper_triangle}
        h = 1 + floor(log10(sum(d[(i, j)] for i, j in II_upper_triangle)))

        # Decision variables
        t = LpVariable.dicts(
            "t",
            ((i, j) for i, j in II_upper_triangle),
            lowBound=0, upBound=1, cat=LpInteger)
        x = LpVariable.dicts(
            "x",
            ((i, k, o) for i in I for k in K for o in O),
            lowBound=0, upBound=1, cat=LpInteger)

        # Objective function
        # (1) in Duque et al. (2012): "The Max-p-Regions Problem"
        prob += -10**h * lpSum(x[i, k, 0] for k in K for i in I) \
            + lpSum(d[i, j] * t[i, j] for i, j in II_upper_triangle)

        # Constraints
        # (2) in Duque et al. (2012): "The Max-p-Regions Problem"
        for k in K:
            prob += lpSum(x[i, k, 0] for i in I) <= 1
        # (3) in Duque et al. (2012): "The Max-p-Regions Problem"
        for i in I:
            prob += lpSum(x[i, k, o] for k in K for o in O) == 1
        # (4) in Duque et al. (2012): "The Max-p-Regions Problem"
        for i in I:
            for k in K:
                for o in range(1, len(O)):
                    prob += x[i, k, o] <= lpSum(x[j, k, o-1]
                                                for j in neighbors(adj, i))
        # (5) in Duque et al. (2012): "The Max-p-Regions Problem"
        if isinstance(spatially_extensive_attr[I[0]], numbers.Real):
            for k in K:
                lhs = lpSum(x[i, k, o] * spatially_extensive_attr[i]
                            for i in I for o in O)
                prob += lhs >= threshold * lpSum(x[i, k, 0] for i in I)
        elif isinstance(spatially_extensive_attr[I[0]], collections.Iterable):
            for el in range(len(spatially_extensive_attr[I[0]])):
                for k in K:
                    lhs = lpSum(x[i, k, o] * spatially_extensive_attr[i][el]
                                for i in I for o in O)
                    if isinstance(threshold, numbers.Real):
                        rhs = threshold * lpSum(x[i, k, 0] for i in I)
                        prob += lhs >= rhs
                    elif isinstance(threshold, np.ndarray):
                        rhs = threshold[el] * lpSum(x[i, k, 0] for i in I)
                        prob += lhs >= rhs
        # (6) in Duque et al. (2012): "The Max-p-Regions Problem"
        for i, j in II_upper_triangle:
            for k in K:
                prob += t[i, j] >= \
                        lpSum(x[i, k, o] + x[j, k, o] for o in O) - 1
        # (7) in Duque et al. (2012): "The Max-p-Regions Problem"
        # already in LpVariable-definition
        # (8) in Duque et al. (2012): "The Max-p-Regions Problem"
        # already in LpVariable-definition

        # additional constraint for speedup (p. 405 in [DAR2012]_)
        for o in O:
            prob += x[I[0], K[0], o] == (1 if o == 0 else 0)

        # Solve the optimization problem
        solver = get_solver_instance(solver)
        print("start solving with", solver)
        prob.solve(solver)
        print("solved")
        result = np.zeros(n_areas)
        for i in I:
            for k in K:
                for o in O:
                    if x[i, k, o].varValue == 1:
                        result[i] = k
        self.labels_ = result
        self.solver = solver

    fit = copy_func(fit_from_scipy_sparse_matrix)
    fit.__doc__ = "Alias for :meth:`fit_from_scipy_sparse_matrix`.\n\n" \
                  + fit_from_scipy_sparse_matrix.__doc__

    def fit_from_dict(self, neighbors_dict, attr, spatially_extensive_attr,
                      threshold, solver="cbc", metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        neighbors_dict : dict
            Each key represents an area and each value is an iterable of
            neighbors of this area.
        attr : dict
            A dict with the same keys as `neighbors_dict` and values
            representing the attributes for calculating homo-/heterogeneity. A
            value can be scalar (e.g. `float` or `int`) or a
            :class:`numpy.ndarray`.
        spatially_extensive_attr : dict
            A dict with the same keys as `neighbors_dict` and values
            representing the spatially extensive attribute (scalar or iterable
            of scalars). In the max-p-regions problem each region's sum of
            spatially extensive attributes must be greater than a specified
            threshold. In case of iterables of scalars as dict-values all
            elements of the iterable have to fulfill the condition.
        threshold : numbers.Real or :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        solver : {"cbc", "cplex", "glpk", "gurobi"}, default: "cbc"
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        metric : str or function, default: "euclidean"
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        if not isinstance(neighbors_dict, dict):
            raise ValueError("The neighbors_dict argument must be dict.")

        not_same_dict_keys_msg = "The {} argument has to be of type dict " \
                                 "with the same keys as neighbors_dict."

        if not isinstance(attr, dict) or attr.keys() != neighbors_dict.keys():
            raise ValueError(not_same_dict_keys_msg.format("attr"))

        if not isinstance(spatially_extensive_attr, dict) or \
                spatially_extensive_attr.keys() != neighbors_dict.keys():
            raise ValueError(
                    not_same_dict_keys_msg.format(spatially_extensive_attr))

        adj = scipy_sparse_matrix_from_dict(neighbors_dict)
        attr_arr = array_from_dict_values(attr)
        spat_ext_attr_arr = array_from_dict_values(spatially_extensive_attr)
        self.fit_from_scipy_sparse_matrix(adj, attr_arr, spat_ext_attr_arr,
                                          threshold=threshold, solver=solver,
                                          metric=metric)

    def fit_from_geodataframe(self, gdf, attr, spatially_extensive_attr,
                              threshold, solver="cbc", metric="euclidean",
                              contiguity="rook"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        gdf : GeoDataFrame

        attr : str or list
            The clustering criteria (columns of the GeoDataFrame `gdf`) are
            specified as string (for one column) or list of strings (for
            multiple columns).
        spatially_extensive_attr : str or list
            The name (`str`) or names (`list` of strings) of column(s) in `gdf`
            containing the spatially extensive attributes.
        threshold : numbers.Real or :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        solver : {"cbc", "cplex", "glpk", "gurobi"}, default: "cbc"
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        metric : str or function, default: "euclidean"
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        contiguity : {"rook", "queen"}, default: "rook"
            Defines the contiguity relationship between areas. Possible
            contiguity definitions are:

            * "rook" - Rook contiguity.
            * "queen" - Queen contiguity.
        """
        w = w_from_gdf(gdf, contiguity)
        attr = array_from_df_col(gdf, attr)
        spat_ext_attr = array_from_df_col(gdf, spatially_extensive_attr)

        self.fit_from_w(w, attr, spat_ext_attr, threshold=threshold,
                        solver=solver, metric=metric)

    def fit_from_networkx(self, graph, attr, spatially_extensive_attr,
                          threshold, solver="cbc", metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        graph : `networkx.Graph`

        attr : str, list or dict
            If the clustering criteria are present in the networkx.Graph
            `graph` as node attributes, then they can be specified as a string
            (for one criterion) or as a list of strings (for multiple
            criteria).
            Alternatively, a dict can be used with each key being a node of the
            networkx.Graph `graph` and each value being the corresponding
            clustering criterion (a scalar (e.g. `float` or `int`) or a
            :class:`numpy.ndarray`).
            If there are no clustering criteria present in the networkx.Graph
            `graph` as node attributes, then a dictionary must be used for this
            argument. Refer to the corresponding argument in
            :meth:`fit_from_dict` for more details about the expected dict.
        spatially_extensive_attr : str, list or dict
            If the spatially extensive attribute is present in the
            networkx.Graph `graph` as node attributes, then they can be
            specified as a string (for one attribute) or as a list of
            strings (for multiple attributes).
            Alternatively, a dict can be used with each key being a node of the
            networkx.Graph `graph` and each value being the corresponding
            spatially extensive attribute (a scalar (e.g. `float` or `int`) or
            a :class:`numpy.ndarray`).
            If there are no spatially extensive attributes present in the
            networkx.Graph `graph` as node attributes, then a dictionary must
            be used for this argument. Refer to the corresponding argument in
            :meth:`fit_from_dict` for more details about the expected dict.
        threshold : numbers.Real or :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        solver : {"cbc", "cplex", "glpk", "gurobi"}, default: "cbc"
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        metric : str or function, default: "euclidean"
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        attr = array_from_graph_or_dict(graph, attr)
        sp_ext_attr = array_from_graph_or_dict(graph, spatially_extensive_attr)
        self.fit_from_scipy_sparse_matrix(adj, attr, sp_ext_attr,
                                          threshold=threshold, solver=solver,
                                          metric=metric)

    def fit_from_w(self, w, attr, spatially_extensive_attr, threshold,
                   solver="cbc", metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        w : libpysal.weights.W
            W object representing the areas' contiguity relation.
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        spatially_extensive_attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        threshold : numbers.Real or :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        solver : {"cbc", "cplex", "glpk", "gurobi"}, default: "cbc"
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        metric : str or function, default: "euclidean"
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = scipy_sparse_matrix_from_w(w)
        self.fit_from_scipy_sparse_matrix(adj, attr, spatially_extensive_attr,
                                          threshold=threshold, solver=solver,
                                          metric=metric)
