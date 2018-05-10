import collections
import numbers

from geopandas import GeoDataFrame
import numpy as np
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum

from region.csgraph_utils import neighbors
from region.util import array_from_df_col, array_from_dict_values,\
    array_from_graph_or_dict, array_from_region_list, check_solver, copy_func,\
    get_metric_function, get_solver_instance, raise_distance_metric_not_set,\
    scipy_sparse_matrix_from_dict, scipy_sparse_matrix_from_w, w_from_gdf


class PRegionsExact:
    """
    A class for solving the p-regions problem by transforming it into a
    mixed-integer-programming problem (MIP) as described in [DCM2011]_.

    Attributes
    ----------
    labels_ : :class:`numpy.ndarray`
        Each element is a region label specifying to which region the
        corresponding area was assigned to by the last run of a fit-method.
    method : str
        The method used in the last call of a fit-method for translating the
        p-regions problem into an MIP.
    metric : function
        The distance metric specified in the last call of a fit-method.
    n_regions : int
        The number of regions the areas were clustered into by the last run of
        a fit-method.
    solver : str
        The solver used in the last call of a fit-method.
    """
    def __init__(self):
        self.n_regions = None
        self.labels_ = None
        self.method = None
        self.solver = None
        self.metric = raise_distance_metric_not_set

    def fit_from_scipy_sparse_matrix(self, adj, attr, n_regions,
                                     method="flow", solver="cbc",
                                     metric="euclidean"):
        """
        Solve the p-regions problem as MIP as described in [DCM2011]_.

        The resulting region labels are assigned to the instance's
        :attr:`labels_` attribute.

        Parameters
        ----------
        adj : class:`scipy.sparse.csr_matrix`
            Adjacency matrix representing the areas' contiguity relation.
        attr : :class:`numpy.ndarray`
            Array (number of areas x number of attributes) of areas' attributes
            relevant to clustering.
        n_regions : `int`
            Number of desired regions.
        method : {"flow", "order", "tree"}, default: "flow"
            The method to translate the clustering problem into an exact
            optimization model.

            * "flow" - Flow model on p. 112-113 in [DCM2011]_
            * "order" - Order model on p. 110-112 in [DCM2011]_
            * "tree" - Tree model on p. 108-110 in [DCM2011]_

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
        if not isinstance(n_regions, numbers.Integral) or n_regions <= 0:
            raise ValueError("The n_regions argument must be a positive "
                             "integer.")
        if adj.shape[0] < n_regions:
            raise ValueError("The number of regions must be less than the "
                             "number of areas.")
        if attr.ndim == 1:
            attr = attr.reshape(adj.shape[0], -1)
        self._check_method(method)
        check_solver(solver)
        metric = get_metric_function(metric)

        opt_func = {"flow": _flow,
                    "order": _order,
                    "tree": _tree}[method.lower()]

        result_dict = opt_func(adj, attr, n_regions, solver, metric)
        self.labels_ = result_dict
        self.n_regions = n_regions
        self.method = method
        self.metric = metric
        self.solver = solver

    fit = copy_func(fit_from_scipy_sparse_matrix)
    fit.__doc__ = "Alias for :meth:`fit_from_scipy_sparse_matrix`.\n\n" \
                  + fit_from_scipy_sparse_matrix.__doc__

    def fit_from_dict(self, neighbors_dict, attr, n_regions, method="flow",
                      solver="cbc", metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        neighbors_dict : dict
            Each key represents an area and each value is an iterable of
            neighbors of this area.
        attr : dict
            A dict with the same keys as `neighbors_dict` and values
            representing the clustering criteria. A value can be scalar (e.g.
            float or int) or a :class:`numpy.ndarray`.
        n_regions : int
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        method : str
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        solver : str
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        metric : str or function, default: "euclidean"
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        if not isinstance(neighbors_dict, dict):
            raise ValueError("The neighbors_dict argument must be dict.")

        if not isinstance(attr, dict) or attr.keys() != neighbors_dict.keys():
            raise ValueError("The attr argument has to be of type dict with "
                             "the same keys as neighbors_dict.")

        adj = scipy_sparse_matrix_from_dict(neighbors_dict)
        attr_arr = array_from_dict_values(attr)

        self.fit_from_scipy_sparse_matrix(adj, attr_arr, n_regions,
                                          method=method, solver=solver,
                                          metric=metric)

    def fit_from_geodataframe(self, gdf, attr, n_regions, method="flow",
                              solver="cbc", metric="euclidean",
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
        n_regions : int
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        method : str
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        solver : str
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        contiguity : {"rook", "queen"}, default: "rook"
            Defines the contiguity relationship between areas. Possible
            contiguity definitions are:

            * "rook" - Rook contiguity.
            * "queen" - Queen contiguity.

        metric : str or function, default: "euclidean"
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        w = w_from_gdf(gdf, contiguity)
        attr = array_from_df_col(gdf, attr)
        self.fit_from_w(w, attr, n_regions, method=method, solver=solver,
                        metric=metric)

    def fit_from_networkx(self, graph, attr, n_regions, method="flow",
                          solver="cbc", metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        graph : `networkx.Graph`
            Graph representing the areas' contiguity relation.
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
        n_regions : int
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        method : str
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        solver : str
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        metric : str or function, default: "euclidean"
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        attr = array_from_graph_or_dict(graph, attr)
        self.fit_from_scipy_sparse_matrix(adj, attr, n_regions, method=method,
                                          solver=solver, metric=metric)

    def fit_from_w(self, w, attr, n_regions, method="flow", solver="cbc",
                   metric="euclidean"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        w : libpysal.weights.W
            W object representing the areas' contiguity relation.
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        n_regions : int
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        method : str
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        solver : str
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        metric : str or function, default: "euclidean"
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = scipy_sparse_matrix_from_w(w)
        self.fit_from_scipy_sparse_matrix(adj, attr, n_regions, method=method,
                                          solver=solver, metric=metric)

    @staticmethod
    def _check_method(method):
        if not isinstance(method, str) \
                or method.lower() not in ["flow", "order", "tree"]:
            raise ValueError("The method argument must be one of the following"
                             " strings: 'flow', 'order', or 'tree'.")


def _flow(adj, attr, n_regions, solver, metric):
    """
    Parameters
    ----------
    adj : class:`scipy.sparse.csr_matrix`
        See the corresponding argument in
        :meth:`PRegionsExact.fit_from_scipy_sparse_matrix`.
    attr : :class:`numpy.ndarray`
        See the corresponding argument in
        :meth:`PRegionsExact.fit_from_scipy_sparse_matrix`.
    n_regions : int
        See the corresponding argument in
        :meth:`PRegionsExact.fit_from_scipy_sparse_matrix`.
    solver : str
        See the corresponding argument in
        :meth:`PRegionsExact.fit_from_scipy_sparse_matrix`.
    metric : function
        A function fulfilling the 4 conditions described in the docsting of
        :func:`region.util.get_metric_function`.

    Returns
    -------
    result : :class:`numpy.ndarray`
        A one-dimensional array containing each area's region label.
    """
    print("running FLOW algorithm")  # TODO: rm
    prob = LpProblem("Flow", LpMinimize)

    # Parameters of the optimization problem
    n_areas = adj.shape[0]
    I = list(range(n_areas))  # index for areas
    II = [(i, j)
          for i in I
          for j in I]
    II_upper_triangle = [(i, j) for i, j in II if i < j]
    K = range(n_regions)  # index for regions
    d = {(i, j): metric(attr[i].reshape(attr.shape[1], 1),  # reshaping to...
                        attr[j].reshape(attr.shape[1], 1))  # ...avoid warnings
         for i, j in II_upper_triangle}

    # Decision variables
    t = LpVariable.dicts(
        "t",
        ((i, j) for i, j in II_upper_triangle),
        lowBound=0, upBound=1, cat=LpInteger)
    f = LpVariable.dicts(           # The amount of flow (non-negative integer)
        "f",                        # from area i to j in region k.
        ((i, j, k) for i in I for j in neighbors(adj, i) for k in K),
        lowBound=0, cat=LpInteger)
    y = LpVariable.dicts(  # 1 if area i is assigned to region k. 0 otherwise.
        "y",
        ((i, k) for i in I for k in K),
        lowBound=0, upBound=1, cat=LpInteger)
    w = LpVariable.dicts(  # 1 if area i is chosen as a sink. 0 otherwise.
        "w",
        ((i, k) for i in I for k in K),
        lowBound=0, upBound=1, cat=LpInteger)

    # Objective function
    # (20) in Duque et al. (2011): "The p-Regions Problem"
    prob += lpSum(d[i, j] * t[i, j] for i, j in II_upper_triangle)

    # Constraints
    # (21) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        prob += sum(y[i, k] for k in K) == 1
    # (22) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for k in K:
            prob += w[i, k] <= y[i, k]
    # (23) in Duque et al. (2011): "The p-Regions Problem"
    for k in K:
        prob += sum(w[i, k] for i in I) == 1
    # (24) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in neighbors(adj, i):
            for k in K:
                prob += f[i, j, k] <= y[i, k] * (n_areas - n_regions)
    # (25) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in neighbors(adj, i):
            for k in K:
                prob += f[i, j, k] <= y[j, k] * (n_areas - n_regions)
    # (26) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for k in K:
            lhs = sum(f[i, j, k] - f[j, i, k] for j in neighbors(adj, i))
            prob += lhs >= y[i, k] - (n_areas - n_regions) * w[i, k]
    # (27) in Duque et al. (2011): "The p-Regions Problem"
    for i, j in II_upper_triangle:
        for k in K:
            prob += t[i, j] >= y[i, k] + y[j, k] - 1
    # (28) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (29) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (30) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (31) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition

    # Solve the optimization problem
    solver = get_solver_instance(solver)
    prob.solve(solver)
    result = np.zeros(n_areas)
    for i in I:
        for k in K:
            if y[i, k].varValue == 1:
                result[i] = k
    return result


def _order(adj, attr, n_regions, solver, metric):
    """
    Parameters
    ----------
    adj : class:`scipy.sparse.csr_matrix`
        Refer to the corresponding argument in :func:`_flow`.
    attr : :class:`numpy.ndarray`
        Refer to the corresponding argument in :func:`_flow`.
    n_regions : int
        Refer to the corresponding argument in :func:`_flow`.
    solver : str
        Refer to the corresponding argument in :func:`_flow`.
    metric : function
        Refer to the corresponding argument in :func:`_flow`.

    Returns
    -------
    result : :class:`numpy.ndarray`
        Refer to the return value in :func:`_flow`.
    """
    print("running ORDER algorithm")  # TODO: rm
    prob = LpProblem("Order", LpMinimize)

    # Parameters of the optimization problem
    n_areas = attr.shape[0]
    I = list(range(n_areas))  # index for areas
    II = [(i, j)
          for i in I
          for j in I]
    II_upper_triangle = [(i, j) for i, j in II if i < j]
    K = range(n_regions)  # index for regions
    O = range(n_areas - n_regions)  # index for orders
    d = {(i, j): metric(attr[i].reshape(attr.shape[1], 1),  # reshaping to...
                        attr[j].reshape(attr.shape[1], 1))  # ...avoid warnings
         for i, j in II_upper_triangle}

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
    # (13) in Duque et al. (2011): "The p-Regions Problem"
    prob += lpSum(d[i, j] * t[i, j] for i, j in II_upper_triangle)

    # Constraints
    # (14) in Duque et al. (2011): "The p-Regions Problem"
    for k in K:
        prob += sum(x[i, k, 0] for i in I) == 1
    # (15) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        prob += sum(x[i, k, o] for k in K for o in O) == 1
    # (16) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for k in K:
            for o in range(1, len(O)):
                    prob += x[i, k, o] <= \
                            sum(x[j, k, o-1] for j in neighbors(adj, i))
    # (17) in Duque et al. (2011): "The p-Regions Problem"
    for i, j in II_upper_triangle:
        for k in K:
            summ = sum(x[i, k, o] + x[j, k, o] for o in O) - 1
            prob += t[i, j] >= summ
    # (18) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (19) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition

    # Solve the optimization problem
    solver = get_solver_instance(solver)
    prob.solve(solver)
    result = np.zeros(n_areas)
    for i in I:
        for k in K:
            for o in O:
                if x[i, k, o].varValue == 1:
                    result[i] = k
    return result


def _tree(adj, attr, n_regions, solver, metric):
    """
    Parameters
    ----------
    adj : class:`scipy.sparse.csr_matrix`
        Refer to the corresponding argument in :func:`_flow`.
    attr : :class:`numpy.ndarray`
        Refer to the corresponding argument in :func:`_flow`.
    n_regions : int
        Refer to the corresponding argument in :func:`_flow`.
    solver : str
        Refer to the corresponding argument in :func:`_flow`.
    metric : function
        Refer to the corresponding argument in :func:`_flow`.

    Returns
    -------
    result : :class:`numpy.ndarray`
        Refer to the return value in :func:`_flow`.
    """
    print("running TREE algorithm")  # TODO: rm
    prob = LpProblem("Tree", LpMinimize)

    # Parameters of the optimization problem
    n_areas = attr.shape[0]
    I = list(range(n_areas))  # index for areas
    II = [(i, j)
          for i in I
          for j in I]
    II_upper_triangle = [(i, j) for i, j in II if i < j]
    d = {(i, j): metric(attr[i].reshape(attr.shape[1], 1),  # reshaping to...
                        attr[j].reshape(attr.shape[1], 1))  # ...avoid warnings
         for i, j in II}
    # Decision variables
    t = LpVariable.dicts(
        "t",
        ((i, j) for i, j in II),
        lowBound=0, upBound=1, cat=LpInteger)
    x = LpVariable.dicts(
        "x",
        ((i, j) for i, j in II),
        lowBound=0, upBound=1, cat=LpInteger)
    u = LpVariable.dicts(
        "u",
        (i for i in I),
        lowBound=0, cat=LpInteger)

    # Objective function
    # (3) in Duque et al. (2011): "The p-Regions Problem"
    prob += lpSum(d[i, j] * t[i, j] for i, j in II_upper_triangle)

    # Constraints
    # (4) in Duque et al. (2011): "The p-Regions Problem"
    lhs = lpSum(x[i, j] for i in I for j in neighbors(adj, i))
    prob += lhs == n_areas - n_regions
    # (5) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        prob += lpSum(x[i, j] for j in neighbors(adj, i)) <= 1
    # (6) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in I:
            for m in I:
                if i != j and i != m and j != m:
                    prob += t[i, j] + t[i, m] - t[j, m] <= 1
    # (7) in Duque et al. (2011): "The p-Regions Problem"
    for i, j in II:
        prob += t[i, j] - t[j, i] == 0
    # (8) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in neighbors(adj, i):
            prob += x[i, j] <= t[i, j]
    # (9) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        for j in neighbors(adj, i):
            prob += u[i] - u[j] + (n_areas - n_regions) * x[i, j] \
                    + (n_areas - n_regions - 2) * x[j, i] \
                    <= n_areas - n_regions - 1
    # (10) in Duque et al. (2011): "The p-Regions Problem"
    for i in I:
        prob += u[i] <= n_areas - n_regions
        prob += u[i] >= 1
    # (11) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition
    # (12) in Duque et al. (2011): "The p-Regions Problem"
    # already in LpVariable-definition

    # Solve the optimization problem
    solver = get_solver_instance(solver)
    prob.solve(solver)

    # build a list of regions like [[0, 1, 2, 5], [3, 4, 6, 7, 8]]
    idx_copy = set(I)
    regions = [[] for _ in range(n_regions)]
    for i in range(n_regions):
        area = idx_copy.pop()
        regions[i].append(area)

        for other_area in idx_copy:
            if t[area, other_area].varValue == 1:
                regions[i].append(other_area)

        idx_copy.difference_update(regions[i])

    result = array_from_region_list(regions)
    return result
