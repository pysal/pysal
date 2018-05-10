import abc
from collections import deque
import math
import random

import numpy as np
import networkx as nx

from region.csgraph_utils import sub_adj_matrix, neighbors, is_connected
from region.objective_function import ObjectiveFunctionPairwise
from region.p_regions.azp_util import AllowMoveStrategy, \
                                            AllowMoveAZP,\
                                            AllowMoveAZPSimulatedAnnealing
from region.util import array_from_df_col, array_from_dict_values, \
    assert_feasible, copy_func, count, generate_initial_sol, \
    make_move, Move, pop_randomly_from, random_element_from,\
    scipy_sparse_matrix_from_w, separate_components, w_from_gdf,\
    array_from_graph_or_dict, scipy_sparse_matrix_from_dict


class AZP:
    """
    Class offering the implementation of the AZP algorithm (see [OR1995]_).

    Attributes
    ----------
    labels_ : :class:`numpy.ndarray`
        Each element is a region label specifying to which region the
        corresponding area was assigned to by the last run of a fit-method.
    """
    def __init__(self, allow_move_strategy=None, random_state=None):
        """
        Parameters
        ----------
        allow_move_strategy : None or :class:`AllowMoveStrategy`, default: None
            If None, then the AZP algorithm in [OR1995]_ is chosen.
            For a different behavior for allowing moves an AllowMoveStrategy
            instance can be passed as argument.
        random_state : None, int, str, bytes, or bytearray, default: None
            Random seed.
        """
        self.n_regions = None
        self.labels_ = None
        self.random_state = random_state
        random.seed(self.random_state)

        if isinstance(allow_move_strategy, AllowMoveStrategy):
            self.allow_move_strategy = allow_move_strategy
        elif allow_move_strategy is None:
            self.allow_move_strategy = AllowMoveAZP()
        else:
            raise ValueError("The allow_move_strategy argument must be either "
                             "None, or an instance of AllowMoveStrategy.")

        self.objective_func = None

    def fit_from_scipy_sparse_matrix(
            self, adj, attr, n_regions, initial_labels=None,
            objective_func=ObjectiveFunctionPairwise()):
        """
        Perform the AZP algorithm as described in [OR1995]_.

        The resulting region labels are assigned to the instance's
        :attr:`labels_` attribute.

        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            Adjacency matrix representing the contiguity relation.
        attr : :class:`numpy.ndarray`
            Array (number of areas x number of attributes) of areas' attributes
            relevant to clustering.
        n_regions : `int`
            Number of desired regions.
        initial_labels : :class:`numpy.ndarray` or None, default: None
            One-dimensional array of labels at the beginning of the algorithm.
            If None, then a random initial clustering will be generated.
        objective_func : :class:`region.objective_function.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            The objective function to use.
        """
        if attr.ndim == 1:
            attr = attr.reshape(adj.shape[0], -1)
        self.allow_move_strategy.attr_all = attr
        self.objective_func = objective_func
        # step 1
        if initial_labels is not None:
            assert_feasible(initial_labels, adj, n_regions)
            initial_labels_gen = separate_components(adj, initial_labels)
        else:
            initial_labels_gen = generate_initial_sol(adj, n_regions)
        labels = -np.ones(adj.shape[0])
        for labels_comp in initial_labels_gen:
            comp_idx = np.where(labels_comp != -1)[0]
            adj_comp = sub_adj_matrix(adj, comp_idx)
            labels_comp = labels_comp[comp_idx]
            attr_comp = attr[comp_idx]
            self.allow_move_strategy.start_new_component(
                    labels_comp, attr_comp, self.objective_func, comp_idx)
            
            labels_comp = self._azp_connected_component(
                    adj_comp, labels_comp, attr_comp)
            labels[comp_idx] = labels_comp

        self.n_regions = n_regions
        self.labels_ = labels

    fit = copy_func(fit_from_scipy_sparse_matrix)
    fit.__doc__ = "Alias for :meth:`fit_from_scipy_sparse_matrix`.\n\n" \
                  + fit_from_scipy_sparse_matrix.__doc__

    def fit_from_w(self, w, attr, n_regions, initial_labels=None,
                   objective_func=ObjectiveFunctionPairwise()):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        w : :class:`libpysal.weights.weights.W`
            W object representing the contiguity relation.
        attr : :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        n_regions : `int`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        initial_labels : :class:`numpy.ndarray` or None, default: None
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = scipy_sparse_matrix_from_w(w)
        self.fit_from_scipy_sparse_matrix(adj, attr, n_regions, initial_labels,
                                          objective_func=objective_func)

    def fit_from_networkx(self, graph, attr, n_regions, initial_labels=None,
                          objective_func=ObjectiveFunctionPairwise()):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        graph : `networkx.Graph`
            Graph representing the contiguity relation.
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
        n_regions : `int`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        initial_labels : str or dict or None, default: None
            If str, then the string names the graph's attribute holding the
            information about the initial clustering.
            If dict, then each key is a node and each value is the region the
            key area is assigned to at the beginning of the algorithm.
            If None, then a random initial clustering will be generated.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        attr = array_from_graph_or_dict(graph, attr)
        if initial_labels is not None:
            initial_labels = array_from_graph_or_dict(graph, initial_labels)
        self.fit_from_scipy_sparse_matrix(adj, attr, n_regions, initial_labels,
                                          objective_func=objective_func)

    def fit_from_geodataframe(self, gdf, attr, n_regions, contiguity="rook",
                              initial_labels=None,
                              objective_func=ObjectiveFunctionPairwise()):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        gdf : :class:`geopandas.GeoDataFrame`

        attr : `str` or `list`
            The clustering-relevant attributes (columns of the GeoDataFrame
            `gdf`) are specified as string (for one column) or list of strings
            (for multiple columns).
        n_regions : `int`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        contiguity : {"rook", "queen"}, default: "rook"
            Defines the contiguity relationship between areas. Possible
            contiguity definitions are:

            * "rook" - Rook contiguity.
            * "queen" - Queen contiguity.

        initial_labels : :class:`numpy.ndarray` or None, default: None
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        w = w_from_gdf(gdf, contiguity)
        attr = array_from_df_col(gdf, attr)
        self.fit_from_w(w, attr, n_regions, initial_labels,
                        objective_func=objective_func)

    def fit_from_dict(self, neighbor_dict, attr, n_regions,
                      initial_labels=None,
                      objective_func=ObjectiveFunctionPairwise()):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        neighbor_dict : `dict`
            Each key is an area and each value is an iterable of the key area's
            neighbors.
        attr : `dict`
            Each key is an area and each value is the corresponding
            clustering-relevant attribute.
        n_regions : `int`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        initial_labels : `dict` or None, default: None
            Each key represents an area. Each value represents the region, the
            corresponding area is assigned to at the beginning of the
            algorithm.
            If None, then a random initial clustering will be generated.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        sorted_areas = sorted(neighbor_dict)

        adj = scipy_sparse_matrix_from_dict(neighbor_dict)
        attr_arr = array_from_dict_values(attr, sorted_areas)

        if initial_labels is not None:
            initial_labels = array_from_dict_values(initial_labels,
                                                    sorted_areas,
                                                    flat_output=True,
                                                    dtype=np.int32)
        self.fit_from_scipy_sparse_matrix(adj, attr_arr, n_regions,
                                          initial_labels,
                                          objective_func=objective_func)

    def _azp_connected_component(self, adj, initial_clustering, attr):
        """
        Implementation of the AZP algorithm for a spatially connected set of
        areas (i.e. for every area there is a path to every other area).

        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            Adjacency matrix representing the contiguity relation. The matrix'
            shape is (N, N) where N denotes the number of areas in the
            currently considered connected component.
        initial_clustering : :class:`numpy.ndarray`
            Array of labels. Shape: (N,) where N denotes the number of areas in
            the currently considered connected component.
        attr : :class:`numpy.ndarray`
            Array of labels. Shape: (N, M) where N denotes the number of areas
            in the currently considered connected component and M denotes the
            number of attributes per area.

        Returns
        -------
        labels : :class:`numpy.ndarray`
            One-dimensional array of region labels after the AZP algorithm has
            been performed. Only region labels of the currently considered
            connected component are returned.
        """
        # if there is only one region in the initial solution, just return it.
        distinct_regions = list(np.unique(initial_clustering))
        if len(distinct_regions) == 1:
            return initial_clustering
        distinct_regions_copy = distinct_regions.copy()

        #  step 2: make a list of the M regions
        labels = initial_clustering

        # print("Init with: ", initial_clustering)
        obj_val_start = float("inf")
        obj_val_end = self.allow_move_strategy.objective_val
        print("start with obj. val.:", obj_val_end)

        region_neighbors = {}
        for region in distinct_regions:
            region_areas = set(np.where(labels == region)[0])
            # print("region consists of areas", region_areas)
            # print("adj", adj.todense())
            neighs = set()
            for area in region_areas:
                neighs.update(neighbors(adj, area))
            region_neighbors[region] = neighs.difference(region_areas)
        print("RN", region_neighbors)
        del neighs

        # step 7: Repeat until no further improving moves are made
        while obj_val_end < obj_val_start:  # improvement
            print("obj_val:", obj_val_start, "-->", obj_val_end,
                  "...continue...")
            # print("=" * 45)
            # print("step 7")
            obj_val_start = float(obj_val_end)
            print("step 2")
            distinct_regions = distinct_regions_copy.copy()
            # step 6: when the list for region K is exhausted return to step 3
            # and select another region and repeat steps 4-6
            # print("-" * 35)
            # print("step 6")
            while distinct_regions:
                # step 3: select & remove any region K at random from this list
                print("step 3")
                recipient = pop_randomly_from(distinct_regions)
                print("  chosen region:", recipient)
                while True:
                    # step 4: identify a set of zones bordering on members of
                    # region K that could be moved into region K without
                    # destroying the internal contiguity of the donor region(s)
                    print("step 4")
                    # print("  labels:", labels)
                    # print("  neighbors per region:")
                    # print(region_neighbors)
                    candidates = []
                    for neigh in region_neighbors[recipient]:
                        neigh_region = labels[neigh]
                        sub_adj = sub_adj_matrix(
                                adj,
                                np.where(labels == neigh_region)[0],
                                wo_nodes=neigh)
                        if is_connected(sub_adj):
                            # if area is alone in its region, it must stay
                            if count(labels, neigh_region) > 1:
                                candidates.append(neigh)
                    # step 5: randomly select zones from this list until either
                    # there is a local improvement in the current value of the
                    # objective function or a move that is equivalently as good
                    # as the current best. Then make the move, update the list
                    # of candidate zones, and return to step 4 or else repeat
                    # step 5 until the list is exhausted.
                    print("step 5")
                    while candidates:
                        print("step 5 loop")
                        cand = pop_randomly_from(candidates)
                        if self.allow_move_strategy(cand, recipient, labels):
                            donor = labels[cand]
                            print("  MOVING {} from {} to {}".format(
                                    cand, donor, recipient))
                            make_move(cand, recipient, labels)

                            region_neighbors[donor].add(cand)
                            region_neighbors[recipient].discard(cand)

                            neighs_of_cand = neighbors(adj, cand)

                            recipient_region_areas = set(
                                    np.where(labels == recipient)[0])
                            region_neighbors[recipient].update(neighs_of_cand)
                            region_neighbors[recipient].difference_update(
                                    recipient_region_areas)

                            donor_region_areas = set(
                                    np.where(labels == donor)[0])
                            not_donor_neighs_anymore = set(
                                    area for area in neighs_of_cand
                                    if not any(a in donor_region_areas
                                               for a in neighbors(adj, area)))
                            region_neighbors[donor].difference_update(
                                    not_donor_neighs_anymore)
                            print("improved objective to {}".format(float(self.allow_move_strategy.objective_val)))
                            break
                    else:
                        print(self.allow_move_strategy, "denied move.")
                        break
            obj_val_end = float(self.allow_move_strategy.objective_val)
        return labels


class AZPSimulatedAnnealing:
    """
    Class offering the implementation of the AZP-SA algorithm (see [OR1995]_).

    Attributes
    ----------
    labels_ : :class:`numpy.ndarray`
        Each element is a region label specifying to which region the
        corresponding area was assigned to by the last run of a fit-method.
    """
    def __init__(self, init_temperature=None,
                 max_iterations=float("inf"), sa_moves_term=10,
                 nonmoving_steps_before_stop=3,
                 repetitions_before_termination=5, random_state=None):
        """
        Parameters
        ----------
        init_temperature : float
            The initial temperature used in the simulated annealing algorithm.
        max_iterations : int or {float("inf")}, default: float("inf")
            Termination condition for step b: Terminate if the AZP algorithm
            has run for `max_iterations` times.
        sa_moves_term : int, default: 0
            Termination condition for step b: Count the SA-moves made by the
            repeated runs of the AZP (modified in step 5) and terminate after
            the AZP run that made the cumulative number of SA-moves reach or
            exceed `sa_moves_term`.
        nonmoving_steps_before_stop : int, default: 3
            Termination condition: Repeat steps b and c until no further moves
            occur for `nonmoving_steps_before_stop` times.
        repetitions_before_termination : int, default: 5
            Termination condition not present in [OR1995]_: Terminate if the
            AZP runs returned a given solution for
            `repetitions_before_termination` times.
        random_state : None, int, str, bytes, or bytearray
            Random seed.
        """
        self.allow_move_strategy = None
        self.azp = None

        if init_temperature is not None:
            self.init_temperature = init_temperature
        else:
            raise NotImplementedError("TODO")  # todo

        self.maxit = max_iterations
        self.sa_moves_term = sa_moves_term

        self.sa_moves_term_reached = False
        self.move_made = False
        self.nonmoving_steps_before_stop = nonmoving_steps_before_stop

        self.visited = []
        self.reps_before_termination = repetitions_before_termination

        self.random_state = random_state

        self.n_regions = None

    def fit_from_geodataframe(self, gdf, attr, n_regions,
                              contiguity="rook", initial_labels=None,
                              cooling_factor=0.85,
                              objective_func=ObjectiveFunctionPairwise()):
        """
        Parameters
        ----------
        gdf : :class:`geopandas.GeoDataFrame`
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_geodataframe`.
        attr : `str` or `list`
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_geodataframe`.
        n_regions : `int`
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_geodataframe`.
        contiguity : `str`
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_geodataframe`.
        initial_labels : :class:`numpy.ndarray` or None, default: None
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_geodataframe`.
        cooling_factor : float, default: 0.85
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        w = w_from_gdf(gdf, contiguity)
        attr = array_from_df_col(gdf, attr)
        self.fit_from_w(w, attr, n_regions, initial_labels,
                        cooling_factor=cooling_factor,
                        objective_func=objective_func)

    def fit_from_dict(self, neighbor_dict, attr, n_regions,
                      initial_labels=None, cooling_factor=0.85,
                      objective_func=ObjectiveFunctionPairwise()):
        """
        Parameters
        ----------
        neighbor_dict : `dict`
            Refer to the corresponding argument in :meth:`AZP.fit_from_dict`.
        attr : `dict`
            Refer to the corresponding argument in :meth:`AZP.fit_from_dict`.
        n_regions : `int`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        initial_labels : `dict` or None, default: None
            Refer to the corresponding argument in :meth:`AZP.fit_from_dict`.
        cooling_factor : float, default: 0.85
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        sorted_areas = sorted(neighbor_dict)
        adj = scipy_sparse_matrix_from_dict(neighbor_dict)
        attr_arr = array_from_dict_values(attr, sorted_areas)

        if initial_labels is not None:
            initial_labels = array_from_dict_values(initial_labels,
                                                    sorted_areas,
                                                    flat_output=True,
                                                    dtype=np.int32)
        self.fit_from_scipy_sparse_matrix(
                adj, attr_arr, n_regions, initial_labels=initial_labels,
                cooling_factor=cooling_factor, objective_func=objective_func)

    def fit_from_networkx(self, graph, attr, n_regions, initial_labels=None,
                          cooling_factor=0.85,
                          objective_func=ObjectiveFunctionPairwise()):
        """
        Parameters
        ----------
        graph : `networkx.Graph`
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_networkx`.
        attr : str, list or dict
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_networkx`.
        n_regions : `int`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        initial_labels : str or dict or None, default: None
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_networkx`.
        cooling_factor : float, default: 0.85
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        attr = array_from_graph_or_dict(graph, attr)
        if initial_labels is not None:
            initial_labels = array_from_graph_or_dict(graph, initial_labels)
        self.fit_from_scipy_sparse_matrix(adj, attr, n_regions, initial_labels,
                                          cooling_factor=cooling_factor,
                                          objective_func=objective_func)

    def fit_from_scipy_sparse_matrix(
            self, adj, attr, n_regions, initial_labels=None,
            cooling_factor=0.85, objective_func=ObjectiveFunctionPairwise()):
        """
        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_scipy_sparse_matrix`.
        attr : :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_scipy_sparse_matrix`.
        n_regions : `int`
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_scipy_sparse_matrix`.
        initial_labels : :class:`numpy.ndarray` or None, default: None
            Refer to the corresponding argument in
            :meth:`AZP.fit_from_scipy_sparse_matrix`.
        cooling_factor : float, default: 0.85
            Float :math:`\\in (0, 1)` specifying the cooling factor for the
            simulated annealing.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        if not (0 < cooling_factor < 1):
            raise ValueError("The cooling_factor argument must be greater "
                             "than 0 and less than 1")
        if attr.ndim == 1:
            attr = attr.reshape(adj.shape[0], -1)
        self.allow_move_strategy = AllowMoveAZPSimulatedAnnealing(
                init_temperature=self.init_temperature,
                sa_moves_term=self.sa_moves_term)
        self.allow_move_strategy.register_sa_moves_term(self.sa_moves_alert)
        self.allow_move_strategy.register_move_made(self.move_made_alert)

        self.azp = AZP(allow_move_strategy=self.allow_move_strategy,
                       random_state=self.random_state)
        # todo: rm print() calls
        # step a
        # print(("#"*60 + "\n") * 5 + "STEP A")
        t = self.init_temperature
        nonmoving_steps = 0
        # step d: repeat step b and c
        while nonmoving_steps < self.nonmoving_steps_before_stop:
            # print(("#"*60 + "\n") * 2 + "STEP B")
            it = 0
            self.sa_moves_term_reached = False
            self.allow_move_strategy.reset()
            # step b
            while it < self.maxit and not self.sa_moves_term_reached:
                it += 1
                old_sol = initial_labels
                self.azp.fit_from_scipy_sparse_matrix(adj, attr, n_regions,
                                                      initial_labels,
                                                      objective_func)
                initial_labels = self.azp.labels_

                # print("old_sol", old_sol)
                # print("new_sol", initial_labels)
                if old_sol is not None:
                    # print("EQUAL" if (old_sol == initial_labels).all()
                    #       else "NOT EQUAL")
                    if (old_sol == initial_labels).all():
                        # print("BREAK")
                        break
            # print("visited", self.visited)
            # added termination condition (not in Openshaw & Rao (1995))
            # print(initial_labels)
            if self.visited.count(tuple(initial_labels)) \
                    >= self.reps_before_termination:
                print("VISITED", initial_labels, "FOR",
                      self.reps_before_termination,
                      "TIMES --> TERMINATING.")
                break
            self.visited.append(tuple(initial_labels))
            # step c
            # print(("#"*60 + "\n") * 2 + "STEP C")
            t *= cooling_factor
            self.allow_move_strategy.update_temperature(t)

            if self.move_made:
                # print("MOVE MADE")
                self.move_made = False
            else:
                # print("NO MOVE MADE")
                nonmoving_steps += 1
        self.labels_ = initial_labels

    def fit_from_w(self, w, attr, n_regions, initial_labels=None,
                   cooling_factor=0.85,
                   objective_func=ObjectiveFunctionPairwise()):
        """
        Parameters
        ----------
        w : :class:`libpysal.weights.weights.W`
            Refer to the corresponding argument in :meth:`AZP.fit_from_w`.
        attr : :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        n_regions : `int`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        initial_labels : :class:`numpy.ndarray` or None, default: None
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        cooling_factor : float, default: 0.85
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = scipy_sparse_matrix_from_w(w)
        self.fit_from_scipy_sparse_matrix(adj, attr, n_regions, initial_labels,
                                          cooling_factor=cooling_factor,
                                          objective_func=objective_func)

    def sa_moves_alert(self):
        self.sa_moves_term_reached = True

    def move_made_alert(self):
        self.move_made = True


class AZPTabu(AZP, abc.ABC):
    """
    Superclass for tabu variants of the AZP.
    """
    def _make_move(self, area, new_region, labels):
        old_region = labels[area]
        make_move(area, new_region, labels)
        # step 5: Tabu the reverse move for R iterations.
        reverse_move = Move(area, new_region, old_region)
        self.tabu.append(reverse_move)

    def reset_tabu(self, tabu_len=None):
        tabu_len = self.tabu.maxlen if tabu_len is None else tabu_len
        self.tabu = deque([], tabu_len)


class AZPBasicTabu(AZPTabu):
    """
    Implementation of the AZP with basic tabu (refer to [OR1995]_).

    Attributes
    ----------
    labels_ : :class:`numpy.ndarray`
        Each element is a region label specifying to which region the
        corresponding area was assigned to by the last run of a fit-method.
    """
    def __init__(self, tabu_length=None, repetitions_before_termination=5,
                 random_state=None):
        """
        Parameters
        ----------
        tabu_length : numbers.Integral
            The size of the tabu list.
        repetitions_before_termination : numbers.integral
            This argument specifies a termination condition. If a solution has
            been visited for `repetitions_before_termination` times, the
            clustering function will terminate.
        random_state : None, int, str, bytes, or bytearray, default: None
            Refer to the corresponding argument in
            :meth:`AZP.__init__`.
        """
        self.tabu = deque([], tabu_length)
        self.visited = []
        self.reps_before_termination = repetitions_before_termination
        super().__init__(random_state=random_state)

    def _azp_connected_component(self, adj, initial_clustering, attr):
        """
        Implementation of the basic tabu version of the AZP algorithm (refer
        to [OR1995]_) for a spatially connected set of areas (i.e. for every
        area there is a path to every other area).

        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            Refer to the corresponding argument in
            :meth:`AZP._azp_connected_component`.
        initial_clustering : :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`AZP._azp_connected_component`.
        attr : :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`AZP._azp_connected_component`.

        Returns
        -------
        labels : :class:`numpy.ndarray`
            Refer to the return value in :meth:`AZP._azp_connected_component`.
        """
        self.reset_tabu()
        # if there is only one region in the initial solution, just return it.
        distinct_regions = list(np.unique(initial_clustering))
        if len(distinct_regions) == 1:
            return initial_clustering

        #  step 2: make a list of the M regions
        labels = initial_clustering

        # todo: rm print-statements
        # print("Init with: ", initial_clustering)
        visited = []
        stop = False
        while True:
            # print("visited", visited)
            # added termination condition (not in Openshaw & Rao (1995))
            label_tup = tuple(labels)
            if visited.count(label_tup) >= self.reps_before_termination:
                stop = True
                # print("VISITED", label_tup, "FOR",
                #       self.reps_before_termination_,
                #       "TIMES --> TERMINATING BEFORE NEXT NON-IMPROVING MOVE")
            visited.append(label_tup)
            # print("=" * 45)
            # print("obj_value:", obj_val_end)
            # print(region_list)
            # print("-" * 35)
            # step 1 Find the global best move that is not prohibited or tabu.
            # print("step 1")
            # find possible moves (globally)
            best_move = None
            best_objval_diff = float("inf")
            for area in range(labels.shape[0]):
                old_region = labels[area]
                sub_adj = sub_adj_matrix(
                            adj,
                            np.where(labels == old_region)[0],
                            wo_nodes=area)
                # moving the area must not destroy spatial contiguity in donor
                # region and if area is alone in its region, it must stay:
                if is_connected(sub_adj) and count(labels, old_region) > 1:
                    for neigh in neighbors(adj, area):
                        new_region = labels[neigh]
                        if new_region != old_region:
                            possible_move = Move(area, old_region, new_region)
                            if possible_move not in self.tabu:
                                objval_diff = self.objective_func.update(
                                        possible_move.area,
                                        possible_move.new_region, labels, attr)
                                if objval_diff < best_objval_diff:
                                    best_move = possible_move
                                    best_objval_diff = objval_diff
            # print("  best move", best_move, "objval_diff", best_objval_diff)
            # step 2: Make this move if it is an improvement or equivalet in
            # value.
            print("step 2")
            if best_move is not None and best_objval_diff <= 0:
                print(labels)
                print("IMPROVING MOVE")
                self._make_move(best_move.area, best_move.new_region, labels)
            else:
                # step 3: if no improving move can be made, then see if a tabu
                # move can be made which improves on the current local best
                # (termed an aspiration move)
                print("step 3")
                print("Tabu:", self.tabu)
                improving_tabus = [
                    move for move in self.tabu
                    if labels[move.area] == move.old_region and
                    self.objective_func.update(move.area, move.new_region,
                                               labels, attr) < 0
                ]
                print(labels)
                if improving_tabus:
                    aspiration_move = random_element_from(improving_tabus)
                    # print("ASPIRATION MOVE")
                    self._make_move(aspiration_move.area,
                                    aspiration_move.new_region, labels)
                else:
                    # step 4: If there is no improving move and no aspirational
                    # move, then make the best move even if it is nonimproving
                    # (that is, results in a worse value of the objective
                    # function).
                    print("step 4")
                    print("No improving, no aspiration ==> make the best move")
                    if stop:
                        break
                    if best_move is not None:
                        self._make_move(best_move.area, best_move.new_region,
                                        labels)
        return labels


class AZPReactiveTabu(AZPTabu):
    """
    Implementation of the AZP with reactive tabu (refer to [OR1995]_).

    Attributes
    ----------
    labels_ : :class:`numpy.ndarray`
        Each element is a region label specifying to which region the
        corresponding area was assigned to by the last run of a fit-method.
    """
    def __init__(self, max_iterations, k1, k2, random_state=None):
        """
        Parameters
        ----------
        max_iterations : int
           Termination condition: The algorithm terminates after steps 3-11
           (see [OR1995]_) are repeated for `max_max_iterations` times.
        k1 : int
            Defining a necessary condition for jumping from step 7 to step 11
            in the algorithm (see [OR1995]_). Such a jump requires (besides the
            condition involving `k2`) a solution to be visited more than `k1`
            times.
        k2 : int
            Defining a necessary condition for jumping from step 7 to step 11
            in the algorithm (see [OR1995]_). Such a jump requires (besides the
            condition involving `k1`) a cycle of solutions to be found at least
            `k2` times.
        random_state : None, int, str, bytes, or bytearray, default: None
            Refer to the corresponding argument in
            :meth:`AZP.__init__`.
        """
        self.tabu = deque([], maxlen=1)
        super().__init__(random_state=random_state)
        self.avg_it_until_rep = 1
        self.rep_counter = 1
        if max_iterations <= 0:
            raise ValueError("The `max_iterations` argument must be > 0.")
        self.maxit = max_iterations
        self.visited = []
        self.k1 = k1
        self.k2 = k2

    def _azp_connected_component(self, adj, initial_labels, attr):
        """
        Implementation of the reactive tabu version of the AZP algorithm (refer
        to [OR1995]_) for a spatially connected set of areas (i.e. for every
        area there is a path to every other area).

        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            Refer to the corresponding argument in
            :meth:`AZP._azp_connected_component`.
        initial_labels : :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`AZP._azp_connected_component`.
        attr : :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`AZP._azp_connected_component`.

        Returns
        -------
        labels : :class:`numpy.ndarray`
            Refer to the return value in :meth:`AZP._azp_connected_component`.
        """
        self.reset_tabu(1)
        # if there is only one region in the initial solution, just return it.
        distinct_regions = list(np.unique(initial_labels))
        if len(distinct_regions) == 1:
            return initial_labels


        #  step 2: make a list of the M regions
        labels = initial_labels

        # todo: rm print-statements
        print("Init with: ", initial_labels)
        it_since_tabu_len_changed = 0
        obj_val_start = float("inf")
        # step 12: Repeat steps 3-11 until either no further improvements are
        # made or a maximum number of iterations are exceeded.
        for it in range(self.maxit):
            print("=" * 45)
            obj_val_end = self.objective_func(labels, attr)
            print("obj_value:", obj_val_end)
            if not obj_val_end < obj_val_start:
                break  # step 12
            obj_val_start = obj_val_end

            it_since_tabu_len_changed += 1
            print("-" * 35)
            # step 3: Define the list of all possible moves that are not tabu
            # and retain regional connectivity.
            print("step 3")
            possible_moves = []
            for area in range(labels.shape[0]):
                old_region = labels[area]
                sub_adj = sub_adj_matrix(
                            adj,
                            np.where(labels == old_region)[0],
                            wo_nodes=area)
                # moving the area must not destroy spatial contiguity in donor
                # region and if area is alone in its region, it must stay:
                if is_connected(sub_adj) and count(labels, old_region) > 1:
                    for neigh in neighbors(adj, area):
                        new_region = labels[neigh]
                        if new_region != old_region:
                            possible_move = Move(area, old_region, new_region)
                            if possible_move not in self.tabu:
                                possible_moves.append(possible_move)
            # step 4: Find the best nontabu move.
            print("step 4")
            best_move = None
            best_move_index = None
            best_objval_diff = float("inf")
            for i, move in enumerate(possible_moves):
                obj_val_diff = self.objective_func.update(
                        move.area, move.new_region, labels, attr)
                if obj_val_diff < best_objval_diff:
                    best_move_index, best_move = i, move
                    best_objval_diff = obj_val_diff
            # print("  best move:", best_move)
            # step 5: Make the move. Update the tabu status.
            # print("step 5: make", best_move)
            self._make_move(best_move.area, best_move.new_region, labels)
            # step 6: Look up the current zoning system in a list of all zoning
            # systems visited so far during the search. If not found then go
            # to step 10.
            print("step 6")
            # Sets can't be permuted so we convert our list to a set:
            label_tup = tuple(labels)
            if label_tup in self.visited:
                # step 7: If it is found and it has been visited more than K1
                # times already and this cyclical behavior has been found on
                # at least K2 other occasions (involving other zones) then go
                # to step 11.
                print("step 7")
                print("  labels", labels)
                print("  self.visited:", self.visited)
                times_visited = self.visited.count(label_tup)
                cycle = list(reversed(self.visited))
                cycle = cycle[:cycle.index(label_tup) + 1]
                cycle = list(reversed(cycle))
                print("  cycle:", cycle)
                it_until_repetition = len(cycle)
                if times_visited > self.k1:
                    times_cycle_found = 0
                    if self.k2 > 0:
                        for i in range(len(self.visited) - len(cycle)):
                            if self.visited[i:i+len(cycle)] == cycle:
                                times_cycle_found += 1
                                if times_cycle_found >= self.k2:
                                    break
                    if times_cycle_found >= self.k2:
                        # step 11: Delete all stored zoning systems and make P
                        # random moves, P = 1 + self.avg_it_until_rep/2, and
                        # update tabu to preclude a return to the previous
                        # state.
                        print("step 11")
                        # we save the labels such that we can access it if
                        # this step yields a poor solution.
                        last_step = (11, tuple(labels))
                        self.visited = []
                        p = math.floor(1 + self.avg_it_until_rep/2)
                        possible_moves.pop(best_move_index)
                        for _ in range(p):
                            move = possible_moves.pop(
                                    random.randrange(len(possible_moves)))
                            self._make_move(move.area, move.new_region,
                                            labels)
                        continue
                    # step 8: Update a moving average of the repetition
                    # interval self.avg_it_until_rep, and increase the
                    # prohibition period R to 1.1*R.
                    print("step 8")
                    self.rep_counter += 1
                    avg_it = self.avg_it_until_rep
                    self.avg_it_until_rep = 1 / self.rep_counter * \
                        ((self.rep_counter-1)*avg_it + it_until_repetition)

                    self.tabu = deque(self.tabu, 1.1*self.tabu.maxlen)
                    # step 9: If the number of iterations since R was last
                    # changed exceeds self.avg_it_until_rep, then decrease R to
                    # max(0.9*R, 1).
                    print("step 9")
                    if it_since_tabu_len_changed > self.avg_it_until_rep:
                        new_tabu_len = max([0.9*self.tabu.maxlen, 1])
                        new_tabu_len = math.floor(new_tabu_len)
                        self.tabu = deque(self.tabu, new_tabu_len)
                    it_since_tabu_len_changed = 0  # step 8

            # step 10: Save the zoning system and go to step 12.
            print("step 10")
            self.visited.append(tuple(labels))
            last_step = 10

        if last_step == 10:
            try:
                return np.array(self.visited[-2])
            except IndexError:
                return np.array(self.visited[-1])
        # if step 11 was the last one, the result is in last_step[1]
        return np.array(last_step[1])
