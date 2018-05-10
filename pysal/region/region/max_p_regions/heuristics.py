import random

import networkx as nx
import libpysal.api as ps_api

from region.objective_function import ObjectiveFunctionPairwise
from region.p_regions.azp import AZP
from region.p_regions.azp_util import AllowMoveAZPMaxPRegions
from region.util import array_from_df_col, array_from_dict_values, \
    array_from_graph_or_dict, array_from_region_list, copy_func, \
    find_sublist_containing, pop_randomly_from, raise_distance_metric_not_set,\
    random_element_from,scipy_sparse_matrix_from_w, \
    scipy_sparse_matrix_from_dict, w_from_gdf


class MaxPRegionsHeu:
    def __init__(self, local_search=None, random_state=None):
        """
        Class offering the implementation of the algorithm for solving the
        max-p-regions problem as described in [DAR2012]_.

        Parameters
        ----------
        local_search : None or :class:`AZP` or :class:`AZPSimulatedAnnealing`
            If None, then the AZP is used.
            Pass an instance of :class:`AZP` (or one of its subclasses) or
            :class:`AZPSimulatedAnnealing` to use a customized local search
            algorithm.
        random_state : None, int, str, bytes, or bytearray
            Random seed.
        """
        self.n_regions = None
        self.labels_ = None
        self.local_search = local_search
        self.random_state = random_state
        random.seed(random_state)
        self.metric = raise_distance_metric_not_set

    def fit_from_scipy_sparse_matrix(
            self, adj, attr, spatially_extensive_attr, threshold, max_it=10,
            objective_func=ObjectiveFunctionPairwise()):
        """
        Solve the max-p-regions problem in a heuristic way (see [DAR2012]_).

        The resulting region labels are assigned to the instance's
        :attr:`labels_` attribute.

        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
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
        max_it : int, default: 10
            The maximum number of partitions produced in the algorithm's
            construction phase.
        objective_func : :class:`region.objective_function.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            The objective function to use.
        """
        print("f_f_SCIPY got:\n", attr, "\n", spatially_extensive_attr, "\n", threshold, sep="")
        weights = ps_api.WSP(adj).to_W()
        areas_dict = weights.neighbors
        self.metric = objective_func.metric

        best_partition = None
        best_obj_value = float("inf")
        feasible_partitions = []
        partitions_before_enclaves_assignment = []
        max_p = 0  # maximum number of regions

        # construction phase
        # print("constructing")
        for _ in range(max_it):
            # print(" ", _)
            partition, enclaves = self.grow_regions(
                    adj, attr, spatially_extensive_attr, threshold)
            n_regions = len(partition)
            if n_regions > max_p:
                partitions_before_enclaves_assignment = [(partition, enclaves)]
                max_p = n_regions
            elif n_regions == max_p:
                partitions_before_enclaves_assignment.append((partition,
                                                              enclaves))

        # print("\n" + "assigning enclaves")
        for partition, enclaves in partitions_before_enclaves_assignment:
            # print("  cleaning up in partition", partition)
            feasible_partitions.append(self.assign_enclaves(
                    partition, enclaves, areas_dict, attr))

        for partition in feasible_partitions:
            print(partition, "\n")

        # local search phase
        if self.local_search is None:
            self.local_search = AZP()
        self.local_search.allow_move_strategy = AllowMoveAZPMaxPRegions(
                spatially_extensive_attr, threshold,
                self.local_search.allow_move_strategy)
        for partition in feasible_partitions:
            self.local_search.fit_from_scipy_sparse_matrix(
                    adj, attr, max_p,
                    initial_labels=array_from_region_list(partition),
                    objective_func=objective_func)
            partition = self.local_search.labels_
            # print("optimized partition", partition)
            obj_value = objective_func(partition, attr)
            if obj_value < best_obj_value:
                best_obj_value = obj_value
                best_partition = partition
        self.labels_ = best_partition

    fit = copy_func(fit_from_scipy_sparse_matrix)
    fit.__doc__ = "Alias for :meth:`fit_from_scipy_sparse_matrix`.\n\n" \
                  + fit_from_scipy_sparse_matrix.__doc__

    def fit_from_dict(self, neighbors_dict, attr, spatially_extensive_attr,
                      threshold, max_it=10,
                      objective_func=ObjectiveFunctionPairwise()):
        """
        Solve the max-p-regions problem in a heuristic way (see [DAR2012]_).

        The resulting region labels are assigned to the instance's
        :attr:`labels_` attribute.

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
        max_it : int, default: 10
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
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
                                          threshold=threshold, max_it=max_it,
                                          objective_func=objective_func)

    def fit_from_geodataframe(self, gdf, attr, spatially_extensive_attr,
                              threshold, max_it=10,
                              objective_func=ObjectiveFunctionPairwise(),
                              contiguity="rook"):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        gdf : :class:`geopandas.GeoDataFrame`

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
        max_it : int, default: 10
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
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
                        max_it=max_it, objective_func=objective_func)

    def fit_from_networkx(self, graph, attr, spatially_extensive_attr,
                          threshold, max_it=10,
                          objective_func=ObjectiveFunctionPairwise()):
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
        max_it : int, default: 10
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = nx.to_scipy_sparse_matrix(graph)
        attr = array_from_graph_or_dict(graph, attr)
        sp_ext_attr = array_from_graph_or_dict(graph, spatially_extensive_attr)
        self.fit_from_scipy_sparse_matrix(adj, attr, sp_ext_attr,
                                          threshold=threshold, max_it=max_it,
                                          objective_func=objective_func)

    def fit_from_w(self, w, attr, spatially_extensive_attr, threshold,
                   max_it=10, objective_func=ObjectiveFunctionPairwise()):
        """
        Alternative API for :meth:`fit_from_scipy_sparse_matrix`.

        Parameters
        ----------
        w : :class:`libpysal.weights.weights.W`
            W object representing the contiguity relation.
        attr : :class:`numpy.ndarray`
            Each element specifies an area's attribute which is used for
            calculating the objective function.
        spatially_extensive_attr : :class:`numpy.ndarray`
            Each element specifies an area's spatially extensive attribute
            which is used to ensure that the sum of spatially extensive
            attributes in each region adds up to a threshold defined by the
            `threshold` argument.
        threshold : numbers.Real or :class:`numpy.ndarray`
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        max_it : int, default: 10
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        objective_func : :class:`region.ObjectiveFunction`, default: ObjectiveFunctionPairwise()
            Refer to the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        """
        adj = scipy_sparse_matrix_from_w(w)
        self.fit_from_scipy_sparse_matrix(adj, attr, spatially_extensive_attr,
                                          threshold, max_it=max_it,
                                          objective_func=objective_func)

    def grow_regions(self, adj, attr, spatially_extensive_attr, threshold):
        """
        Parameters
        ----------
        adj : :class:`scipy.sparse.csr_matrix`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        spatially_extensive_attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.
        threshold : numbers.Real or :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.

        Returns
        -------
        result : `tuple`
            `result[0]` is a `list`. Each list element is a `set` of a region's
            areas. Note that not every area is assigned to a region after this
            function has terminated, so they won't be in any of the `set`s in
            `result[0]`.
            `result[1]` is a `list` of areas not assigned to any region.
        """
        # print("grow_regions called with spatially_extensive_attr", spatially_extensive_attr)
        partition = []
        enclave_areas = []
        unassigned_areas = list(range(adj.shape[0]))
        assigned_areas = []

        # todo: rm prints
        while unassigned_areas:
            # print("partition", partition)
            area = pop_randomly_from(unassigned_areas)
            # print("seed in area", area)
            assigned_areas.append(area)
            if (spatially_extensive_attr[area] >= threshold).all():
                # print("  seed --> region :)")
                # print("because", spatially_extensive_attr[area], ">=", threshold)
                partition.append({area})
            else:
                region = {area}
                # print("  all neighbors:", neigh_dict[area])
                # print("  already assigned:", assigned_areas)
                unassigned_neighs = set(adj[area].nonzero()[1]).difference(
                        assigned_areas)
                feasible = True
                spat_ext_attr = spatially_extensive_attr[area].copy()
                while (spat_ext_attr < threshold).any():
                    # print(" ", spat_ext_attr, "<", threshold, "Need neighs!")
                    # print("  potential neighbors:", unassigned_neighs)
                    if unassigned_neighs:
                        neigh = self.find_best_area(region, unassigned_neighs,
                                                    attr)
                        # print(" we choose neighbor", neigh)
                        region.add(neigh)
                        unassigned_neighs.remove(neigh)
                        unassigned_neighs.update(set(adj[neigh].nonzero()[1]))
                        unassigned_neighs.difference_update(assigned_areas)
                        spat_ext_attr += spatially_extensive_attr[neigh]
                        unassigned_areas.remove(neigh)
                        assigned_areas.append(neigh)
                    else:
                        # print("  Oh no! No neighbors left :(")
                        enclave_areas.extend(region)
                        feasible = False
                        # the following line (present in the algorithm in
                        # [DAR2012]) is commented out because it leads to an
                        # infinite loop:
                        # unassigned_areas.extend(region)
                        for area in region:
                            assigned_areas.remove(area)
                        break
                if feasible:
                    partition.append(region)
                # print("  unassigned:", unassigned_areas)
                # print("  assigned:", assigned_areas)
                # print()
        # print("grow_regions partit.:", partition, "enclaves:", enclave_areas)
        return partition, enclave_areas

    def find_best_area(self, region, candidates, attr):
        """
        Parameters
        ----------
        region : iterable
            Each element represents an area.
        candidates : iterable
            Each element represents an area bordering on region.
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.

        Returns
        -------
        best_area :
            An element of `candidates` with minimal dissimilarity when being
            moved to the region `region`.
        """
        candidates = {area: sum(self.metric(attr[area].reshape(1, -1),
                                            attr[area2].reshape(1, -1))
                                for area2 in region)
                      for area in candidates}
        best_candidates = [area for area in candidates
                           if candidates[area] == min(candidates.values())]
        return random_element_from(best_candidates)

    def assign_enclaves(self, partition, enclave_areas, neigh_dict, attr):
        """
        Start with a partial partition (not all areas are assigned to a region)
        and a list of enclave areas (i.e. areas not present in the partial
        partition). Then assign all enclave areas to regions in the partial
        partition and return the resulting partition.

        Parameters
        ----------
        partition : `list`
            Each element (of type `set`) represents a region.
        enclave_areas : `list`
            Each element represents an area.
        neigh_dict : `dict`
            Each key represents an area. Each value is an iterable of the
            corresponding neighbors.
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.

        Returns
        -------
        partition : `list`
            Each element (of type `set`) represents a region.
        """
        # print("partition:", partition, "- enclaves:", enclave_areas)
        while enclave_areas:
            neighbors_of_assigned = [area for area in enclave_areas
                                     if any(neigh not in enclave_areas
                                            for neigh in neigh_dict[area])]
            area = pop_randomly_from(neighbors_of_assigned)
            neigh_regions_idx = []
            for neigh in neigh_dict[area]:
                try:
                    neigh_regions_idx.append(
                        find_sublist_containing(neigh, partition, index=True))
                except LookupError:
                    pass
            region_idx = self.find_best_region_idx(area, partition,
                                                   neigh_regions_idx, attr)
            partition[region_idx].add(area)
            enclave_areas.remove(area)
        return partition

    def find_best_region_idx(self, area, partition, candidate_regions_idx,
                             attr):
        """

        Parameters
        ----------
        area :
            The area to be moved to one of the regions specified by
            `candidate_regions_idx`.
        partition : `list`
            Each element (of type `set`) represents a region.
        candidate_regions_idx : iterable
            Each element is the index of a region in the `partition` list.
        attr : :class:`numpy.ndarray`
            See the corresponding argument in
            :meth:`fit_from_scipy_sparse_matrix`.

        Returns
        -------
        best_idx : int
            The index of a region (w.r.t. `partition`) which has the smallest
            sum of dissimilarities after area `area` is moved to the region.
        """
        dissim_per_idx = {region_idx:
                          sum(self.metric(attr[area].reshape(1, -1),
                                          attr[area2].reshape(1, -1))
                              for area2 in partition[region_idx])
                          for region_idx in candidate_regions_idx}
        minimum_dissim = min(dissim_per_idx.values())
        best_idxs = [idx for idx in dissim_per_idx
                     if dissim_per_idx[idx] == minimum_dissim]
        return random_element_from(best_idxs)
