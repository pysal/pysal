import networkx as nx

from region.max_p_regions.heuristics import MaxPRegionsHeu
from region.tests.util import compare_region_lists, region_list_from_array

from .data import adj, neighbors_dict, gdf, graph, w, optimal_clustering
# for tests with scalar attr and spatially_extensive_attr
from .data import attr_str, spatially_extensive_attr_str, \
                  attr, spatially_extensive_attr, threshold, \
                  attr_dict, spatially_extensive_attr_dict
# for tests with non-scalar attr and spatially_extensive_attr
from .data import double_attr, double_spatially_extensive_attr, \
                  double_threshold, \
                  double_attr_dict, double_spatially_extensive_attr_dict

import warnings
warnings.filterwarnings("error")


attr = attr.reshape(-1)
spatially_extensive_attr = spatially_extensive_attr.reshape(-1)
# ### TESTS WITH SCALAR attr AND spatially_extensive_attr #####################
# test with csr_matrix
def test_scipy_sparse_matrix():
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_scipy_sparse_matrix(adj, attr,
                                                spatially_extensive_attr,
                                                threshold=threshold)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with a GeoDataFrame
def test_geodataframe_basic():
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_geodataframe(gdf, attr_str,
                                         spatially_extensive_attr_str,
                                         threshold=threshold)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with a dict as areas argument
def test_dict_basic():
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_dict(neighbors_dict, attr_dict,
                                 spatially_extensive_attr_dict,
                                 threshold=threshold)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with Graph
# ... with dicts as attr and spatially_extensive_attr
def test_graph_dict_basic():
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_networkx(graph, attr_dict,
                                     spatially_extensive_attr_dict,
                                     threshold=threshold)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# ... with strings as attr and spatially_extensive_attr
def test_graph_str_basic():
    nx.set_node_attributes(graph, attr_str, attr_dict)
    nx.set_node_attributes(graph, spatially_extensive_attr_str,
                           spatially_extensive_attr_dict)
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_networkx(graph, attr_str,
                                     spatially_extensive_attr_str,
                                     threshold=threshold)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# test with W
def test_w_basic():
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_w(w, attr, spatially_extensive_attr,
                              threshold=threshold)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# ### TESTS WITH NON-SCALAR attr AND spatially_extensive_attr #################
# test with csr_matrix
def test_scipy_sparse_matrix_multi_attr():
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_scipy_sparse_matrix(
            adj, double_attr, double_spatially_extensive_attr,
            threshold=double_threshold)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with a GeoDataFrame
def test_geodataframe_multi_attr():
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_geodataframe(gdf,
                                         [attr_str] * 2,
                                         [spatially_extensive_attr_str] * 2,
                                         threshold=double_threshold)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with a dict as areas argument
def test_dict_multi_attr():
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_dict(neighbors_dict,
                                 double_attr_dict,
                                 double_spatially_extensive_attr_dict,
                                 threshold=double_threshold)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with Graph
# ... with dicts as attr and spatially_extensive_attr
def test_graph_dict_multi_attr():
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_networkx(graph,
                                     double_attr_dict,
                                     double_spatially_extensive_attr_dict,
                                     threshold=double_threshold)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# ... with strings as attr and spatially_extensive_attr
def test_graph_str_multi_attr():
    nx.set_node_attributes(graph, attr_str, attr_dict)
    nx.set_node_attributes(graph, spatially_extensive_attr_str,
                           spatially_extensive_attr_dict)
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_networkx(graph,
                                     [attr_str] * 2,
                                     [spatially_extensive_attr_str] * 2,
                                     threshold=double_threshold)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# test with W
def test_w_multi_attr():
    print(double_threshold)
    cluster_object = MaxPRegionsHeu()
    cluster_object.fit_from_w(w, double_attr, double_spatially_extensive_attr,
                              threshold=double_threshold)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)
