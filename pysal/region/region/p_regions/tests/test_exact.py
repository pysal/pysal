import networkx as nx
import pytest

from region.p_regions.exact import PRegionsExact
from region.tests.util import region_list_from_array, compare_region_lists
from region.util import dataframe_to_dict

from region.p_regions.tests.data import adj, neighbors_dict, gdf, graph, w, \
                  attr, attr_dict, attr_str, double_attr_str, \
                  double_attr, double_attr_dict, \
                  optimal_clustering


@pytest.fixture(params=["flow", "order", "tree"])
def method(request):
    return request.param


# ### TESTS WITH SCALAR attr ##################################################
# test with csr_matrix
def test_scipy_sparse_matrix(method):
    cluster_object = PRegionsExact()
    cluster_object.fit_from_scipy_sparse_matrix(adj, attr, n_regions=2,
                                                method=method)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with a GeoDataFrame as areas argument
def test_geodataframe(method):
    cluster_object = PRegionsExact()
    cluster_object.fit_from_geodataframe(gdf, attr_str, n_regions=2,
                                         method=method)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# tests with a dict as areas argument
def test_dict(method):
    value_dict = dataframe_to_dict(gdf, attr_str)
    cluster_object = PRegionsExact()
    cluster_object.fit_from_dict(neighbors_dict, value_dict, n_regions=2,
                                 method=method)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)

# tests with Graph
# ... with dicts as attr and spatially_extensive_attr
def test_graph_dict_basic(method):
    cluster_object = PRegionsExact()
    cluster_object.fit_from_networkx(graph, attr_dict, n_regions=2,
                                     method=method)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# ... with strings as attr and spatially_extensive_attr
def test_graph_str_basic(method):
    nx.set_node_attributes(graph, attr_str, attr_dict)
    cluster_object = PRegionsExact()
    cluster_object.fit_from_networkx(graph, attr_str, n_regions=2,
                                     method=method)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# test with W
def test_w_basic(method):
    cluster_object = PRegionsExact()
    cluster_object.fit_from_w(w, attr, n_regions=2, method=method)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# ### TESTS WITH NON-SCALAR attr AND spatially_extensive_attr #################
# test with csr_matrix
def test_scipy_sparse_matrix_multi_attr(method):
    cluster_object = PRegionsExact()
    cluster_object.fit_from_scipy_sparse_matrix(
            adj, double_attr, n_regions=2, method=method)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with a GeoDataFrame
def test_geodataframe_multi_attr(method):
    cluster_object = PRegionsExact()
    cluster_object.fit_from_geodataframe(gdf, double_attr_str, n_regions=2,
                                         method=method)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with a dict as areas argument
def test_dict_multi_attr(method):
    cluster_object = PRegionsExact()
    cluster_object.fit_from_dict(neighbors_dict, double_attr_dict, n_regions=2,
                                 method=method)
    obtained = region_list_from_array(cluster_object.labels_)
    compare_region_lists(obtained, optimal_clustering)


# tests with Graph
# ... with dicts as attr and spatially_extensive_attr
def test_graph_dict_multi_attr(method):
    cluster_object = PRegionsExact()
    cluster_object.fit_from_networkx(graph, double_attr_dict, n_regions=2,
                                     method=method)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# ... with strings as attr and spatially_extensive_attr
def test_graph_str_multi_attr(method):
    nx.set_node_attributes(graph, attr_str, attr_dict)
    cluster_object = PRegionsExact()
    cluster_object.fit_from_networkx(graph, double_attr_str, n_regions=2,
                                     method=method)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)


# test with W
def test_w_multi_attr(method):
    cluster_object = PRegionsExact()
    cluster_object.fit_from_w(w, double_attr, n_regions=2, method=method)
    result = region_list_from_array(cluster_object.labels_)
    compare_region_lists(result, optimal_clustering)
