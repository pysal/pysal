import networkx as nx
import libpysal.api as ps_api


def compare_region_lists(actual, desired):
    """
    Parameters
    ----------
    actual : list
        Every element (of type ``set``) represents a region.
    desired : list
        Every element (of type ``set``) represents a region.

    Raises
    ------
    AssertionError
        If the two arguments don't represent the same clustering.
    """
    # check number of regions
    assert len(actual) == len(desired)
    # check equality of regions
    assert all(region in desired for region in actual)


def region_list_from_array(labels):
    """

    Parameters
    ----------
    labels : :class:`numpy.ndarray`
        One-dimensional array of the areas' region labels.

    Returns
    -------
    region_list : `list`
        Each element is a `set` of areas belonging to the same region. The
        list's elements (the sets) are sorted by their smallest entry.

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array([0, 0, 0,
    ...                    1, 1, 0,
    ...                    1, 1, 1])
    >>> desired = [{0, 1, 2, 5}, {3, 4, 6, 7, 8}]
    >>> all(r1 == r2
    ...     for r1, r2 in zip(region_list_from_array(labels), desired))
    True
    """
    region_idx = {}  # mapping from region-ID to index in the returned list
    current_region_idx = 0
    region_list = []
    for area, region in enumerate(labels):
        if region not in region_idx:
            region_idx[region] = current_region_idx
            current_region_idx += 1
            region_list.append(set())
        region_list[region_idx[region]].add(area)
    return region_list


def convert_from_geodataframe(gdf):
    """
    Convert a GeoDataFrame to other types representing the contiguity relation
    of the GeoDataFrame's areas.

    Parameters
    ----------
    gdf : GeoDataFrame

    Returns
    -------
    other_formats : tuple
        The 1st entry is a sparse adjacency matrix of type
        :class:`scipy.sparse.csr_matrix`.
        The 2nd entry is a networkx graph.
        The 3rd entry is a dict. Each key is an area and each value is an
        iterable of the key area's neighbors.
        The 4th entry is a PySAL W object.
    """
    w = ps_api.Rook.from_dataframe(gdf)
    graph = w.to_networkx()
    adj = nx.to_scipy_sparse_matrix(graph)
    neighbor_dict = w.neighbors
    return adj, graph, neighbor_dict, w