"""
Utility functions for graph-related operations with sparse adjacency matrices
(scipy.sparse.csr_matrix) using and supplementing scipy's
[compressed sparse graph routines](
https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html).
"""
from scipy.sparse import csgraph as csg
from scipy.sparse import csr_matrix
import numpy as np


def is_connected(adj):
    """
    Parameters
    ----------
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix.

    Returns
    -------
    connected : `bool`
        `True` if graph defined by adjecency matrix `adj` is connected.
        `False` otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> connected = csr_matrix(np.array([[0, 1],
    ...                                  [1, 0]]))
    >>> is_connected(connected)
    True
    >>> disconnected = csr_matrix(np.array([[0, 0],
    ...                                     [0, 0]]))
    >>> is_connected(disconnected)
    False
    """
    n_connected_components = csg.connected_components(adj, directed=False,
                                                      return_labels=False)
    return True if n_connected_components == 1 else False


def neighbors(adj, area):
    """
    Parameters
    ----------
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix.
    area : int
        An area specified as index in the graph represented by the adjacency
        matrix `adj`. The integer must be in `{0, 1, ..., adj.shape[0]-1}`.

    Returns
    -------
    neighs : :class:`numpy.ndarray`
        The neighbors of `area`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> adjacency_matrix = csr_matrix(np.array([[0, 1, 1],
    ...                                         [1, 0, 0],
    ...                                         [1, 0, 0]]))
    >>> (neighbors(adjacency_matrix, 0) == np.array([1, 2])).all()
    True
    >>> (neighbors(adjacency_matrix, 1) == np.array([0])).all()
    True
    """
    return adj[area].nonzero()[1]


def sub_adj_matrix(adj, nodes, wo_nodes=None):
    """
    Parameters
    ----------
    adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix.
    nodes : :class:`numpy.ndarray`
        1-dimensional array of nodes in the graph represented by the `adj`
        argument. The elements in `nodes` are (distinct) integers in
        `{0, 1, ..., nodes.shape[0]}`.
    wo_nodes : slice, int, array of ints, or None
        Nodes to neglect from the `nodes` argument.

    Returns
    -------
    sub_adj : :class:`scipy.sparse.csr_matrix`
        Adjacency matrix of the subgraph consisting of only the nodes in the
        `nodes` argument.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> adjacency_matrix = csr_matrix(np.array([[0, 1, 1],
    ...                                         [1, 0, 0],
    ...                                         [1, 0, 0]]))
    >>> nodes = np.array([0, 2])
    >>> obtained = sub_adj_matrix(adjacency_matrix, nodes)
    >>> desired = np.array([[0, 1],
    ...                     [1, 0]])
    >>> (obtained.todense() == desired).all()
    True
    >>> nodes = np.array([1, 2])
    >>> obtained = sub_adj_matrix(adjacency_matrix, nodes)
    >>> desired = np.array([[0, 0],
    ...                     [0, 0]])
    >>> (obtained.todense() == desired).all()
    True
    >>> type(obtained) == csr_matrix
    True
    >>> # tests for `wo_nodes` argument
    >>> all_nodes = np.arange(adjacency_matrix.shape[0])
    >>> neglected_nodes = np.array([1])
    >>> obtained = sub_adj_matrix(adjacency_matrix, all_nodes, neglected_nodes)
    >>> desired = np.array([[0, 1],
    ...                     [1, 0]])
    >>> (obtained.todense() == desired).all()
    True
    """
    if wo_nodes is not None:
        mask = np.in1d(nodes, wo_nodes, invert=True)
        nodes = nodes[mask]
    nodes = nodes[:, None]
    return csr_matrix(adj[nodes, nodes.T])

