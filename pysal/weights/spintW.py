"""
Spatial weights for spatial interaction including contiguity OD weights (ODW),
network based weights (netW), and distance-decay based vector weights (vecW).

"""

__author__ = "Taylor Oshan  <tayoshan@gmail.com> "

from scipy.sparse import kron
from pysal.weights import W, WSP
from pysal.weights.util import WSP2W
from pysal.weights.Distance import DistanceBand
from collections import OrderedDict

def ODW(Wo, Wd, transform='r'):
    """
    Constructs an o*d by o*d origin-destination style spatial weight for o*d
    flows using standard spatial weights on o origins and d destinations. Input
    spatial weights must be binary or able to be sutiably transformed to binary.

    Parameters
    ----------
    Wo          : W object for origin locations
                  o x o spatial weight object amongst o origins

    Wd          : W object for destination locations
                  d x d spatial weight object amongst d destinations

    transform   : Transformation for standardization of final OD spatial weight; default
                  is 'r' for row standardized
    Returns
    -------
    W           : spatial contiguity W object for assocations between flows
                 o*d x o*d spatial weight object amongst o*d flows between o
                 origins and d destinations
    
    Examples
    --------

    >>> O = pysal.weights.lat2W(2,2)
    >>> D = pysal.weights.lat2W(2,2)
    >>> OD = pysal.weights.spintW.ODW(O,D)
    >>> OD.weights[0]
    [0.25, 0.25, 0.25, 0.25]
    >>> OD.neighbors[0]
    array([ 5,  6,  9, 10], dtype=int32)
    >>> OD.full()[0][0]
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.25,  0.  ,  0.  ,
            0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ])

    """
    if Wo.transform is not 'b':
        try:
    	    Wo.tranform = 'b'
        except:
            raise AttributeError('Wo is not binary and cannot be transformed to '
                    'binary. Wo must be binary or suitably transformed to binary.')
    if Wd.transform is not 'b':
        try:
    	    Wd.tranform = 'b'
        except:
            raise AttributeError('Wd is not binary and cannot be transformed to '
                   'binary. Wd must be binary or suitably transformed to binary.')
    Wo = Wo.sparse
    Wo.eliminate_zeros()
    Wd = Wd.sparse
    Wd.eliminate_zeros()
    Ww = kron(Wo, Wd, format='csr')
    Ww.eliminate_zeros()
    Ww = WSP2W(WSP(Ww))
    Ww.transform = transform
    return Ww

def netW(link_list, share='A', transform = 'r'):
    """
    Create a network-contiguity based weight object based on different nodal
    relationships encoded in a network.

    Parameters
    ----------
    link_list   : list 
                  of tuples where each tuple is of the form (o,d) where o is an
                  origin id and d is a destination id

    share       : string
                  denoting how to define the nodal relationship used to
                  determine neighboring edges; defualt is 'A' for any shared
                  nodes between two network edges; options include:
                    'A': any shared nodes 
                    'O': a shared origin node
                    'D': a shared destination node
                    'OD' a shared origin node or a shared destination node
                    'C': a shared node that is the destination of the first
                         edge and the origin of the second edge - i.e., a
                         directed chain is formed moving from edge one to edge
                         two.

    transform   : Transformation for standardization of final OD spatial weight; default
                  is 'r' for row standardized
       
    Returns
    -------
     W          : nodal contiguity W object for networkd edges or flows
                  W Object representing the binary adjacency of the network edges
                  given a definition of nodal relationships.

    Examples
    --------

    >>> links = [('a','b'), ('a','c'), ('a','d'), ('c','d'), ('c', 'b'), ('c','a')]
    >>> O = pysal.weights.spintW.netW(links, share='O')
    >>> W.neighbors[('a', 'b')]
    [('a', 'c'), ('a', 'd')]
    >>> OD = pysal.weights.spintW.netW(links, share='OD')
    >>> OD.neighbors[('a', 'b')]
    [('a', 'c'), ('a', 'd'), ('c', 'b')]
    >>> any_common = pysal.weights.spintW.netw(links, share='A')
    [('a', 'c'), ('a', 'd'), ('c', 'b'), ('c', 'a')]

    """
    neighbors = {}
    neighbors = OrderedDict()
    edges = link_list
    for key in edges:
        neighbors[key] = []
        for neigh in edges:
            if key == neigh:
                continue
            if share.upper() == 'OD':
                if key[0] == neigh[0] or key[1] == neigh[1]:
                    neighbors[key].append(neigh)
            elif share.upper() == 'O':
                if key[0] == neigh[0]:
                    neighbors[key].append(neigh)
            elif share.upper() == 'D':
                if key[1] == neigh[1]:
                	neighbors[key].append(neigh)
            elif share.upper() == 'C':
                if key[1] == neigh[0]:
                    neighbors[key].append(neigh)
            elif share.upper() == 'A':
                if key[0] == neigh[0] or key[0] == neigh[1] or \
                	key[1] == neigh[0] or key[1] == neigh[1]:
                    neighbors[key].append(neigh)
            else:
                raise AttributeError("Parameter 'share' must be 'O', 'D',"
                       " 'OD', or 'C'")
    netW = W(neighbors)
    netW.tranform = transform
    return netW

def vecW(origin_x, origin_y, dest_x, dest_y, threshold, p=2, alpha=-1.0,
        binary=True, ids=None, build_sp=False, silent=False):
    """
    Distance-based spatial weight for vectors that is computed using a
    4-dimensional distance between the origin x,y-coordinates and the
    destination x,y-coordinates

    Parameters
    ----------
    origin_x   : list or array
                 of vector origin x-coordinates
    origin_y   : list or array
                 of vector origin y-coordinates
    dest_x     : list or array
                 of vector destination x-coordinates
    dest_y     : list or array
                 of vector destination y-coordinates
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    binary     : boolean
                 If true w_{ij}=1 if d_{i,j}<=threshold, otherwise w_{i,j}=0
                 If false wij=dij^{alpha}
    alpha      : float
                 distance decay parameter for weight (default -1.0)
                 if alpha is positive the weights will not decline with
                 distance. If binary is True, alpha is ignored

    ids         : list
                  values to use for keys of the neighbors and weights dicts
    build_sp    : boolean
                  True to build sparse distance matrix and false to build dense
                  distance matrix; significant speed gains may be obtained
                  dending on the sparsity of the of distance_matrix and
                  threshold that is applied
    silent      : boolean
                  By default PySAL will print a warning if the
                  dataset contains any disconnected observations or
                  islands. To silence this warning set this
                  parameter to True.
    
    Returns
    ------
    W           : DistanceBand W object that uses 4-dimenional distances between
                  vectors origin and destination coordinates. 

    Examples
    --------

    >>> x1 = [5,6,3]
    >>> y1 = [1,8,5]
    >>> x2 = [2,4,9]
    >>> y2 = [3,6,1]
    >>> W1 = pysal.weights.spintW.vecW(x1, y1, x2, y2, threshold=999)
    >>> W1.neighbors[0]
    [1, 2]
    >>> W2 = pysal.weights.spintW.vecW(x1, y2, x1, y2, threshold=8.5)
    >>> W2.neighbors[0]
    [1]

    """
    data = zip(origin_x, origin_y, dest_x, dest_y)
    W = DistanceBand(data, threshold=threshold, p=p, binary=binary, alpha=alpha,
            ids=ids, build_sp=False, silent=silent)
    return W

def mat2L(edge_matrix):
    """
    Convert a matrix denoting network connectivity (edges or flows) to a list
    denoting edges

    Parameters
    ----------
    edge_matrix   : array 
                    where rows denote network edge origins, columns denote
                    network edge destinations, and non-zero entries denote the
                    existence of an edge between a given origin and destination

    Returns
    -------
     edge_list    : list
                    of tuples where each tuple is of the form (o,d) where o is an
                    origin id and d is a destination id

    Examples
    --------
    """
    if len(edge_matrix.shape) !=2:
    	raise AttributeError("Matrix of network edges should be two dimensions"
    	        "with edge origins on one axis and edge destinations on the"
    	        "second axis with non-zero matrix entires denoting an edge"
    	        "between and origin and destination")
    edge_list = []
    rows, cols = edge_matrix.shape
    for row in range(rows):
        for col in range(cols):
            if edge_matrix[row, col] != 0:
                edge_list.append((row,col))
    return edge_list
