from scipy.sparse import kron
from pysal.weights import W 
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
    Wd = Wd.sparse
    Ww = kron(Wo, Wd)
    Ww = w.WSP2W(w.WSP(Ww))
    Ww.transform = transform
    return Ww

def netW(link_list, share='A'):
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
       
    Returns
    -------
     W          : nodal contiguity W object for networkd edges or flows
                  W Object representing the binary adjacency of the network edges
                  given a definition of nodal relationships.

    Examples
    --------
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
    return W(neighbors)

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
