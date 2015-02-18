import numpy as np
import operator as op

def ntw_from_w(path, W = None):
    """
    Construction of the dual network from a lattice shapefile
    
    Parameters
    ----------
    
    path : string
           path to shapefile 
    
    Returns
    -------
    ntw : network
          a pysal.network.network.Network object
    """

    ntw = ps.network.network.Network()

    #read shapefile stuff
    ntw.in_shp = shp
    shp = ps.open(shp)
    
    if 'rook' in W:
        W = ps.rook_from_shapefile
    elif not W or 'queen' in W:
        W = ps.rook_from_shapefile
  
    #set node_list and edges using the weights
    ed = set()
    ntw.node_list = []
    for poly,neighbs in W.neighbors.iteritems():
        tl = [(p, n) for p,n in zip([poly]*len(neighbs), neighbs) ]
        ed.update(tl)
        ntw.node_list.append(poly) #add key to the node_list
    ed = list(ed)
    ed.sort(key=op.itemgetter(0)) #sort on first element of tup
    ntw.edges = ed
    
    #set nodes and node_coords if shp exists
    #there's some index problems here:
    #weights usu. 0-index but shps usu. 1-index
    #so, for now, any call against shps will decrement index

    ntw.nodes = {poly.centroid: poly.id-1 for poly in shp}

    ntw.node_coords = {poly.id-1: poly.centroid for poly in shp}#invert!
    
    ntw.edge_lengths = {edge: np.linalg.norm((ntw.node_coords[edge[0]],\
        ntw.node_coords[edge[1]])) for edge in ntw.edges}

    #no pp's
    ntw.pointpatterns = None

    #this is just weights
    ntw.adjacencylist = W.neighbors
    
    ntw.extractgraph()   
    return ntw
