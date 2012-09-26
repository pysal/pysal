"""
Wing-edge Data Structure for Networks

"""

__author__ = "Sergio J. Rey <srey@asu.edu>"



def get_regions(graph):
    regions = nx.cycle_basis(G)
    for region in regions:
        region.append(region[0])
    regions.insert(0,())
    return regions

def pcw(coords):
    """ test if polygon coordinates are clockwise ordered """
    n = len(coords)
    xl = coords[0:n-1,0]
    yl = coords[1:,1]
    xr = coords[1:,0]
    yr = coords[0:n-1,1]
    a = xl*yl - xr*yr
    area = a.sum()
    if area < 0:
        return 1
    else:
        return 0
     
class WingEdge(object):
    """Data Structure for Networks
    
    Parameters
    ----------

    G: networkx graph

    P: nx2 array of coordinates for nodes in G


    Notes
    -----

    This implementation follows the description in Okabe and Sugihara (2012)
    "Spatial Analysis Along Networks: Statistical and Computational Methods."
    Wiley. Details are from Section 3.1.2.

    As the start_c, end_c, start_cc, and end_cc pointers are only vaguely
    described, the logic from
    http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/model/winged-e.html is
    used to implement these points
    
    """
    def __init__(self, G, P):
        super(WingEdge, self).__init__()
        self.G = G 
        self.P = P
        self.regions = get_regions(G)

        self.node_link = {}     # key: node, value: incident link (edge)
        self.region_link = {}   # key: region (face), value: incident link (edge) 
        self.start_node = {}    # key: link (edge), value: start node 
        self.end_node = {}      # key: link (edge), value: end node 
        self.right_region = {}  # key: link (edge), value: region (face)
        self.left_region = {}   # key: link (edge), value: region (face)
        self.start_c_link = {}  # key: link, value: first incident cw link 
        self.start_cc_link = {} # key: link, value: first incident ccw link 
        self.end_c_link = {}    # key: link, value: first incident c link (end node)
        self.end_cc_link = {}   # key: link, value: first incident ccw link (end node)

        edges = G.edges()
        for edge in edges:
            o,d = edge
            if not o in self.node_link:
                self.node_link[o] = edge

            if not edge in self.start_node:
                self.start_node[edge] = o

            if not edge in self.end_node:
                self.end_node[edge] = d


        for r, region in enumerate(self.regions):
            if not r in self.region_link and r > 0:
                self.region_link[r] = (region[0], region[1])
            if r > 0:
                rcw = pcw(self.P[region[:-1],:])
                for i in xrange(len(region)-1):
                    o,d = region[i:i+2]
                    if rcw:
                        self.right_region[(o,d)] = r
                        self.left_region[(d,o)] = r
                    else:
                        self.left_region[(o,d)] = r
                        self.right_region[(d,o)] = r

        # now for external face
        G = self.G.to_directed()

        missing = [ edge for edge in G.edges() if edge not in self.left_region]
        for edge in missing:
            self.left_region[edge] = 0

        missing = [ edge for edge in G.edges() if edge not in self.right_region]
        for edge in missing:
            self.right_region[edge] = 0
        
        # ccw and cw links
        for edge in self.left_region:
            left_r = self.left_region[edge]
            right_r = self.right_region[edge]
            self.start_c_link[edge] = None
            self.start_cc_link[edge] = None
            self.end_c_link[edge] = None
            self.end_cc_link[edge] = None
            if left_r > 0:
                region = self.regions[left_r]
                n = len(region)
                o = region.index(edge[0])
                d = region.index(edge[1])
                # predecessor
                pred = None
                nxt = None
                if o == 0:
                    pred = (region[-2], region[-1])
                    nxt = (region[o+1], region[o+2])

                if o == n-2:
                    nxt = (region[0], region[1])
                    pred = (region[o-1], region[o])

                if o > 0 and o < n-2:
                    nxt = (region[o+1], region[o+2])
                    pred = (region[o-1], region[o])

                self.start_c_link[edge] = pred
                self.start_cc_link[edge] = nxt

            if right_r > 0:
                region = self.regions[right_r]
                n = len(region)
                o = region.index(edge[0])
                d = region.index(edge[1])
                # predecessor
                pred = None
                nxt = None
                if o == 0:
                    pred = (region[-2], region[-1])
                    nxt = (region[o+1], region[o+2])

                if o == n-2:
                    nxt = (region[0], region[1])
                    pred = (region[o-1], region[o])

                if o > 0 and o < n-2:
                    nxt = (region[o+1], region[o+2])
                    pred = (region[o-1], region[o])

                self.end_c_link[edge] = pred
                self.end_cc_link[edge] = nxt


if __name__ == '__main__':

    # example from figure 3.19 of okabe
    # ommiting 1-3 link to ensure planarity

    import networkx as nx
    import numpy as np

    V = [ (0,0), (1,1), (2,0), (1,-1), (2,-1) ] 
    E = [ (1,2,3), (0,2), (1,0,3), (0,2,4), (3,) ]

    P = np.array(V)

    G = nx.Graph()
    for i,d in enumerate(E):
        for j in d:
            G.add_edge(i,j)

    wed = WingEdge(G,P)

