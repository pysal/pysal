"""
Winged-edge Data Structure for Networks

"""

__author__ = "Sergio J. Rey <srey@asu.edu>"

from shapely.ops import polygonize
import pysal as ps






def regions_from_graph(vertices, edges):
    """
    Extract regions from vertices and edges of a planar graph

    Arguments
    ---------

    vertices: dictionary with vertex id as key, coordinates of vertex as value

    edges: list of (head,tail) edges

    Returns
    ------

    regions: list of lists of nodes defining a region. Includes the external
    region



    Examples
    --------
    >>> vertices = {1: (1,2), 2:(0,1), 3:(2,1), 4:(0,0), 5:(2,0)}
    >>> edges = [ (1,2), (1,3), (2,3), (2,4), (4,5), (5,3) ]
    >>> r = regions_from_graph(vertices, edges)
    >>> r
    [[1, 2, 3, 1], [1, 3, 5, 4, 2, 1], [2, 4, 5, 3, 2]]


    Notes
    -----

    Based on Jiang, X.Y. and H. Bunke (1993) "An optimal algorithm for
    extracting the regions of a plane graph." Pattern Recognition Letters,
    14:533-558.
    """

    # step 1
    # have a twin for each directed edge
    dedges = edges[:]
    for edge in edges:
        new_edge = edge[1], edge[0]
        if new_edge not in dedges:
            dedges.append( (edge[1],edge[0]) )

    # step 2 complement each directed edge with an angle formed with horizontal
    # line passing through edge[0] for each edge
    angles = []
    from math import atan2, degrees

    for edge in dedges:

        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        at = atan2(dy, dx)
        d = degrees(at)
        if d < 0:
            d = 360 + d
        angles.append( [ (edge[0],d), (edge[0],edge[1]) ])

    # step 3 sort the list into ascending order using vi and angle as primary and
    # secondary keys
    angles.sort()


    # form wedges on consecutive entries with same vi (vi,vj,dij), (vi,vk,dik)
    # gives the wedge (vk,vi,vj)
    wedges = []
    start = angles[0]
    c = 0
    for i in range(1,len(angles)):
        next_edge = angles[i]
        previous_edge = angles[i-1]
        if next_edge[0][0] == start[0][0]:
            wedge = [ next_edge[1][1], previous_edge[1][0], previous_edge[1][1] ]
            wedges.append(wedge)
        else:
            # first form wedge with last and first entry of current group
            # to do
            wedge = [ start[1][1], previous_edge[1][0], previous_edge[1][1] ]
            wedges.append(wedge)
            start = next_edge

    # final pair

    wedge = [ start[1][1], previous_edge[1][0], next_edge[1][1] ]
    wedges.append(wedge)


    # phase two
    # form regions from contiguous wedges

    nw = len(wedges)
    used = [0]*nw
    wedges.sort()
    #print wedges

    #print 'forming regions'

    i = 0
    regions = []
    while sum(used) < nw:
        i = used.index(0)
        wi = wedges[i]
        start = wedges[i]
        used[i] = 1
        region = [start]
        # find next contiguous wedge for wi
        forming = True
        while forming:


            # find first wedge contiguous to wi
            for j in xrange(nw):
                wj = wedges[j]
                if wj[0] ==  wi[1] and wj[1] == wi[2]:
                    region.append(wj)
                    used[j] = 1
                    wi = wj
                    if wi[1] == start[0] and wi[2] == start[1]:
                        forming = False
                        regions.append(region)
                        #print start, regions
                        #raw_input('h')
                    break

    # put in closed cartographic form
    nodes = []
    for region in regions:
        wedge0 = [ wedge[0] for wedge in region]
        wedge0.append(wedge0[0])
        nodes.append(wedge0)

    return nodes




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

        
class Vertex(object):
    """Vertex for Winged Edge Data Structure"""
    def __init__(self, x,y, edge=None):
        super(Vertex, self).__init__()
        self.x= x
        self.y =y
        self.edge = edge # one incident edge for the vertex
    def __str__(self):
        return "(%f, %f)"%(self.x, self.y)

class Edge(object):
    """Edge for Winged Edge Data Structure"""
    def __init__(self, startV, endV, left=None, right=None,
                pl=None, sl=None, pr=None, sr=None, name=None):
        super(Edge, self).__init__()

        self.start = startV  # start vertex
        self.end = endV      # end vertex
        self.left = left     # left face
        self.right = right   # right face
        self.pl = pl         # preceding edge for cw traversal of left face
        self.sl = sl         # successor edge for cw traversal of left face
        self.pr = pr         # preceding edge for cw traversal of right face
        self.sr = sr         # successor edge for cw traversal of right face 
        self.name = name

    def __str__(self):
        if not self.name:
            self.name = 'Edge'
        return "%s: (%f,%f)--(%f,%f)"%(self.name, self.start.x, self.start.y, self.end.x,
                self.end.y) 
        
class Face(object):
    """Face for Winged Edge Data Structure"""
    def __init__(self, nodes, edge=None):
        super(Face, self).__init__()
        self.nodes = nodes # nodes/vertices defining face
        self.edge = edge # one incident edge for the face

        if self.nodes[0] != self.nodes[-1]:
            self.nodes.append(self.nodes[0]) # put in closed form

    def __str__(self):
        n = len(self.nodes)
        nv = [ "(%f,%f)"%(v.x,v.y) for v in self.nodes]
        return "--".join(nv)

def face_boundary(face):
    """Clockwise traversal around face edges

    Arguments
    --------
    face: face instance

    Returns
    ------
    edges: list of edges on face boundary ordered clockwise

    """
    l0 = face.edge
    l = l0
    edges = []
    traversing = True
    while traversing:
        edges.append(l)
        r = l.right
        if r == face:
            l = l.sr
        else:
            l = l.sl
        if l == l0:
            traversing = False
    return edges


def incident_cw_edges_node(node):
    """Clockwise traversal of edges incident with node

    Arguments
    ---------

    node: Vertex instance

    Returns
    ------
    edges: list of edges that are cw incident with node

    """
    l0 = node.edge
    l = l0
    edges = []
    traversing = True
    while traversing:
        edges.append(l)
        v = l.start
        if v == node:
            l = l.pr
        else:
            l = l.pl
        #print l0, l
        if l0 == l:
            traversing = False
        #raw_input('here')
    return edges


def _lat2Network(k):
    """helper function to create a network from a square lattice.
    
    Used for testing purposes 
    """
    lat = ps.lat2W(k+1,k+1) 
    k1 = k+1
    nodes = {}
    edges = []
    for node in lat.id_order:
        for neighbor in lat[node]:
            edges.append((node,neighbor))
        nodes[node] = ( node/k1, node%k1 )

    res = {"nodes": nodes, "edges": edges}

    return res

class NPWED(object):
    """Winged edge data structure for Nonplanar network"""
    def __init__(self, G, P):
        super(NPWED, self).__init__()
        self.G = G
        self.P = P
        self.node_link = {}     # key: node, value: incident link (edge)
        self.start_node = {}    # key: link (edge), value: start node 
        self.end_node = {}      # key: link (edge), value: end node 
        self.start_c_link = {}  # key: link, value: first incident cw link 
        self.start_cc_link = {} # key: link, value: first incident ccw link 
        self.end_c_link = {}    # key: link, value: first incident cw link (end node)
        self.end_cc_link = {}   # key: link, value: first incident ccw link (end node)

        for edge in self.G.edges():
            s,e = edge
            self.node_link[s] = edge
            self.start_node[edge] = s
            self.end_node[edge] = e

    def incident_links(self, node):
        links = []
        links.extend(self.G.out_edges(node))
        links.extend(self.G.in_edges(node))
        return links







        
        
if __name__ == '__main__':

    # example from figure 3.19 of okabe
    # omitting 1-3 link to ensure planarity

    import networkx as nx
    import numpy as np

    V = [ (0,0), (1,1), (2,0), (1,-1), (2,-1) ] 
    E = [ (1,2,3), (0,2), (1,0,3), (0,2,4), (3,) ]

    P = np.array(V)

    G = nx.DiGraph()
    for i,d in enumerate(E):
        for j in d:
            G.add_edge(i,j)


    npwed = NPWED(G,P)

    # Alternative implementation of WED

    """Simple network example

    A a B b C m J
    c 1 d 2 e
    D f E g F
    h 3 i 4 j
    G k H l I

    Where upper case letters are Nodes/Vertices, lower case letters are edges,
    and integers are face ids.
    
    There are four faces 1-4, but one external face 0 (implied)
    """



    vertices = {}
    vertices['A'] = Vertex(0.,2.)
    vertices['B'] = Vertex(1.,2.)
    vertices['C'] = Vertex(2.,2.)
    vertices['D'] = Vertex(0.,1.)
    vertices['E'] = Vertex(1.,1.)
    vertices['F'] = Vertex(2.,1.)
    vertices['G'] = Vertex(0.,0.)
    vertices['H'] = Vertex(1.,0.)
    vertices['I'] = Vertex(2.,0.)
    vertices['J'] = Vertex(3.,2.)



    edges = {}
    edata = [ ('a', 'A', 'B'),
              ('b', 'B', 'C'),
              ('c', 'A', 'D'),
              ('d', 'E', 'B'),
              ('e', 'C', 'F'),
              ('f', 'D', 'E'),
              ('g', 'E', 'F'),
              ('h', 'D', 'G'),
              ('i', 'H', 'E'),
              ('j', 'F', 'I'),
              ('k', 'G', 'H'),
              ('l', 'I', 'H'),
              ('m', 'C', 'J')]

    for edge in edata:
        edges[edge[0]] = Edge(vertices[edge[1]], vertices[edge[2]])

    faces = {}
    fdata = [ ('A', 'B', 'E', 'D'),
              ('B', 'C', 'F', 'E'),
              ('D', 'E', 'H', 'G'),
              ('E', 'F', 'I', 'H') ]

    fe = [ 'c', 'd', 'f', 'j']

    for i, face in enumerate(fdata):
        i+=1
        coords = [ vertices[j] for j in face]
        faces[i] = Face(coords)
        faces[i].edge = edges[fe[i-1]]

    faces[0] = Face([0.,0.,0.0])
    faces[0].edge = edges['h']

    lrdata = [ (0,1),
               (0,2),
               (1,0),
               (1,2),
               (0,2),
               (1,3),
               (2,4),
               (3,0),
               (3,4),
               (0,4),
               (3,0),
               (0,4),
               (0,0)]
    ekeys = edges.keys()
    ekeys.sort()

    for i, lr in enumerate(lrdata):
        edges[ekeys[i]].left = faces[lr[0]]
        edges[ekeys[i]].right = faces[lr[1]]



    psdata = [ #node pre_left successor_left, pre_right, successor_right
            ('a', 'b', 'c', 'c', 'd'),
            ('b', 'm', 'a', 'd', 'e'),
            ('c', 'f', 'a', 'a', 'h'),
            ('d', 'a', 'f', 'g', 'b'),
            ('e', 'j', 'm', 'b', 'g'),
            ('f', 'd', 'c', 'h', 'i'),
            ('g', 'e', 'd', 'i', 'j'),
            ('h', 'k', 'f', 'c', 'k'),
            ('i', 'f', 'k', 'l', 'g'),
            ('j', 'l', 'e', 'g', 'l'),
            ('k', 'i', 'h', 'h', 'l'),
            ('l', 'k', 'j', 'j', 'i'),
            ('m', 'e', 'b', 'e', 'b') ]

    for pdata in psdata:
        e,pl,sl,pr,sr = pdata
        edges[e].pl = edges[pl]
        edges[e].sl = edges[sl]
        edges[e].pr = edges[pr] 
        edges[e].sr = edges[sr]
        edges[e].name = e

    n2e = [
            ('A', 'c'),
            ('B', 'b'),
            ('C', 'e'),
            ('D', 'f'),
            ('E', 'd'),
            ('F', 'g'),
            ('G', 'h'),
            ('H', 'k'),
            ('I', 'j'),
            ('J', 'm')]
    for node in n2e:
        v,e = node
        vertices[v].edge = edges[e]

    cv = 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', "I"
    for v in cv:
        icwe = incident_cw_edges_node(vertices[v])
        print "Node: ", v, "cw incident edges: "
        for e in icwe:
            print e


    for f in range(1,5):
        ecwf = face_boundary(faces[f])
        print "Face: ",f, " cw edges:"
        for e in ecwf:
            print e


    # test region extraction

    vertices = {1: (1,2),
            2:(0,1),
            3:(2,1),
            4:(0,0),
            5:(2,0)}

    edges = [
            (1,2),
            (1,3),
            (2,3),
            (2,4),
            (4,5),
            (5,3) ]

    """
    r = regions_from_graph(vertices,edges)
    """


    network = _lat2Network(3)
    vertices = network['nodes']
    edges = network['edges']
    r1 = regions_from_graph(vertices, edges)


    import doctest
    doctest.testmod()

