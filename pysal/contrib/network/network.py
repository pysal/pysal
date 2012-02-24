import math
import pysal
from pysal.cg.shapes import Point, Chain, LineSegment, Rectangle
from pysal.cg.locators import Grid
import random, copy
from heapq import heappush, heappop
import time

def no_nodes(G):
    """
    returns the number of nodes in a undirected network
    """
    return len(G)

def no_edges(G):
    """
    returns the number of edges in a undirected network
    """
    e = 0.0
    for n in G:
        e += len(G[n])
    return e/2.0 

def tot_net_length(G):
    """
    returns the total length of a undirected network
    """
    l = 0.0
    done = set()
    for n in G:
        for m in G[n]:
            if m in done: continue
            l += G[n][m]
        done.add(n)
    return l/2.0

def walk(G, s, S=set()):
    """ 
    Returns a traversal path from s on G
    source: Python Algorithms Mastering Basic Algorithms in the Python Language, 2010, p.104
    """
    P, Q, SG = dict(), set(), dict()
    P[s] = None
    SG[s] = G[s]
    Q.add(s)
    while Q:
        u = Q.pop()
        for v in set(G[u].keys()).difference(P, S):
            Q.add(v)
            P[v] = u
            SG[v] = G[v]
    return SG

def components(G):
    """ 
    Returns connected components of G
    source: Python Algorithms Mastering Basic Algorithms in the Python Language, 2010, p.105
    Complexity: O(E+V) where E is the number of edges and V is the number of nodes in a graph
    """
    comp, seen = [], set()
    for u in G:
        if u in seen: continue
        C = walk(G, u)
        seen.update(set(C.keys()))
        comp.append(C)
    return comp

def no_components(G):
    return len(components(G))

def net_global_stats(G, boundary=None, detour=True):
    v = no_nodes(G)
    e = no_edges(G)
    L = tot_net_length(G)
    p = no_components(G)
    u = e - v + p # cyclomatic number
    alpha = u*1.0/(2*v - 5)
    beta = e*1.0/v
    emax = 3*(v-2)
    gamma = e*1.0/emax
    eta = L*1.0/e
    net_den = None
    if boundary:
        s = pysal.open(boundary)
        if s.type != pysal.cg.shapes.Polygon: 
            raise ValueError, 'File is not of type POLYGON'
        net_den = s.next().area
    net_dist, eucl_dist = 0.0, 0.0
    det = None
    if detour:
        nodes = G.keys()
        for n in nodes:
            net_D = dijkstras(G, n)
            net_dist += sum(net_D.values())
            eucl_D = [ math.sqrt((n[0] - m[0])**2 + (n[1] - m[1])**2) for m in nodes]
            eucl_dist += sum(eucl_D)
        net_dist /= 2.0
        eucl_dist /= 2.0
        if net_dist > 0.0:
            det = eucl_dist*1.0/net_dist
    return v, e, L, p, u, alpha, beta, emax, gamma, eta, net_den, det


def random_projs(G, n):
    """
    Returns a list of random locations on the network as projections
    with the form (src, dest, dist_from, dist_from_src, dist_from_dest)
    """

    def binary_search(list, q):
        l = 0
        r = len(list)
        while l < r:
            m = (l + r)/2
            if list[m][0] > q:
                r = m
            else:
                l = m + 1
        return list[l][1]

    total_net_len = 0
    for src in G:
        for dest in G[src]:
            total_net_len += G[src][dest]

    lengthogram = [(0, (None, None))]
    for src in G:
        for dest in G[src]:
            lengthogram.append((lengthogram[-1][0] + G[src][dest], (src, dest)))

    projs = []
    for i in xrange(n):
        e = binary_search(lengthogram, random.random() * total_net_len)
        wgt = G[e[0]][e[1]]
        along = wgt * random.random()
        # (src, dest, dist_from_src, dist_from_dest)
        projs.append((e[0], e[1], along, wgt - along))

    return projs

def proj_distances_undirected(G, src, dests, r=1e600, cache=None):
    if cache and src[0] in cache:
        SND = cache[src[0]]
    else:
        SND = dijkstras(G, src[0], r) # Distance from edge start node to other nodes
    if cache and src[1] in cache:
        DND = cache[src[1]]
    else:
        DND = dijkstras(G, src[1], r) # Distance from edge end node to other nodes
    D = {}
    for d in dests:
        # If the dest lies on the same edge as the src (or its inverse)
        if (d[0] == src[0] and d[1] == src[1]) or (d[0] == src[1] and d[1] == src[0]):
                dist = abs(src[2] - d[2])
        else:
            # get the four path distances
            # src edge start to dest edge start
            src2src, src2dest, dest2src, dest2dest = 1e600, 1e600, 1e600, 1e600
            if d[0] in SND: 
                src2src = src[2] + SND[d[0]] + d[2] 
            # src edge start to dest edge end
            if d[1] in SND:
                src2dest = src[2] + SND[d[1]] + d[3] 
            # src edge end to dest edge start
            if d[0] in DND:
                dest2src = src[3] + DND[d[0]] + d[2]
             # src edge end to dest edge end
            if d[1] in DND:
                dest2dest = src[3] + DND[d[1]] + d[3]
            dist = min(src2src, src2dest, dest2src, dest2dest)

        if dist <= r:
            D[d] = dist

    return D

def proj_distances_directed(G, src, dests, r=1e600):
    ND = dijkstras(G, src[1], r) # Distance from edge destination node to other nodes
    D = {}
    for d in dests:
        if d[0] in ND:
            if d[0] == src[0] and d[1] == src[1]: # Same edge and dest further along
                dist = abs(src[2] - d[2])
                if dist <= r:
                    D[d] = dist
            else:
                # dist from edge proj to end of src edge + 
                # dist from src edge dest node to dest edge src node +
                # dist from start of dest edge to edge proj
                dist = src[3] + ND[d[0]] + d[2] 
                #print dist
                if dist <= r:
                    D[d] = dist
    return D

def relax(G, u, v, D, P, r=1e600): 
    """ 
    Update the distance to v 
    if the route to v through u is shorter than the existing route to v
    Code from Hetland 2010 
    Python Algorithms Mastering Basic Algorithms in the Python Language, p.200
    """
    d = D.get(u, r) + G[u][v]
    if d <= D.get(v, r):
        D[v], P[v] = d, u
        return True

def dijkstras(G, start, r=1e600, p=False): 
    """ 
    Find a shortest path from s to all nodes in the network G
    Code from Hetland 2010 
    Python Algorithms Mastering Basic Algorithms in the Python Language, p.205
    Complexity: O(M*lgN) where M is the number of edges and N is the number of nodes
    """
    D, P, Q, S = {start:0}, {}, [(0,start)], set()  # Distance estimates, tree (path), queue, visited
    while Q:
        _, u = heappop(Q)
        if u in S: 
            continue 
        S.add(u)
        for v in G[u]:
            relaxed = relax(G, u, v, D, P, r=r)
            if relaxed: 
                heappush(Q, (D[v], v)) 
    if p: 
        return D, P
    return D

class Snapper:
    """
    Snaps points to their nearest location on the network.

    Uses a novel algorithm which relies on two properties of the input network:
    1.  Most of the edges are very short relative to the total area 
        encompassed by the network.
    2.  The edges have a relatively constant density throughout this area.

    The algorithm works by creating a binning of the midpoints of all the edges.
    When a query point is given, all the edges in a region around the query
    point (located by their binned midpoints) are compared to the query point. 
    If none are found, the neighborhood is enlarged. When a closest edge is found, 
    the neighborhood is enlarged slightly and checked again. The enlargement is 
    such that if the closest edge found remains the closest edge, then it will
    always be the closest edge.
    """

    def __init__(self, network):
        """
        Test tag <tc>#is#Snapper.__init__</tc>
        """

        """
        Generate a list of the edge lengths and pick a maximum length
        allowed based on the median length. This maximum length will be
        used to use multiple midpoints to represent edges which are 
        exceptionally long, lest they ruin the efficiency of the algorithm.
        """

        # Generate list of lengths
        self.network = network
        edge_lens = []
        for n in network:
            for m in network[n]:
                if n != m: 
                    edge_lens.append(pysal.cg.get_points_dist(Point(n), Point(m))) # it can be optional
        if edge_lens == []:
            raise ValueError, 'Network has no positive-length edges'
        edge_lens.sort()
        max_allowed_edge_len = 5 * edge_lens[len(edge_lens)/2]

        """ 
        Create a bin structures with proper range to hold all of the edges.
        The size of the bin is on the order of the length of the longest edge (and
        of the neighborhoods searched around each query point.
        """
        endpoints = network.keys()
        endpoints_start, endpoints_end = [ep[0] for ep in endpoints], [ep[1] for ep in endpoints]
        bounds = Rectangle(min(endpoints_start),min(endpoints_end),max(endpoints_start),max(endpoints_end))
        self.grid = Grid(bounds, max_allowed_edge_len)

        """
        Insert the midpoint of each edge into the grid. If an edge is too long, 
        break it into edges of length less than the maximum allowed length and
        add the midpoint of each.
        """
        self.search_rad = max_allowed_edge_len*0.55
        for n in network:
            for m in network[n]:
                edge_len = pysal.cg.get_points_dist(Point(n), Point(m)) # it can be a direct extraction
                if edge_len > max_allowed_edge_len:
                    mid_edge = []
                    num_parts = int(math.ceil(edge_len/max_allowed_edge_len))
                    part_step = 1.0/num_parts
                    dx = m[0] - n[0]
                    dy = m[1] - n[1]
                    midpoint = (n[0] + dx*part_step/2, n[1] + dy*part_step/2)
                    for r in [part_step*t for t in xrange(num_parts)]:
                        mid_edge.append(((n, m), midpoint))
                        midpoint = (midpoint[0] + dx*part_step, midpoint[1] + dy*part_step)
                    for me in mid_edge:
                        self.grid.add(me[0], Point(me[1]))
                else:
                    self.grid.add((n, m), Point(((n[0] + m[0])/2, (n[1] + m[1])/2)))

        """
        During the snapping of a query point we will initialize the closest point on the network
        to be a dummy location known to be invalid. This must be done in case the neighborhood
        search does not find any edge midpoints and it must be grown repeatedly. In this case
        we want to make sure we don't give up having not found a valid closest edge.
        """
        self.dummy_proj = (None, None, 0, 0) # Src, dest, dist_from_src, dist_from_dest)

    def snap(self, p):
        """
        Test tag <tc>#is#Snapper.snap</tc>
        """

        """
        Initialize the closest location found so far to be infinitely far away.
        Then begin with a neighborhood on the order of the maximum edge allowed and
        repeatedly growing it. When a closest edge is found, grow once more and check again.
        """
        
        cur_s_rad = self.search_rad
        found_something = False
        # Whle neighborhood is empty, enlarge and check again    
        while not found_something: 
            if self.grid.proximity(Point(p), cur_s_rad) != []:
                found_something = True
            cur_s_rad *= 2
        # Expand to include any edges whose endpoints might lie just outside
        # the search radius
        cur_s_rad += self.search_rad
        # Now find closest in this neighborhood
        best_seg_dist = 1e600
        for e in self.grid.proximity(Point(p), cur_s_rad):
            seg = LineSegment(Point(e[0]), Point(e[1]))
            p2seg = pysal.cg.get_segment_point_dist(seg, Point(p))
            dist = p2seg[0]
            if p2seg[0] < best_seg_dist:
                # (src, dest, dist_from_src, dist_from_dest)
                best_proj = (e[0], e[1], dist*p2seg[1], dist*(1-p2seg[1]))
                best_seg_dist = p2seg[0]
        return best_proj

def network_from_endnodes(s, d, wgt, undirected=True):
    G = {}
    for g, r in zip(s,d):
        start = g.vertices[0]
        end = g.vertices[-1]
        G.setdefault(start, {})
        G.setdefault(end, {})
        r_w = wgt(g,r)
        G[start][end] = r_w
        if undirected:
            G[end][start] = r_w
    s.close()
    d.close()
    return G

def network_from_allvertices(s, d):
    G = {}
    for g, r in zip(s, d):
        vertices = g.vertices
        for i, vertex in enumerate(vertices[:-1]):
            n1, n2 = vertex, vertices[i+1]
            dist = pysal.cg.get_points_dist(Point(n1), Point(n2)) 
            G.setdefault(n1, {}) 
            G.setdefault(n2, {}) 
            G[n1][n2] = dist 
            G[n2][n1] = dist 
    s.close()
    d.close()
    return G

def read_hierarchical_network(s, d):
    G, Gj, G_to_Gj = {}, {}, {}
    for g, r in zip(s, d):
        vertices = g.vertices
        Gj.setdefault(vertices[0], {}) 
        Gj.setdefault(vertices[-1], {}) 
        d_total = 0.0 
        for i, vertex in enumerate(vertices[:-1]):
            n1, n2 = vertex, vertices[i+1]
            dist = pysal.cg.get_points_dist(Point(n1), Point(n2)) 
            G.setdefault(n1, {}) 
            G.setdefault(n2, {}) 
            G[n1][n2] = dist 
            G[n2][n1] = dist
            G_to_Gj[(n1,n2)] = [(vertices[0], vertices[-1]), d_total] # info for the opposite direction 
            d_total += dist
        Gj[vertices[0]][vertices[-1]] = d_total
        Gj[vertices[-1]][vertices[0]] = d_total
    s.close()
    d.close()
    return G, Gj, G_to_Gj

def read_network(filename, wgt_field=None, undirected=True, endnodes=False, hierarchical=False, attrs=None):
    s = pysal.open(filename)
    dbf = pysal.open(filename[:-3] + 'dbf')
    if s.type != pysal.cg.shapes.Chain:
        raise ValueError, 'File is not of type ARC'
    if not endnodes and not undirected:
        raise ValueError, 'Network using all vertices should be undirected'
    if hierarchical and not (undirected and not endnodes):
        raise ValueError, 'Hierarchial network should be undirected and use all vertices'
    if endnodes:
        if wgt_field and attrs == None:
            w = dbf.header.index(wgt_field)
            def wgt(g, r):
                return r[w]
        elif wgt_field and attrs:
            attrs = [wgt_field] + attrs
            w_indices = [dbf.header.index(field) for field in attrs]
            def wgt(g, r):
                return [r[w] for w in w_indices]
        elif wgt_field is None and attrs:
            w_indices = [dbf.header.index(field) for field in attrs]
            def wgt(g, r):
                d = pysal.cg.get_points_dist(Point(g.vertices[0]), Point(g.vertices[-1]))
                return [d] + [r[w] for w in w_indices] 
        else:
            def wgt(g, r):
                return pysal.cg.get_points_dist(Point(g.vertices[0]), Point(g.vertices[-1]))
        return network_from_endnodes(s, dbf, wgt, undirected)
    if not endnodes and not hierarchical:
        return network_from_allvertices(s, dbf)
    if hierarchical:
        return read_hierarchial_netowrk(s, dbf)

def proj_pnt_coor(proj_pnt):
    n1, n2 = proj_pnt[0], proj_pnt[1]
    dist_n12 = pysal.cg.get_points_dist(Point(n1), Point(n2))
    len_ratio = proj_pnt[2]*1.0/dist_n12
    xrange, yrange = n2[0] - n1[0], n2[1] - n1[1]
    x = n1[0] + xrange*len_ratio
    y = n1[1] + yrange*len_ratio
    return (x,y)

def inject_points(network, proj_pnts):

    pnts_by_seg = {}
    proj_pnt_coors = []
    for pnt in proj_pnts:
        target_edge = None
        if (pnt[0], pnt[1]) not in pnts_by_seg and (pnt[1], pnt[0]) not in pnts_by_seg:
            target_edge = (pnt[0], pnt[1])
            pnts_by_seg[target_edge] = set()
        elif (pnt[0], pnt[1]) in pnts_by_seg:
            target_edge = (pnt[0], pnt[1])
        elif (pnt[1], pnt[0]) in pnts_by_seg:
            target_edge = (pnt[1], pnt[0])
        coor = proj_pnt_coor(pnt)
        pnts_by_seg[target_edge].add(coor)
        proj_pnt_coors.append(coor)

    new_network = copy.deepcopy(network)
    
    for seg in pnts_by_seg:
        proj_nodes = set(list(pnts_by_seg[seg]) + [seg[0], seg[1]])
        proj_nodes = list(proj_nodes)
        if seg[0][0] == seg[1][0]:
            proj_nodes.sort(key=lambda coords: coords[1])
        else:
            proj_nodes.sort()
        proj_nodes_len = len(proj_nodes)
        prev_seg_d, next_seg_d = 0.0, 0.0
        for i in range(proj_nodes_len - 1):
            start, end = proj_nodes[i], proj_nodes[i+1]
            if start not in new_network:
                new_network[start] = {}
            if end not in new_network:
                new_network[end] = {}
            d = pysal.cg.get_points_dist(Point(start), Point(end))
            new_network[start][end] = d
            new_network[end][start] = d
        if new_network.has_key(seg[0]) and new_network[seg[0]].has_key(seg[1]):
            del new_network[seg[0]][seg[1]]
            del new_network[seg[1]][seg[0]]
        else:
            print seg, network.has_key(seg[0]), network[seg[0]], network.has_key(seg[1]), network[seg[1]]

    return new_network, proj_pnt_coors

def mesh_network(network, cellwidth, at_center=False):
    mesh_net = {}
    done = {}
    #done = set()
    for n1 in network:
        for n2 in network[n1]:
            #if n2 in done: continue
            if (n1,n2) in done or (n2,n1) in done:
                continue
            len_ratio = cellwidth*1.0/network[n1][n2]
            start, end = n1, n2
            # The order for reading a network edge is slightly different from SANET. 
            # SANET does not seem to have a set of consistent rules. 
            if n1[0] < n2[0] or (n1[0] == n2[0] and n1[1] < n2[1]):
                start, end = n2, n1
            xrange, yrange = end[0] - start[0], end[1] - start[1]
            dx, dy = xrange*len_ratio, yrange*len_ratio
            no_segments = int(math.floor(1.0/len_ratio))
            if at_center:
                xs = [start[0], start[0] + dx/2.0]
                ys = [start[1], start[1] + dy/2.0]
                xs = xs + [xs[-1] + i*dx for i in range(1, no_segments + 1)]
                ys = ys + [ys[-1] + i*dy for i in range(1, no_segments + 1)] 
            else:
                xs = [start[0] + i*dx for i in range(no_segments + 1)]
                ys = [start[1] + i*dy for i in range(no_segments + 1)]
            if xs[-1] != end[0] or ys[-1] != end[1]:
                xs.append(end[0])
                ys.append(end[1])
            new_nodes = zip(xs, ys)
            for i in range(len(new_nodes) - 1):
                n, m = new_nodes[i], new_nodes[i+1]
                d = pysal.cg.get_points_dist(Point(n), Point(m))
                if n not in mesh_net: mesh_net[n] = {}
                if m not in mesh_net: mesh_net[m] = {}
                mesh_net[n][m] = d
                mesh_net[m][n] = d
            done[(n1,n2)] = True
        #done.add(n1)
    return mesh_net

def write_network_to_pysalshp(network, filename, header=None, field_spec=None):

    if not filename.endswith('shp') and not filename.endswith('SHP'):
        print 'filename would end with shp or SHP'
        return

    shp = pysal.open(filename, 'w')
    dbf = pysal.open(filename[:-3] + 'dbf', 'w')
    if not header:
        dbf.header = ['ID', 'VALUE']
    else:
        dbf.header = ['ID'] + header
    if not field_spec:
        dbf.field_spec = [('N', 9, 0), ('N', 15, 8)]
        def getValue(G, n, m):
            return [G[n][m]]
    else:
        dbf.field_spec = [('N', 9, 0)] + field_spec
        v = network[network.keys()[0]] 
        if type(v) == dict:
            v = v.values()[0]
        if type(v) == list:
            wrap_func = list
        else:
            def wrap_func(value):
                return [value]
        def getValue(G, n, m):
            return wrap_func(G[n][m])
             
    used, counter = set(), 0
    for n1 in network:
        for n2 in network[n1]:
            if n2 in used: continue
            shp.write(Chain([Point(n1), Point(n2)]))                           
            dbf.write([counter] + getValue(network,n1,n2))
            counter += 1
        used.add(n1)

    shp.close()
    dbf.close()

def write_valued_network_to_shp(filename, fields, types, net, values, valFunc):
    oShp = pysal.open(filename, 'w')        
    oDbf = pysal.open(filename[:-3] + 'dbf', 'w')
    oDbf.header = fields                    
    oDbf.field_spec = types                 
    #for n in net:                           
    #    for m in net:                       
    #        oShp.write(Chain([Point(n), Point(m)]))
    #        oDbf.write([valFunc(values, n), valFunc(values, m)])
    used, counter = set(), 0
    for n in net:                           
        for m in net[n]:
            if m in used: continue                       
            oShp.write(Chain([Point(n), Point(m)]))
            oDbf.write([valFunc(values, n), valFunc(values, m)])
            counter += 1
        used.add(n)
    oShp.close()                            
    oDbf.close()

def write_list_network_to_shp(filename, fields, types, net):
    oShp = pysal.open(filename, 'w')        
    oDbf = pysal.open(filename[:-3] + 'dbf', 'w')
    oDbf.header = ['ID'] + fields                    
    oDbf.field_spec = [('N',9,0)] + types                 
    for i, rec in enumerate(net):
        geom = rec[0]
        table_data = list(rec[1:])
        oShp.write(Chain([Point(geom[0]), Point(geom[1])]))
        oDbf.write([i] + table_data)
    oShp.close()
    oDbf.close()

