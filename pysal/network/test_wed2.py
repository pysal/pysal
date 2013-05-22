import pysal as ps
import numpy as np
import operator

def adj_nodes(start_key, edges):
    start_key
    vnext = []
    for edge in edges:
        if edge[0] == start_key:
            vnext.append(edge[1])
    if len(vnext) == 0:
        pass
        #print "Vertex is end point."
    return vnext

def regions_from_graph(nodes, edges, remove_holes = False):
    """
    Extract regions from nodes and edges of a planar graph

    Arguments
    ---------

    nodes: dictionary with vertex id as key, coordinates of vertex as value

    edges: list of (head,tail), (tail, head) edges

    Returns
    ------

    regions: list of lists of nodes defining a region. Includes the external
    region
    filaments: list of lists of nodes defining filaments and isolated
    vertices



    Examples
    --------
    >>> vertices = {0: (1, 8), 1: (1, 7), 2: (4, 7), 3: (0, 4), 4: (5, 4), 5: (3, 5), 6: (2, 4.5), 7: (6.5, 9), 8: (6.2, 5), 9: (5.5, 3), 10: (7, 3), 11: (7.5, 7.25), 12: (8, 4), 13: (11.5, 7.25), 14: (9, 1), 15: (11, 3), 16: (12, 2), 17: (12, 5), 18: (13.5, 6), 19: (14, 7.25), 20: (16, 4), 21: (18, 8.5), 22: (16, 1), 23: (21, 1), 24: (21, 4), 25: (18, 3.5), 26: (17, 2), 27: (19, 2)}
    >>> edges = [(1, 2),(1, 3),(2, 1),(2, 4),(2, 7),(3, 1),(3, 4),(4, 2),(4, 3),(4, 5),(5, 4),(5, 6),(6, 5),(7, 2),(7, 11),(8, 9),(8, 10),(9, 8),(9, 10),(10, 8),(10, 9),(11, 7),(11, 12),(11, 13),(12, 11),(12, 13),(12, 20),(13, 11),(13, 12),(13, 18),(14, 15),(15, 14),(15, 16),(16, 15),(18, 13),(18, 19),(19, 18),(19, 20),(19, 21),(20, 12),(20, 19),(20, 21),(20, 22),(20, 24),(21, 19),(21, 20),(22, 20),(22, 23),(23, 22),(23, 24),(24, 20),(24, 23),(25, 26),(25, 27),(26, 25),(26, 27),(27, 25),(27, 26)]
    >>> r = regions_from_graph(vertices, edges)
    >>> r['filaments']
    [[6, 5, 4], 0, [2, 7, 11], [14, 15, 16], 17]
    >>> r['regions']
    [[3, 4, 2, 1, 3], [9, 10, 8, 9], [11, 12, 13, 11], [12, 20, 19, 18, 13, 12], [19, 20, 21, 19], [22, 23, 24, 20, 22], [26, 27, 25, 26]]

    Notes
    -----
    Based on
    Eberly http://www.geometrictools.com/Documentation/MinimalCycleBasis.pdf.
    """
    def find_start_node(nodes,node_coord):
        start_node = []
        minx = float('inf')
        for key,node in nodes.items():
            if node[0] <= minx:
                minx = node[0]
                start_node.append(key)
        if len(start_node) > 1:
            miny = float('inf')
            for i in range(len(start_node)):
                if nodes[i][1] < miny:
                    miny = nodes[i][1]
                else:
                    start_node.remove(i)
        return nodes[start_node[0]], node_coord[nodes[start_node[0]]]
    
    def clockwise(nodes,vnext,start_key, v_prev, vertices=None):
        v_curr = np.asarray(nodes[start_key])
        v_next = None
        if v_prev == None:
            v_prev = np.asarray([0,-1]) #This should be a vertical tangent to the start node at initialization.
        else:
            pass
        d_curr = v_curr - v_prev
        
        for v_adj in vnext:
            #No backtracking
            if np.array_equal(np.asarray(nodes[v_adj]),v_prev) == True:
                
                continue
            if type(v_prev) == int:
                if v_adj == v_prev:
                    continue
            
            #The potential direction to move in
            d_adj = np.asarray(nodes[v_adj]) - v_curr
            
            #Select the first candidate
            if v_next is None:
                v_next = np.asarray(nodes[v_adj])
                d_next = d_adj
                convex = d_next[0]*d_curr[1] - d_next[1]*d_curr[0]
                if convex <= 0:
                    convex = True
                else:
                    convex = False
            
            #Update if the next candidate is clockwise of the current clock-wise most
            if convex == True:
                if (d_curr[0]*d_adj[1] - d_curr[1]*d_adj[0]) < 0 or (d_next[0]*d_adj[1]-d_next[1]*d_adj[0]) < 0:
                    v_next = np.asarray(nodes[v_adj])
                    d_next = d_adj 
                    convex = d_next[0]*d_curr[1] - d_next[1]*d_curr[0]
                    if convex <= 0:
                        convex = True
                    else:
                        convex = False
            else:
                if (d_curr[0]*d_adj[1] - d_curr[1]*d_adj[0]) < 0 and (d_next[0]*d_adj[1]-d_next[1]*d_adj[0]) < 0:
                    v_next = np.asarray(nodes[v_adj])
                    d_next = d_adj 
                    convex = d_next[0]*d_curr[1] - d_next[1]*d_curr[0] 
                    if convex <= 0:
                        convex = True
                    else:
                        convex = False
        prev_key = start_key
        if vertices == None:
            return tuple(v_next.tolist()), node_coord[tuple(v_next.tolist())], prev_key
        else:
            return tuple(v_next.tolist()), vertices[tuple(v_next.tolist())], prev_key
    def counterclockwise(nodes, vnexts, start_key, prev_key):
        v_next = None
        v_prev = np.asarray(nodes[prev_key])
        v_curr = np.asarray(nodes[start_key])
        d_curr = v_curr - v_prev
        
        for v_adj in vnexts: 
            #Prohibit Back-tracking
            if v_adj == prev_key:
                continue
            d_adj = np.asarray(nodes[v_adj]) - v_curr
            
            if v_next == None:
                v_next = np.asarray(nodes[v_adj])
                d_next = d_adj
                convex = d_next[0]*d_curr[1] - d_next[1]*d_curr[0]
                
            if convex <= 0:
                if d_curr[0]*d_adj[1] - d_curr[1]*d_adj[0] > 0 and d_next[0]*d_adj[1] - d_next[1]*d_adj[0] > 0:
                    v_next = np.asarray(nodes[v_adj])
                    d_next = d_adj 
                    convex = d_next[0]*d_curr[1] - d_next[1]*d_curr[0]
                else:
                    pass
            else:
                if d_curr[0]*d_adj[1] - d_curr[1]*d_adj[0] > 0 or d_next[0]*d_adj[1]-d_next[1]*d_adj[0] > 0:
                    v_next = np.asarray(nodes[v_adj])
                    d_next = d_adj 
                    convex = d_next[0]*d_curr[1] - d_next[1]*d_curr[0]
                else:
                    pass
        prev_key = start_key
        try:
            return tuple(v_next.tolist()), node_coord[tuple(v_next.tolist())], prev_key
        except: 
            return v_next, None, prev_key
    def remove_edge(v0,v1,edges, ext_edges):
        try:
            ext_edges.append((v0,v1))
            ext_edges.append((v1,v0))
            edges.remove((v0,v1))
            edges.remove((v1,v0))
        except:
            pass
        return edges, ext_edges
    
    def remove_heap(v0,sorted_nodes):
        sorted_nodes[:] = [x for x in sorted_nodes if x[0] != v0]
        return sorted_nodes
            
    def remove_node(v0, nodes, nodes_coord, vertices):
        vertices[v0] = nodes[v0]
        del nodes_coord[nodes[v0]]
        del nodes[v0]
        return nodes, nodes_coord, vertices
        
    def extractisolated(nodes,node_coord,v0,primitives, vertices, ext_edges):
        primitives.append(v0)
        nodes, node_coord, vertices = remove_node(v0, nodes, node_coord, vertices)
        return nodes, node_coord, primitives, vertices, ext_edges
    
    def extractfilament(v0,v1, nodes, node_coord,sorted_nodes, edges, primitives,cycle_edge, vertices, ext_edges, iscycle=False):
        if (v0,v1) in cycle_edge or (v1,v0) in cycle_edge:
            iscycle = True
        if iscycle == True:
        #This deletes edges that are part of a cycle, but does not add them as primitives.
            if len(adj_nodes(v0,edges)) >= 3:
                edges, ext_edges = remove_edge(v0,v1,edges, ext_edges)
                v0 = v1
                if len(adj_nodes(v0, edges)) == 1:
                    v1 = adj_nodes(v0, edges)[0]
            while len(adj_nodes(v0, edges)) == 1:
                v1 = adj_nodes(v0, edges)[0]
                #Here I need to do the cycle check again.
                iscycle = False
                if (v0,v1) in cycle_edge or (v1,v0) in cycle_edge:
                    iscycle = True
                
                if iscycle == True:
                    edges, ext_edges = remove_edge(v0,v1,edges, ext_edges)
                    nodes, node_coord, vertices = remove_node(v0, nodes, node_coord, vertices)
                    sorted_nodes = remove_heap(v0, sorted_nodes)
                    v0 = v1
                else:
                    break
            if len(adj_nodes(v0, edges)) == 0:
                
                nodes, node_coord, vertices = remove_node(v0, nodes, node_coord, vertices)
                sorted_nodes = remove_heap(v0, sorted_nodes)
        else:
            #Filament found
            primitive = []
            if len(adj_nodes(v0,edges)) >= 3:
                primitive.append(v0)
                edges, ext_edges = remove_edge(v0,v1,edges, ext_edges)
                v0 = v1
                if len(adj_nodes(v0, edges)) == 1:
                    v1 = adj_nodes(v0, edges)[0]
    
            while len(adj_nodes(v0, edges)) == 1:
                primitive.append(v0)
                v1 = adj_nodes(v0, edges)[0]
                sorted_nodes = remove_heap(v0, sorted_nodes)
                edges, ext_edges = remove_edge(v0, v1, edges, ext_edges)
                nodes, node_coord, vertices = remove_node(v0, nodes, node_coord, vertices)
                v0 = v1
            
            primitive.append(v0)
            if len(adj_nodes(v0, edges)) == 0:
                sorted_nodes = remove_heap(v0, sorted_nodes)
                edges, ext_edges = remove_edge(v0, v1, edges, ext_edges)
                nodes, node_coord, vertices = remove_node(v0, nodes, node_coord, vertices)
            primitives.append((primitive))
        return sorted_nodes, edges, nodes, node_coord, primitives, vertices, ext_edges
    
    def extract_primitives(start_key,sorted_nodes, edges, nodes, node_coord, primitives,minimal_cycles,cycle_edge, vertices, ext_edges):
        v0 = start_key
        visited = []
        sequence = []
        sequence.append(v0)
    
        #Find the CWise most vertex
        vnext = adj_nodes(start_key, edges)
        start_node,v1,v_prev = clockwise(nodes,vnext,start_key,prev_key)
        v_curr = v1
        v_prev = v0
        #Find minimal cycle using CCWise rule
        process = True
        if v_curr == None:
            process = False
        elif v_curr == v0:
            process = False
        elif v_curr in visited:
            process = False
        
        while process == True:
            sequence.append(v_curr)
            visited.append(v_curr)
            vnext = adj_nodes(v_curr, edges)
            v_curr_coords,v_next,v_prev = counterclockwise(nodes,vnext,v_curr, v_prev)
            v_curr = v_next
            if v_curr == None:
                process = False
            elif v_curr == v0:
                process = False
            elif v_curr in visited:
                process = False
        
        if v_curr is None:
            #Filament found, not necessarily at start_key
            sorted_nodes, edges, nodes, node_coord, primitives, vertices, ext_edges = extractfilament(v_prev, adj_nodes(v_prev, edges)[0],nodes, node_coord, sorted_nodes, edges, primitives, cycle_edge, vertices, ext_edges)
            
        elif v_curr == v0:
            #Minimal cycle found
            primitive = []
            iscycle=True
            sequence.append(v0)
            minimal_cycles.append(list(sequence))
            #Remove the v0, v1 edges from the graph.
            edges, ext_edges = remove_edge(v0,v1,edges, ext_edges)
            sorted_nodes = remove_heap(v0, sorted_nodes)#Not in pseudo-code, but in source.
            #Mark all the edges as being part of a minimal cycle.
            if len(adj_nodes(v0, edges)) == 1:
                cycle_edge.append((v0, adj_nodes(v0, edges)[0]))
                cycle_edge.append((adj_nodes(v0, edges)[0], v0))
                sorted_nodes, edges, nodes, node_coord, primitives, vertices, ext_edges = extractfilament(v0, adj_nodes(v0, edges)[0],nodes, node_coord, sorted_nodes, edges, primitives,cycle_edge, vertices, ext_edges)
            if len(adj_nodes(v1, edges)) == 1:
                cycle_edge.append((v1, adj_nodes(v1, edges)[0]))
                cycle_edge.append((adj_nodes(v1, edges)[0],v1))
                sorted_nodes, edges, nodes, node_coord, primitives, vertices, ext_edges = extractfilament(v1, adj_nodes(v1, edges)[0],nodes, node_coord, sorted_nodes, edges, primitives, cycle_edge, vertices, ext_edges)
           
            for i,v in enumerate(sequence[1:-1]):
                cycle_edge.append((v,sequence[i]))
                cycle_edge.append((sequence[i],v))
            
        else:
            #vcurr was visited earlier, so traverse the filament to find the end
            while len(adj_nodes(v0,edges)) == 2:
                if adj_nodes(v0,edges)[0] != v1:
                    v1 = v0
                    v0 = adj_nodes(v0,edges)[0]
                else:
                    v1 = v0
                    v0 = adj_nodes(v0, edges)[1]
            sorted_nodes, edges, nodes, node_coord, primitives = extractfilament(v0,v1,nodes, node_coord, sorted_nodes, edges, primitives,cycle_edge)
        return sorted_nodes, edges, nodes, node_coord, primitives, minimal_cycles,cycle_edge, vertices, ext_edges
    #1.
    sorted_nodes = sorted(nodes.iteritems(), key=operator.itemgetter(1))
    node_coord = dict (zip(nodes.values(),nodes.keys()))
    
    #2.
    primitives = []
    minimal_cycles = []
    cycle_edge = []
    prev_key = None #This is only true for the first iteration.
    #This handles edge and node deletion we need populated later.
    vertices = {}
    ext_edges = []    
    
    #3.
    while sorted_nodes: #Iterate through the sorted list
        start_key = sorted_nodes[0][0]
        numadj = len(adj_nodes(start_key, edges))
        if numadj == 0:
            nodes, node_coord, primitives, vertices, ext_edges = extractisolated(nodes,node_coord,start_key,primitives, vertices, ext_edges)
            sorted_nodes.pop(0)
        elif numadj == 1:
            sorted_nodes, edges, nodes, node_coord, primitives, vertices, ext_edges = extractfilament(start_key, adj_nodes(start_key, edges)[0],nodes, node_coord, sorted_nodes, edges,primitives,cycle_edge, vertices, ext_edges)
        else:
            sorted_nodes, edges, nodes, node_coord, primitives, minimal_cycles,cycle_edge, vertices, ext_edges = extract_primitives(start_key,sorted_nodes, edges, nodes, node_coord, primitives, minimal_cycles,cycle_edge, vertices, ext_edges)
    
    #4. Remove holes from the graph
    if remove_holes == True:
        polys = []
        for cycle in minimal_cycles:
            polys.append(ps.cg.Polygon([ps.cg.Point(vertices[pnt]) for pnt in cycle]))
            
        pl = ps.cg.PolygonLocator(polys)
           
        # find all overlapping polygon mbrs
        overlaps ={}
        nump = len(minimal_cycles)
        for i in range(nump):
            overlaps[i] = pl.overlapping(polys[i].bounding_box)
        
        # for overlapping mbrs (left,right) check if right polygon is contained in left
        holes = []
        for k in overlaps:
            for  pc in overlaps[k]:
                s = sum( [polys[k].contains_point(v) for v in pc.vertices])
                if s == len(pc.vertices):
                    # print k, pc
                    holes.append((k,pc))    
        
        for hole in holes:
            outer, inner = hole
            inner = polys.index(inner)
            minimal_cycles.pop(inner)
    
    #5. Remove isolated vertices
    filaments = []
    for index, primitive in enumerate(primitives):
        if type(primitive) == list:
            filaments.append(primitive)
    
    results = {}
    results['regions'] = minimal_cycles
    results['filaments'] = filaments
    results['vertices'] = vertices
    results['edges'] = ext_edges
    results['nodes'] = vertices
    return results

def internal_or_external(polys,filament, vertices):
    #Modification of Serge's code to find poly in poly for line in poly
    #pl = ps.cg.PolygonLocator(polys) #Trying to use this for polyline in polygon
    
    #Spatial Index of Filaments and minimal cycles (polygons)
    polyline = ps.cg.Chain([ps.cg.Point(vertices[pnt]) for pnt in filament])
    polyline_mbr =  polyline.bounding_box
    pl = ps.cg.PolygonLocator(polys)
    overlaps = pl.overlapping(polyline_mbr)
    
    #For the overlapping MBRs check to see if the polyline is internal or external to the min cycle
    for k in range(len(overlaps)):
        s = sum(overlaps[k].contains_point(v) for v in polyline.vertices)
        if s == len(polyline.vertices):
            #Internal
            return True
        else:
            return False

def classify_filaments(filaments, cycles, edges, vertices):
    classified_filaments = {}
    #Dict matching nodes to cycles
    node_mem = {}
    bridge_mem = {}
    for cycle in cycles:
        for node in cycle:
            node_mem[node] = cycle
    
    #DS to hold classifications
    bridge_filaments = []
    isolated_filaments = []
    external_filaments = []
    internal_filaments = []
    
    #Polygon Spatial Index
    # build polygon spatial index
    polys = []
    for cycle in cycles:
        polys.append(ps.cg.Polygon([ps.cg.Point(vertices[pnt]) for pnt in cycle]))        
    
    for filament in filaments:
        if filament[0] in node_mem.keys() and filament[-1] in node_mem.keys():
            bridge_filaments.append(filament)
            for node in filament:
                bridge_mem[node] = filament
        elif filament[0] not in node_mem.keys() and filament[-1] not in node_mem.keys():
            if filament[0] not in bridge_mem.keys() and filament[-1] not in bridge_mem.keys():
                isolated_filaments.append(filament)
            else:
                #Check internal or external
                if internal_or_external(polys,filament, vertices) == True:
                    internal_filaments.append(filament)
                else:
                    external_filaments.append(filament)
        else:
            #Check internal or external
            if internal_or_external(polys,filament, vertices) == True:
                internal_filaments.append(filament)
            else:
                external_filaments.append(filament)
    
    classified_filaments['isolated'] = isolated_filaments
    classified_filaments['bridge'] = bridge_filaments
    classified_filaments['external'] = external_filaments
    classified_filaments['internal'] = internal_filaments
    
    return classified_filaments

def generate_wed(regions):
    left_region = {}
    right_region = {}
    edges = {}
    region_edge = {}
    start_c = {}
    end_c = {}
    start_cc = {}
    end_cc = {}
    
    node_edge = {}
    
    for region in regions:
        r = [region[-2]]
        r.extend(region)
        r.append(region[1])
        for i in range(len(region)-1):
            edge = r[i+1],r[i+2]
            if edge[0] not in node_edge:
                node_edge[edge[0]] = edge
            if edge[1] not in node_edge:
                node_edge[edge[1]] = edge
            s_c = r[i],r[i+1]
            e_cc = r[i+2],r[i+3]
            right_region[edge] = region
            region_edge[tuple(region)] = edge
            start_c[edge] = s_c
            end_cc[edge] = e_cc
    
            left_region[edge[1],edge[0]] = region
            start_cc[edge[1], edge[0] ] = end_cc[edge]
            end_c[edge[1], edge[0] ] = start_c[edge]
    
            edges[edge] = edge
    
    wed = {}
    wed['node_edge'] = node_edge
    wed['end_c'] = end_c
    wed['start_c'] = start_c
    wed['start_cc'] = start_cc
    wed['end_cc'] = end_cc
    wed['edges'] = edges
    wed['region_edge'] = region_edge
    wed['right_region'] = right_region
    wed['left_region'] = left_region    

    return wed

if __name__ == "__main__":
    
    #Eberly
    vertices = {0: (1, 8), 1: (1, 7), 2: (4, 7), 3: (0, 4), 4: (5, 4), 5: (3, 5), 6: (2, 4.5), 7: (6.5, 9), 8: (6.2, 5), 9: (5.5, 3), 10: (7, 3), 11: (7.5, 7.25), 12: (8, 4), 13: (11.5, 7.25), 14: (9, 1), 15: (11, 3), 16: (12, 2), 17: (12, 5), 18: (13.5, 6), 19: (14, 7.25), 20: (16, 4), 21: (18, 8.5), 22: (16, 1), 23: (21, 1), 24: (21, 4), 25: (18, 3.5), 26: (17, 2), 27: (19, 2)}
    
    edges = [(1, 2),(1, 3),(2, 1),(2, 4),(2, 7),(3, 1),(3, 4),(4, 2),(4, 3),(4, 5),(5, 4),(5, 6),(6, 5),(7, 2),(7, 11),(8, 9),(8, 10),(9, 8),(9, 10),(10, 8),(10, 9),(11, 7),(11, 12),(11, 13),(12, 11),(12, 13),(12, 20),(13, 11),(13, 12),(13, 18),(14, 15),(15, 14),(15, 16),(16, 15),(18, 13),(18, 19),(19, 18),(19, 20),(19, 21),(20, 12),(20, 19),(20, 21),(20, 22),(20, 24),(21, 19),(21, 20),(22, 20),(22, 23),(23, 22),(23, 24),(24, 20),(24, 23),(25, 26),(25, 27),(26, 25),(26, 27),(27, 25),(27, 26)]    
    
    cycles = regions_from_graph(vertices, edges)
    
    print "Minimal Cycles: ", cycles['regions']
    print "Filaments: ", cycles['filaments']

    filaments = classify_filaments(cycles['filaments'],cycles['regions'], cycles['edges'], cycles['vertices'])
    print "Isolated Filaments: ",filaments['isolated']
    print "Bridge Filaments: ", filaments['bridge']
    print "Internal Filaments: ", filaments['internal']
    print "External Filaments: ", filaments['external']
    
    wed = generate_wed(cycles['regions'])
    
    print wed['node_edge']