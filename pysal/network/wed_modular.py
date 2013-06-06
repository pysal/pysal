# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Extracting WED from a planar graph
# 
# ## Steps
# 
#  1. Extract Connected Components
#  2. Extract Primatives
#  3. Construct WED 
#     1. Identify right polys for each edge in a region
#     2. Test if a region is a hole - if so add innermost containing region as left poly for each edge
#     3. Edges without a left region have external bounding polygon as implicit left poly
#     4. Fill out s_c, s_cc, e_c, e_cc pointers for each edge

# <markdowncell>

# ## Extract Connected Components

# <codecell>

import numpy as np
import pysal as ps
import copy
import networkx as nx
from numpy import array

# <rawcell>

# Generate the test graph

# <codecell>

# from eberly http://www.geometrictools.com/Documentation/MinimalCycleBasis.pdf
coords = {}
coords[0] = 1,8
coords[1] = 1,7
coords[2] = 4,7
coords[3] = 0,4
coords[4] = 5,4
coords[5] = 3,5
coords[6] = 2, 4.5
coords[7] = 6.5, 9
coords[8] = 6.2, 5
coords[9] = 5.5,3
coords[10] = 7,3
coords[11] = 7.5, 7.25
coords[12] = 8,4
coords[13] = 11.5, 7.25
coords[14] = 9, 1
coords[15] = 11, 3
coords[16] = 12, 2
coords[17] = 12, 5
coords[18] = 13.5, 6
coords[19] = 14, 7.25
coords[20] = 16, 4
coords[21] = 18, 8.5
coords[22] = 16, 1
coords[23] = 21, 1
coords[24] = 21, 4
coords[25] = 18, 3.5
coords[26] = 17, 2
coords[27] = 19, 2


vertices = {}
for v in range(28):
    vertices[v] = []
    
vertices[1] = [2,3]
vertices[2] = [1,4,7]
vertices[3] = [1,4]
vertices[4] = [2,3,5]
vertices[5] = [4,6]
vertices[6] = [5]
vertices[7] = [2,11]
vertices[8] = [9,10]
vertices[9] = [8,10]
vertices[10] = [8,9]
vertices[11] = [7,12,13]
vertices[12] = [11,13,20]
vertices[13] = [11,12,18]
vertices[14] = [15]
vertices[15] = [14, 16]
vertices[16] = [15]
vertices[18] = [13,19]
vertices[19] = [18,20,21]
vertices[20] = [12,19,21,22,24]
vertices[21] = [19,20]
vertices[22] = [20,23]
vertices[23] = [22,24]
vertices[24] = [20,23]
vertices[25] = [26,27]
vertices[26] = [25,27]
vertices[27] = [25,26]

eberly = vertices.copy()
pos = coords.values()

eg = nx.Graph(vertices)
g = nx.Graph(vertices)
nx.draw(g,pos = pos)

# <codecell>

coords

# <codecell>

def connected_component(adjacency, node):
    """
    Find the connected component that a node belongs to

    Arguments
    ---------

    adjacency: (dict) key is a node, value is a list of adjacent nodes

    node: id of node

    Returns
    -------

    visited: list of nodes comprising the connected component containing node

    Notes
    -----
    Relies on a depth first search of the graph
    """
    A = copy.deepcopy(adjacency)
    if node not in A:
        # isolated node
        return [node]
    stack = [node]
    visited = []
    children = A[node]
    searching = True
    visited.append(node)
    while searching:
        current = stack[-1]
        if A[current]:
            child = A[current].pop()
            if child not in visited:
                visited.append(child)
                stack.append(child)
        else:
            stack.remove(current)
            if not stack:
                searching = False
    return visited
        
    

# <codecell>

connected_component(vertices,1)

# <codecell>

connected_component(vertices,13)

# <codecell>

edges = []
for vert in vertices:
    for dest in vertices[vert]:
        edges.append((vert,dest))

# <codecell>

def connected_components(adjacency):
    """
    Find all the connected components in a graph

    Arguments
    ---------
    adjacency: (dict) key is a node, value is a list of adjacent nodes

    Returns
    -------

    components: list of lists for connected components
    """
    nodes = adjacency.keys()
    components = []
    while nodes:
        start = nodes.pop()
        component = connected_component(adjacency, start)
        if len(component) > 1:
            for node in component:
                if node in nodes:
                    nodes.remove(node)
        components.append(component)
    return components

# <codecell>

components = connected_components(vertices)
components

# <codecell>

nx.draw(g,pos = pos)

# <markdowncell>

# ## Extract Primatives

# <codecell>

edges = []
for vert in vertices:
    for dest in vertices[vert]:
        edges.append((vert,dest))

# <codecell>

from test_wed2 import regions_from_graph
mcb = regions_from_graph(coords,edges)

# <codecell>

mcb['regions']

# <codecell>

mcb['filaments']

# <codecell>

mcb.keys()

# <markdowncell>

# ## Extract WED

# <markdowncell>

# Build up 10 pointers:
# 
# - start_node[edge]
# - end_node[edge]
# - right_polygon[edge]
# - left_polygon[edge]
# - node_edge[node]
# - region_edge[edge]
# - s_c[edge]
# - s_cc[edge]
# - e_c[edge]
# - e_cc[edge]
# 
# These are treated in different sections below.

# <markdowncell>

# ### Edge pointers
# 
# - start_node[edge]
# - end_node[edge]

# <codecell>

regions = mcb['regions']
edges = mcb['edges']
vertices = mcb['vertices']
start_node = {}
end_node = {}
for edge in edges:
    start_node[edge] = edge[0]
    end_node[edge] = edge[1]

# <markdowncell>

# ### Right polygon for each edge in each region primative
# 
# Also define start_c, end_cc for each polygon edge and start_cc and end_c for its twin

# <codecell>

right_polygon = {}
left_polygon = {}
region_edge = {}
start_c = {}
start_cc = {}
end_c = {}
end_cc = {}
node_edge = {}
for ri,region in enumerate(regions):
    # regions are ccw in mcb
    region.reverse()
    r = [region[-2]]
    r.extend(region)
    r.append(region[1])
    for i in range(len(region)-1):
        edge = r[i+1], r[i+2]
        if edge[0] not in node_edge:
            node_edge[edge[0]] = edge
        if edge[1] not in node_edge:
            node_edge[edge[1]] = edge
        start_c[edge] = r[i],r[i+1]
        end_cc[edge] = r[i+2], r[i+3]
        right_polygon[edge] = ri
        twin = edge[1],edge[0]
        left_polygon[twin] = ri
        start_cc[twin] = end_cc[edge]
        end_c[twin] = start_c[edge]
    region_edge[ri] = edge
     
    

# <codecell>

start_c[13,18]

# <codecell>


# <codecell>

region

# <codecell>

start_c[24,23]

# <codecell>

pos

# <codecell>

left_polygon.keys()

# <codecell>

right_polygon.keys()

# <codecell>

for key in right_polygon.keys():
    if key not in left_polygon:
        print key

# <markdowncell>

# ## Test for holes

# <codecell>

#lp = right_polygon[20,24] # for now just assign as placeholder

# <codecell>

#left_polygon[26,25] = lp
#left_polygon[25,27] = lp
#left_polygon[27,26] = lp

# <markdowncell>

# ### Edges belonging to a minium cycle at this point without a left region have external bounding polygon as implicit left poly. assign this explicitly

# <codecell>

rpkeys = right_polygon.keys() # only minimum cycle regions have explicit right polygons
noleft_poly = [k for k in rpkeys if k not in left_polygon]

# <codecell>

for edge in noleft_poly:
    left_polygon[edge] = ri+1

# <markdowncell>

# ### Fill out s_c, s_cc, e_c, e_cc pointers for each edge (before filaments are added)

# <codecell>

start_cc

# <codecell>

def enum_links_node(start_c, end_c, node_edge, node):
    links = []
    if node not in node_edge:
        return links
    l0 = node_edge[node]
    links.append(l0)
    l = l0
    v = node
    searching = True
    while searching:
        if v == l[0]:
            l = start_c[l]
        else:
            l = end_c[l]
        if (l is None) or (set(l) == set(l0)):
            searching = False
        else:
            links.append(l)
    return links
    

# <codecell>

def enum_edges_region(right_polygon, end_cc, start_cc, region_edge, region):
    l0 = region_edge[region]
    l = copy.copy(l0)
    edges = []
    edges.append(l)
    traveling = True
    while traveling:
        if region == right_polygon[l]:
            l = end_cc[l]
        else:
            l = start_cc[l]
        edges.append(l)
        if set(l) == set(l0):
            traveling = False
    return edges

# <codecell>

enum_edges_region(right_polygon, end_cc, start_cc, region_edge, 0)

# <codecell>

enum_edges_region(right_polygon, end_cc, start_cc, region_edge, 1)

# <codecell>

regions = region_edge.keys()

# <codecell>

for region in regions:
    print region, enum_edges_region(right_polygon, end_cc, start_cc, region_edge, region)

# <markdowncell>

# ### Find union of regions

# <codecell>

import math

def dotproduct(v1, v2):
    return sum((a*b) for a,b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v,v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

# <codecell>

# find the union of adjacent faces/regions
coords = pos
unions = []
while noleft_poly:
    path =[]
    current = noleft_poly.pop()
    path_head = current[0]
    tail = current[1]
    path.append(current)
    while tail != path_head:
        candidates = [ edge for edge in noleft_poly if edge[0] == tail ]
        j=0
        if len(candidates) > 1:
            # we want candidate that forms largest ccw angle from current
            angles = []
            origin = coords[current[1]]
            x0 = coords[current[0]][0] - origin[0]
            y0 = coords[current[0]][1] - origin[1]
            maxangle = 0.0
            v0 = (x0,y0)
            
            for i,candidate in enumerate(candidates):
                x1 = coords[candidate[1]][0] - origin[0]
                y1 = coords[candidate[1]][1] - origin[1]
                v1 = (x1,y1)
                v0_v1 = angle(v0, v1)
                if v0_v1 > maxangle:
                    maxangle = v0_v1
                    j=i
                angles.append(v0_v1)
                
        next_edge = candidates[j]
        path.append(next_edge)
        noleft_poly.remove(next_edge)
        tail = next_edge[1]
    unions.append(path)
        
    

# <codecell>

# unions has the path that traces out the union of contiguous regions (in cw order) 

# <codecell>

unions

# <codecell>

# Walk around each union in cw fashion
# start_cc[current] = prev
# end_c[current] = next
for union in unions:
    #print union
    for prev, edge in enumerate(union[1:-1]):
        start_cc[edge] = union[prev]
        end_c[edge] = union[prev+2]
        #print edge, start_cc[edge]
    start_cc[union[0]] = union[-1]
    end_c[union[0]] = union[1]
    end_c[union[-1]] = union[0]
    start_cc[union[-1]] = union[-2]

# <codecell>

mcb['filaments']

# <codecell>

# we need to attach filaments at this point
# internal filaments get left and right poly set to containing poly
# external filaments get left and right poly set to external poly
# bridge filaments get left and right poly set to external poly
# isolated filaments (not contained in mcb-regions) have left and right poly set to external poly

# after this find the holes in the external polygon (these should be the connected components)

# <codecell>

mcb['regions']

# <codecell>

enum_links_node(start_c, end_c, node_edge, 4) #before filaments

# <markdowncell>

# ### Fill out s_c, s_cc, e_c, e_cc pointers for each edge after filaments are inserted

# <markdowncell>

# 
# regions = [set(region) for region in mcb['regions']]
# filaments = mcb['filaments']
# filament_region = {}
# for f,filament in enumerate(filaments):
#     filament_region[f] = []
#     # set up pointers on filament edges prior to insertion 
#     n_fil = len(filament)
#     if n_fil == 2:
#         edge = filament[0],filament[1]
#         start_c[edge] = edge
#         start_cc[edge] = edge
#         end_c[edge] = edge
#         end_cc[edge] = edge
#         if edge[0] not in node_edge:
#             node_edge[edge[0]] = edge
#         if edge[1] not in node_edge:
#             node_edge[edge[1]] = edge
#     elif n_fil == 3:
#         edge = filament[0],filament[1]
#         start_c[edge] = edge
#         start_cc[edge] = edge
#         node_edge[edge[0]] = edge
#         succ = filament[1], filament[2]
#         end_c[edge] = succ
#         end_cc[edge] = succ
#         start_c[succ] = edge
#         start_cc[succ] = edge
#         end_c[succ] = succ
#         end_cc[succ] = succ
#         if succ[0] not in node_edge:
#             node_edge[succ[0]] = succ
#         if succ[1] not in node_edge:
#             node_edge[succ[1]] = succ
#     else:
#         n_head = n_fil - 2 # intermediate start nodes
#         for i in range(1,n_head):
#             edge = filament[i], filament[i+1]
#             start_c[edge] = filamet[i-1], filament[i]
#             start_cc[edge] = filament[i-1], filament[i]
#             end_c[edge] = filament[i+1], filament[i+2]
#             end_cc[edge] = filament[i+1], filament[i+2]
#             if edge[0] not in node_edge:
#                 node_edge[edge[0]] = edge
#         # first edge
#         start_c[filament[0],filament[1]] = filament[0], filament[1]
#         start_cc[filament[0],filament[1]] = filament[0], filament[1]
#         end_c[filament[0],filament[1]] = filament[1], filament[2]
#         end_cc[filament[0],filament[1]] = filament[1], filament[2]
#         # last edge
#         start_c[filament[-2],filament[-1]] = filament[-3], filament[-2]
#         start_cc[filament[-2],filament[-1]] = filament[-3], filament[-2]
#         end_c[filament[-2],filament[-1]] = filament[-2], filament[-1]
#         end_cc[filament[-2],filament[-1]] = filament[-2], filament[-1]
#         
#     # find which regions the filament is adjacent to
#     sf = set(filament)
#     for r,region in enumerate(regions):
#         sfi = sf.intersection(region)
#         if sfi:
#             node = sfi.pop()
#             filament_region[f].append(r)
#             # find edges in region that that are adjacent to sfi
#             # find which pair of edges in the region that the filament bisects
#             #print node, mcb['regions'][r], filament
#             if mcb['regions'][r].count(node) == 2:
#                 e1 = node, mcb['regions'][r][-2]
#                 e2 = node, mcb['regions'][r][1]
#             else:
#                 i = mcb['regions'][r].index(node)
#                 e1 = node, mcb['regions'][r][i-1]
#                 e2 = node, mcb['regions'][r][i+1]
#                 
#             #print e1,e2
#             
#             # get filament edge
#             fi = filament.index(node)
#         
#             fstart = True # start of filament is adjacent node to region
#             if filament[-1] == filament[fi]:
#                 filament.reverse() # put endnode at tail of list
#                 fstart = False # end of filament is adjacent node to region
#             fi = 0
#             fj = 1
# 
#                 
#             #print filament[fi],filament[fj]
#             node_edge[filament[fj]] = filament[fi], filament[fj]
#             
#             # if filament[j] is right of both e1[1],e1[0], and e2[0], e2[1] it is an internal filament to the region
#             
#             A = vertices[e1[1]]
#             B = vertices[e1[0]]
#             C = vertices[filament[fj]]
#             area_abc = A[0] * (B[1]-C[1]) + B[0] * (C[1]-A[1]) + C[0] * (A[1]- B[1])
#             D = vertices[e2[0]]
#             E = vertices[e2[1]]
#             area_dec = D[0] * (E[1] - C[1]) + E[0] * (C[1] - D[1]) + C[0] * (D[1] - E[1])
#             
# 
#             if area_abc < 0 and area_dec < 0:
# 
#                 # print 'inside'
#                 end_cc[e1[1],e1[0]] = filament[fi],filament[fj]
#                 start_c[e2] = filament[fi],filament[fj]
#                 start_c[filament[fi],filament[fj]] = e1[1],e1[0]
#                 start_cc[filament[fi],filament[fj]] = e2
#                 right_polygon[filament[fi],filament[fj]] = r
#                 left_polygon[filament[fi],filament[fj]] = r
#                 
#                 n_f = len(filament) - 1 # number of filament edges
#                 for j in range(1,n_f):
#                     sj = j
#                     ej = j + 1
#                     start_c[filament[sj],filament[ej]] = filament[sj-1], filament[sj]
#                     start_cc[filament[sj],filament[ej]] = filament[sj-1], filament[sj]
#                     end_c[filament[sj-1], filament[sj]] = filament[sj],filament[ej]
#                     end_cc[filament[sj-1], filament[sj]] = filament[sj],filament[ej]
#                     #node_edge[filament[sj]] = filament[sj],filament[ej]
#                     right_polygon[filament[sj],filament[ej]] = r
#                     left_polygon[filament[sj],filament[ej]] = r
#                 # last edge
#                 end_c[filament[-2],filament[-1]] = filament[-2],filament[-1]
#                 end_cc[filament[-2],filament[-1]] = filament[-2],filament[-1]
#                 #node_edge[filament[-1]] = filament[-2],filament[-1]
#                 right_polygon[filament[-1],filament[-2]] = r
#                 left_polygon[filament[-1],filament[-2]] = r
#                
#                 
#             else:
#                 # print 'outside'
#                 end_c[e1[1],e1[0]] = filament[fi],filament[fj]
#                 start_cc[e2] = filament[fi],filament[fj]
#                 start_cc[filament[fi],filament[fj]] = e1[1],e1[0]
#                 start_c[filament[fi],filament[fj]] = e2
#                 
#                 n_f = len(filament) - 1 # number of filament edges
#                 for j in range(1,n_f):
#                     sj = j
#                     ej = j + 1
#                     start_c[filament[sj],filament[ej]] = filament[sj-1], filament[sj]
#                     start_cc[filament[sj],filament[ej]] = filament[sj-1], filament[sj]
#                     end_c[filament[sj-1], filament[sj]] = filament[sj],filament[ej]
#                     end_cc[filament[sj-1], filament[sj]] = filament[sj],filament[ej]
#                     #node_edge[filament[sj]] = filament[sj],filament[ej]
# 
#                 # last edge
#                 end_c[filament[-2],filament[-1]] = filament[-2],filament[-1]
#                 end_cc[filament[-2],filament[-1]] = filament[-2],filament[-1]
#                 #node_edge[filament[-1]] = filament[-2],filament[-1]
#                
#             
#             
# 
#             
#            
#             
# 
#             
#     
#     
#     

# <codecell>

"""
def filament_pointers(filament):
    ecc = {}
    sc = {}
    scc = {}
    ec = {}
    n_edges = len(filament) - 1
    
    for i in range(n_edges-1):
       s = filament[i]
       e = filament[i+1]
       ns = filament[i+2]
       ecc[s,e] = e,ns
       ecc[ns,e] = e,s
       sc[s,e] = s,e
       scc[s,e] = s,e
       ec[s,e]= e,ns
    # last one 
    ecc[filament[-2], filament[-1]] = filament[-1], filament[-2]
    ecc[filament[-2], filament[-1] ] = filament[-1], filament[-2]
    sc[filament[-2], filament[-1] ] = filament[-1], filament[-2]
    scc[filament[-2], filament[-1] ] = filament[-1], filament[-2]
    ec[filament[-2], filament[-1] ] = filament[-1], filament[-2]
    # last twin
    ec[filament[-1], filament[-2] ] = filament[-2], filament[-1]
    ecc[filament[-1], filament[-2] ] = filament[-2], filament[-1]
    
    # first one twin
    s = filament[1]
    e = filament[0]
    ecc[s,e] = e,s
    
    return ecc, sc, scc, ec
"""
def filament_pointers(filament, node_edge={}):
    nv = len(filament)
    ec = {}
    ecc = {}
    sc = {}
    scc = {}
    for i in range(nv-2):
        s0 = filament[i]
        e0 = filament[i+1]
        s1 = filament[i+2]
        ecc[s0,e0] = e0,s1
        ecc[s1,e0] = e0,s0
        ec[s0,e0] = e0,s1
        sc[e0,s1] = s0,e0
        scc[e0,s1] = s0,e0
        if s0 not in node_edge:
            node_edge[s0] = s0,e0
        if e0 not in node_edge:
            node_edge[e0] = s0,e0
        if s1 not in node_edge:
            node_edge[s1] = e0, s1
    # wrapper pointers for first and last edges
    ecc[filament[-2], filament[-1]] = filament[-1], filament[-2]
    ec[filament[-2], filament[-1]] = filament[-2], filament[-1]
    ecc[filament[1],filament[0]] = filament[0], filament[1]
    ec[filament[1], filament[0]] = filament[1], filament[0]
    sc[filament[0], filament[1]] = filament[0], filament[1]
    # technically filaments have to have at least intermediate node with incidence 2
    # if there is a single edge it isn't a filament, but we handle it here just in case
    # since the "first" edge not be treated in the for loop (which isn't entered)
    if nv == 2:
        sc[filament[0], filament[1]] = filament[0], filament[1]
        ec[filament[0], filament[1]] = filament[0], filament[1]
        ecc[filament[0], filament[1]] = filament[1], filament[0]
        scc[filament[0], filament[1]] = filament[0], filament[1]
        if filament[0] not in node_edge:
            node_edge[filament[0]] = filament[0], filament[1]
        if filament[1] not in node_edge:
            node_edge[filament[1]] = filament[0], filament[1]
    return ecc, ec, scc, sc, node_edge
            

# <codecell>


regions = [set(region) for region in mcb['regions']]
filaments = mcb['filaments']
filament_region = {}
for f,filament in enumerate(filaments):
    filament_region[f] = []
    # set up pointers on filament edges prior to insertion 
    ecc, ec, scc, sc, node_edge = filament_pointers(filament, node_edge)
    print ecc
    end_cc.update(ecc)
    start_c.update(sc)
    start_cc.update(scc)
    end_c.update(ec)
    
    # find which regions the filament is adjacent to
    sf = set(filament)
    for r,region in enumerate(regions):
        sfi = sf.intersection(region)
        if sfi:
            node = sfi.pop()
            filament_region[f].append(r)
            # find edges in region that that are adjacent to sfi
            # find which pair of edges in the region that the filament bisects
            #print node, mcb['regions'][r], filament
            if mcb['regions'][r].count(node) == 2:
                e1 = node, mcb['regions'][r][-2]
                e2 = node, mcb['regions'][r][1]
            else:
                i = mcb['regions'][r].index(node)
                e1 = node, mcb['regions'][r][i-1]
                e2 = node, mcb['regions'][r][i+1]
                
            #print e1,e2
            
            # get filament edge
            fi = filament.index(node)
        
            fstart = True # start of filament is adjacent node to region
            if filament[-1] == filament[fi]:
                filament.reverse() # put endnode at tail of list
                fstart = False # end of filament is adjacent node to region
            fi = 0
            fj = 1

                
            #print filament[fi],filament[fj]
            #node_edge[filament[fj]] = filament[fi], filament[fj]
            
            # if filament[j] is right of both e1[1],e1[0], and e2[0], e2[1] it is an internal filament to the region
            
            A = vertices[e1[1]]
            B = vertices[e1[0]]
            C = vertices[filament[fj]]
            area_abc = A[0] * (B[1]-C[1]) + B[0] * (C[1]-A[1]) + C[0] * (A[1]- B[1])
            D = vertices[e2[0]]
            E = vertices[e2[1]]
            area_dec = D[0] * (E[1] - C[1]) + E[0] * (C[1] - D[1]) + C[0] * (D[1] - E[1])
            

            if area_abc < 0 and area_dec < 0:

                # print 'inside'
                end_cc[e1[1],e1[0]] = filament[fi],filament[fj]
                start_c[e2] = filament[fi],filament[fj]
                start_c[filament[fi],filament[fj]] = e1[1],e1[0]
                start_cc[filament[fi],filament[fj]] = e2
                right_polygon[filament[fi],filament[fj]] = r
                left_polygon[filament[fi],filament[fj]] = r
                right_polygon[filament[fj], filament[fi]] = r
                left_polygon[filament[fj], filament[fi]] = r
                end_cc[filament[fj], filament[fi]] = e2 # twin of first internal edge so enumerate region works
                
                n_f = len(filament) - 1 # number of filament edges
                for j in range(1,n_f):
                    sj = j
                    ej = j + 1
                    #start_c[filament[sj],filament[ej]] = filament[sj-1], filament[sj]
                    #start_cc[filament[sj],filament[ej]] = filament[sj-1], filament[sj]
                    #end_c[filament[sj-1], filament[sj]] = filament[sj],filament[ej]
                    #end_cc[filament[sj-1], filament[sj]] = filament[sj],filament[ej]
                    #node_edge[filament[sj]] = filament[sj],filament[ej]
                    right_polygon[filament[sj],filament[ej]] = r
                    left_polygon[filament[sj],filament[ej]] = r
                    right_polygon[filament[ej],filament[sj]] = r
                    left_polygon[filament[ej],filament[sj]] = r
                # last edge
                #end_c[filament[-2],filament[-1]] = filament[-2],filament[-1]
                #end_cc[filament[-2],filament[-1]] = filament[-2],filament[-1]
                #node_edge[filament[-1]] = filament[-2],filament[-1]
                right_polygon[filament[-1],filament[-2]] = r
                left_polygon[filament[-1],filament[-2]] = r
                right_polygon[filament[-2],filament[-1]] = r
                left_polygon[filament[-2],filament[-1]] = r
                
            else:
                print 'outside', filament[fi], filament[fj]
                end_c[e1[1],e1[0]] = filament[fi],filament[fj]
                start_cc[e2] = filament[fi],filament[fj]
                start_cc[filament[fi],filament[fj]] = e1[1],e1[0]
                start_c[filament[fi],filament[fj]] = e2
                
                n_f = len(filament) - 1 # number of filament edges
                for j in range(1,n_f):
                    sj = j
                    ej = j + 1
                    start_c[filament[sj],filament[ej]] = filament[sj-1], filament[sj]
                    start_cc[filament[sj],filament[ej]] = filament[sj-1], filament[sj]
                    end_c[filament[sj-1], filament[sj]] = filament[sj],filament[ej]
                    end_cc[filament[sj-1], filament[sj]] = filament[sj],filament[ej]
                    #node_edge[filament[sj]] = filament[sj],filament[ej]

                # last edge
                end_c[filament[-2],filament[-1]] = filament[-2],filament[-1]
                end_cc[filament[-2],filament[-1]] = filament[-2],filament[-1]
                #node_edge[filament[-1]] = filament[-2],filament[-1]
               
            
            

            
           
            

print "end_c", end_c          
    
    
    

# <codecell>

end_c

# <codecell>

for node in range(0,28):
    print node, enum_links_node(start_c, end_c, node_edge, node)

# <codecell>

for region in range(5):
    print enum_edges_region(right_polygon, end_cc, start_cc, region_edge, region)
    

# <codecell>

l0 = node_edge[14]

# <codecell>

l0 = region_edge[0]

# <codecell>

l0

# <codecell>

right_polygon[4,3]

# <codecell>

end_cc[4,3]

# <codecell>

right_polygon[3,1]

# <codecell>

end_cc[2,4]

# <codecell>

end_cc[4,5]

# <markdowncell>

# ## WED is complete!

# <codecell>

# test case for multiple internal filaments in one region

vertices = {}
vertices[1] = array([0,10])
vertices[2] = array([10,10])
vertices[4] = array([10, 0])
vertices[3] = array([0, 0])
vertices[5] = array([6,1])
vertices[6] = array([6,3])
vertices[7] = array([6,7])
vertices[8] = array([6,5])
vertices[9] = array([15,5])

# <codecell>

vertices

# <codecell>

regions = [[1,2,4,3,1]]

# <codecell>

regions

# <codecell>

edges = [(1,2),(2,4),(4,3), (3,1)]

# <codecell>

filaments = [ (4,5), (4,6), (4,7), (4,8) ]

# <codecell>

def convex(v0, v1, v):
    """ test if vector v is between v0 and v1 where v0[0]==v1[0]==v[0]
    v0 = list of two points

    """
    d0 = v0[1] - v0[0]
    d = v[1] - v[0]
    
    s1 = d0[0]*d[1]-d0[1]*d[0]
    if s1 > 0.0:
        d1 = v1[1] - v0[0]
        s2 = d[0]*d1[1]-d[1]*d1[0]
        if s2 > 0.0:
            return True
    return False
    

# <codecell>

v0 =[array([10,0]),array([10,10])]
v1 =[array([10,0]),array([0,0])]
v = [array([10,0]),array([6,5])]

# <codecell>

convex(v0,v1,v)

# <codecell>

v0 =[array([10,0]),array([10,10])]
v1 =[array([10,0]),array([0,0])]
v = [array([10,0]),array([15,10])]

# <codecell>

convex(v0,v1,v)

# <codecell>

v0 =[array([0,0]),array([10,0])]
v1 =[array([0,0]),array([0,10])]
v = [array([0,0]),array([2,2])]

# <codecell>

convex(v0,v1,v)

# <codecell>

v0 =[array([0,0]),array([10,0])]
v1 =[array([0,0]),array([0,10])]
v = [array([0,0]),array([-2,-2])]

# <codecell>

convex(v0,v1,v)

# <codecell>



