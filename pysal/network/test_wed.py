# set up okabe wed for eberly graph

"""
to do

 - add in holes
 - add in isolated cycles/regions
 - add in end node filaments

 Currently working for enumeration around nodes and around polygons for other
 cases
"""

import pysal as ps
import numpy as np

nodes = range(27)

# put polygons in cw order
# region will be to the right of these edges
r0 = [1,2,4,3,1]
r1 = [11,13,12,11]
r2 = [12,13,18,19,20,12]
r3 = [19,21,20,19]
r4 = [20,24,23,22,20]
#r5 = [25,27,26,25] # hole
#r6 = [8,9,10,8]
# external polygon
r_1 = [ 1,3, 4, 2,7, 11,12,20,22,23,24,20,21,19,18, 13, 11,7, 2,1]

regions = [r0, r1, r2, r3, r4, r_1]
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

def enum_links_node(wed,v0):
    links = []
    l0 = wed['node_edge'][v0]
    links.append(l0)
    l = l0
    v = v0

    searching = True
    while searching:
        if v == l[0]:
            l = wed['start_c'][l]
        else:
            l = wed['end_c'][l]
        if (l is None) or (l == l0):
            searching = False 
        else:
            links.append(l)
            
    return links

def enum_links_region(wed,region):
    l0 = wed['region_edge'][tuple(region)]
    links = []
    l = l0
    links.append(l)
    searching = True
    while searching:
        if wed['right_region'][l] == region:
            l = wed['end_cc'][l]
        else:
            l = wed['start_cc'][l]
        if l == l0:
            searching = False
        else:
            links.append(l)
    return links


