
# each polygon chain is converted to a series of connected edges

import pysal as ps
import numpy as np
shpf = ps.open(ps.examples.get_path("geodanet/streets.shp"))
shps = []
for shp in shpf:
    shps.append(shp)
shpf.close()

edge2id = {}
id2edge = {}

id2coord = {}
coord2id = {}
n_edges = n_nodes = 0
for i,shp in enumerate(shps):
    print i, len(shp.vertices)
    print shp.vertices
    for j in range(1, len(shp.vertices)):
        o = shp.vertices[j-1]
        d = shp.vertices[j]
        print o,d
        #raw_input('here')
        if o not in coord2id:
            id2coord[n_nodes] = o
            coord2id[o] = n_nodes
            n_nodes += 1
        if d not in coord2id:
            id2coord[n_nodes] = d
            coord2id[d] = n_nodes
            did = n_nodes
            n_nodes+=1

        oid = coord2id[o]
        did = coord2id[d]
        edge = tuple(np.sort((oid,did)))
        if edge not in edge2id:
            edge2id[edge] = n_edges
            n_edges+=1
        id2edge[edge2id[edge]] = edge
    print o,d

coords = id2coord
edges = id2edge.values()


# this will replace createSpatialNetworkShapefile in contrib/spatialnet

shp_out = ps.open("streets_net.shp", 'w')
dbf_out = ps.open("streets_net.dbf", 'w')
dbf_out.header = ["FNODE","TNODE","ONEWAY"]
dbf_out.field_spec = [('N',20,0),('N',20,0),('L',1,0)]
ids = id2coord.keys()
ids.sort()
for edge in edges:
    o = coords[edge[0]]
    d = coords[edge[1]]
    feat = ps.cg.Chain([o,d])
    rec = (edge[0],edge[1],False)
    dbf_out.write(rec)
    shp_out.write(feat)
dbf_out.close()
shp_out.close()

import net_shp_io
file_name = "streets_net.shp"
coords, edges = net_shp_io.reader(file_name)
coords1, edges1 = net_shp_io.reader(file_name, doubleEdges=False)


import wed

wed_streets = wed.WED(edges, coords)
#wed1_streets = wed.extract_wed(edges1, coords1)

regions = wed_streets.region_edge.keys()




