"""
Reader and writer for PySAL network shapefiles
"""


import pysal as ps
import os

def reader(shp_file_name, doubleEdges=True):
    """
    Read a PySAL network (geographic graph) shapefile and create edges and
    coordinates data structures


    Parameters
    ----------

    shp_file_name: Path to shapefile with .shp extension. Has to have been
    created by contrib/spatialnet/

    doubleEdges:  Boolean if True create a twin for each edge



    Returns
    -------

    coords: dict with key a node id and the value a pair of x,y coordinates
    for the node's embedding in the plane

    edges: list of edges (t,f) where t and f are ids of the nodes
    """


    dir_name = os.path.dirname(shp_file_name)
    base_name = os.path.basename(shp_file_name)
    pre,suf = base_name.split(".")
    shp_file = os.path.join(dir_name,pre+".shp")
    dbf_file = os.path.join(dir_name,pre+".dbf")
    sf = ps.open(shp_file)
    df = ps.open(dbf_file)
    edges = []
    coords = {}
    records = df.read()
    df.close()
    for record in records:
        t = record[0]
        f = record[1]
        edges.append((t,f))
    df.close()
    i = 0
    shps = sf.read()
    sf.close()
    for shp in shps:
        t_xy, f_xy = shp.vertices
        t = edges[i][0]
        f = edges[i][1]
        if t not in coords:
            coords[t] = t_xy
        if f not in coords:
            coords[f] = f_xy
        i += 1

    if doubleEdges:
        for edge in edges:
            twin = edge[1],edge[0]
            if twin not in edges:
                edges.append(twin)
    return coords, edges

    

if __name__ == '__main__':

    file_name = "../contrib/spatialnet/eberly_net.shp"
    coords, edges = reader(file_name)
    coords1, edges1 = reader(file_name, doubleEdges=False)






