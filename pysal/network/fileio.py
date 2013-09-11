# IO handlers for network module go here

import cPickle
import json
import ast
import os
import pysal as ps
from data import WED


def wed_to_json(wed, outfile, binary=True):
    #keys need to be strings
    new_wed = {}
    for key, value in vars(wed).iteritems():
        nested_attr = {}
        if isinstance(value, dict):
            for k2, v2 in value.iteritems():
                nested_attr[str(k2)] = v2
            new_wed[key] = nested_attr
        else:
            new_wed[key] = value
    #print new_wed['edge_list']
    if binary:
        with open(outfile, 'w') as outfile:
            outfile.write(cPickle.dumps(new_wed, 1))
    else:
        with open(outfile, 'w') as outfile:
            json_str = json.dumps(new_wed, sort_keys=True, indent=4)
            outfile.write(json_str)


def wed_from_json(infile, binary=True):
    wed = WED()
    if binary:
        with open(infile, 'r') as f:
            data = cPickle.load(f)
    else:
        with open(infile, 'r') as f:
            data = json.loads(f)

    wed.start_c = {ast.literal_eval(key):value for key, value in data['start_c'].iteritems()}
    wed.start_cc = {ast.literal_eval(key):value for key, value in data['start_cc'].iteritems()}
    wed.end_c = {ast.literal_eval(key):value for key, value in data['end_c'].iteritems()}
    wed.end_cc = {ast.literal_eval(key):value for key, value in data['end_cc'].iteritems()}
    wed.region_edge = {ast.literal_eval(key):value for key, value in data['region_edge'].iteritems()}
    wed.node_edge = {ast.literal_eval(key):value for key, value in data['node_edge'].iteritems()}
    wed.right_polygon = {ast.literal_eval(key):value for key, value in data['right_polygon'].iteritems()}
    wed.left_polygon = {ast.literal_eval(key):value for key, value in data['left_polygon'].iteritems()}
    wed.start_node = {ast.literal_eval(key):value for key, value in data['start_node'].iteritems()}
    wed.end_node = {ast.literal_eval(key):value for key, value in data['end_node'].iteritems()}
    wed.node_coords = {ast.literal_eval(key):value for key, value in data['node_coords'].iteritems()}
    wed.edge_list = data['edge_list']

    return wed

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

