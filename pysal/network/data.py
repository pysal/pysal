# Data structures for network module

__author__ = "Sergio Rey <sjsrey@gmail.com>, Jay Laura <jlaura@asu.edu>"

import operator
import math
import numpy as np
import pysal as ps
import util
import networkw


class WED(object):
    """Winged-Edge Data Structure


    """

    def __init__(self, edges=None, coords=None):

        self.start_c = None
        self.start_cc = None
        self.end_c = None
        self.end_cc = None
        self.region_edge = None
        self.node_edge = None
        self.right_polygon = None
        self.left_polygon = None
        self.start_node = None
        self.end_node = None
        self.node_coords = None
        self.edge_list = []
        self.node_list = []

        if edges is not None and coords is not None:
            #Check for single edges and double if needed
            edges = self.check_edges(edges)
            self.edge_list[:] = edges

            #Create the WED object:w

            self.extract_wed(edges, coords)

    def check_edges(self, edges):
        """
        Validator to ensure that edges are double.

        Parameters
        ----------
        edges: list
            edges connecting nodes in the network

        Returns
        -------
        dbl_edges / edges: list
            Either the original edges or double edges
        """

        seen = set()
        seen_add = seen.add
        seen_twice = set()
        for e in edges:
            if e in seen:
                seen_twice.add(e)
            seen_add(e)
            seen_add((e[1], e[0]))
        if len(list(seen_twice)) != len(edges) / 2:
            dbl_edges = []
            for e in edges:
                dbl_edges.append(e)
                dbl_edges.append((e[1], e[0]))
            return dbl_edges
        else:
            return edges

    def _filament_links_node(self, node, node_edge, start_c, end_c):
        """
        Private method that duplicates enum_links_around_node, but
         is callable before the WED is generated.  This is used
         for filament insertion.
        """
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

    def extract_wed(self, edges, coords):
        # helper functions to determine relative position of vectors
        def _dotproduct(v1, v2):
            return sum((a * b) for a, b in zip(v1, v2))

        def _length(v):
            return math.sqrt(_dotproduct(v, v))

        def _angle(v1, v2):
            return math.acos(_dotproduct(v1, v2) / (_length(v1) * _length(v2)))

        """
        Extract the Winged Edge Data structure for a planar graph


        Arguments
        ---------

        edges:  list
                tuples of origin, destination nodes for each edge

        coords: dict
                key is node id, value is a tuple of x,y coordinates for the node


        Returns
        -------
        wed: Dictionary holding the WED with 10 keys

            start_node: dict
                        key is node, value is edge with node as start node

            end_node:   dict
                        key is node, value is edge with node as end node

            right_polygon: dict
                            key is edge, value is id of right polygon to edge

            left_polygon: dict
                        key is edge, value is id of left polygon to edge

            node_edge: dict
                        key is node, value is edge associated with the node

            region_edge: dict
                        key is region, value is an edge on perimeter of region

            start_c:   dict
                        key is edge, value is first edge encountered when rotating
                        clockwise around edge start node

            start_cc:  dict
                        key is edge, value is first edge encountered when rotating
                        counterclockwise around edge start node

            end_c:     dict
                        key is edge, value is first edge encountered when rotating
                        clockwise around edge start end node

            end_cc:    dict
                        key is edge, value is first edge encountered when rotating
                        counterclockwise around edge start end node

        """

        # coords will be destroyed so keep a copy around
        coords_org = coords.copy()

        # find minimum cycles, filaments and isolated nodes
        pos = coords.values()
        mcb = self.regions_from_graph(coords, edges)

        regions = mcb['regions']
        edges = mcb['edges']
        start_node = {}
        end_node = {}
        for edge in edges:
            if edge[0] != edge[1]:  # no self-loops
                start_node[edge] = edge[0]
                end_node[edge] = edge[1]

        right_polygon = {}
        left_polygon = {}
        region_edge = {}
        start_c = {}
        start_cc = {}
        end_c = {}
        end_cc = {}
        node_edge = {}
        for ri, region in enumerate(regions):
            # regions are ccw in mcb
            region.reverse()
            r = [region[-2]]
            r.extend(region)
            r.append(region[1])
            for i in range(len(region) - 1):
                edge = r[i + 1], r[i + 2]
                if edge[0] not in node_edge:
                    node_edge[edge[0]] = edge
                if edge[1] not in node_edge:
                    node_edge[edge[1]] = edge
                start_c[edge] = r[i], r[i + 1]
                end_cc[edge] = r[i + 2], r[i + 3]
                right_polygon[edge] = ri
                twin = edge[1], edge[0]
                left_polygon[twin] = ri
                start_cc[twin] = end_cc[edge]
                end_c[twin] = start_c[edge]
            region_edge[ri] = edge

        rpkeys = right_polygon.keys()  # only minimum cycle regions have explicit right polygons
        noleft_poly = [k for k in rpkeys if k not in left_polygon]

        for edge in noleft_poly:
            left_polygon[edge] = ri + 1
        # Fill out s_c, s_cc, e_c, e_cc pointers for each edge (before filaments are added)
        regions = region_edge.keys()

        # Find the union of adjacent faces/regions
        unions = []
        while noleft_poly:
            path = []
            current = noleft_poly.pop()
            path_head = current[0]
            tail = current[1]
            path.append(current)
            while tail != path_head:
                candidates = [edge for edge in noleft_poly if edge[0] == tail]
                j = 0
                if len(candidates) > 1:
                    # we want candidate that forms largest ccw angle from current
                    angles = []
                    origin = pos[current[1]]
                    x0 = pos[current[0]][0] - origin[0]
                    y0 = pos[current[0]][1] - origin[1]
                    maxangle = 0.0
                    v0 = (x0, y0)

                    for i, candidate in enumerate(candidates):
                        x1 = pos[candidate[1]][0] - origin[0]
                        y1 = pos[candidate[1]][1] - origin[1]
                        v1 = (x1, y1)
                        v0_v1 = _angle(v0, v1)
                        if v0_v1 > maxangle:
                            maxangle = v0_v1
                            j = i
                        angles.append(v0_v1)

                next_edge = candidates[j]
                path.append(next_edge)
                noleft_poly.remove(next_edge)
                tail = next_edge[1]
            unions.append(path)

        for union in unions:
            for prev, edge in enumerate(union[1:-1]):
                start_cc[edge] = union[prev]
                end_c[edge] = union[prev + 2]
            start_cc[union[0]] = union[-1]
            end_c[union[0]] = union[1]
            end_c[union[-1]] = union[0]
            start_cc[union[-1]] = union[-2]

        regions = [set(region) for region in mcb['regions']]
        filaments = mcb['filaments']
        filament_region = {}
        for f, filament in enumerate(filaments):
            filament_region[f] = []
            # set up pointers on filament edges prior to insertion
            ecc, ec, scc, sc, node_edge = self.filament_pointers(filament, node_edge)
            end_cc.update(ecc)
            start_c.update(sc)
            start_cc.update(scc)
            end_c.update(ec)

            # find which regions the filament is incident to
            sf = set(filament)
            incident_nodes = set()
            incident_regions = set()
            for r, region in enumerate(regions):
                sfi = sf.intersection(region)
                while sfi:
                    incident_nodes.add(sfi.pop())
                    incident_regions.add(r)

            while incident_nodes:
                incident_node = incident_nodes.pop()
                incident_links = self._filament_links_node(incident_node, node_edge, start_c, end_c)

                #Polar coordinates centered on incident node, no rotation from x-axis
                origin = coords_org[incident_node]

                #Logic: If the filament has 2 nodes, grab the other one
                # If the filament has 3+, grab the first and last segments
                if filament.index(incident_node) == 0:
                    f = filament[1]
                elif filament.index(incident_node) == 1:
                    f = filament[0]
                else:
                    f = filament[-2]
                filament_end = coords_org[f]
                #print "Filament:{}, Incident_Node:{} ".format(f, incident_node)
                #Determine the relationship between the origin and the filament end
                filamentx = filament_end[0] - origin[0]
                filamenty = filament_end[1] - origin[1]
                filament_theta = math.atan2(filamenty, filamentx) * 180 / math.pi
                if filament_theta < 0:
                    filament_theta += 360
                #Find the rotation necessary to get the filament to theta 0
                f_rotation = 360 - filament_theta

                link_angles = {}
                for link in incident_links:
                    if link[0] == incident_node:
                        link_node = link[1]
                    else:
                        link_node = link[0]
                    #Get the end coord of the incident link
                    link_node_coords = coords_org[link_node]
                    y = link_node_coords[1] - origin[1]
                    x = link_node_coords[0] - origin[0]
                    r = math.sqrt(x**2 + y**2)
                    node_theta = math.atan2(y, x) * 180 / math.pi
                    if node_theta < 0:
                        node_theta += 360
                    #Rotate the edge node to match the new polar axis
                    node_theta += f_rotation
                    if node_theta > 360:
                        node_theta -= 360
                    link_angles[link] = node_theta

                #Get the bisected edges
                ccwise = min(link_angles, key=link_angles.get)
                cwise = max(link_angles, key=link_angles.get)
                #Fix the direction of the bisected edges
                if ccwise.index(incident_node) != 1:
                    ccwise = (ccwise[1], ccwise[0])
                if cwise.index(incident_node) != 1:
                    cwise = (cwise[1], cwise[0])
                #Update the filament pointer in the direction (segment end, incident node)
                end_c[(f, incident_node)] = (cwise[1], cwise[0])
                end_cc[(f, incident_node)] = (ccwise[1], ccwise[0])
                #Reverse the edge direction
                start_c[(incident_node, f)] = (tuple(cwise))
                start_cc[(incident_node, f)] = (tuple(ccwise))
                #Update the bisected edge points in the direction(segment end, incident node)
                #Cwise link
                end_cc[cwise] = (incident_node, f)
                start_cc[(cwise[1], cwise[0])] = (incident_node, f)
                #CCWise link
                start_c[(ccwise[1], ccwise[0])] = (incident_node, f)
                end_c[ccwise] = (incident_node, f)

                for r in incident_regions:
                    poly = ps.cg.Polygon([coords_org[v] for v in regions[r]])
                    if poly.contains_point((coords_org[filament[1]]) or poly.contains_point(coords_org[filament[0]])):
                        for n in range(len(filament)-1):
                            right_polygon[(filament[n], filament[n+1])] = r
                            left_polygon[(filament[n], filament[n+1])] = r
                            right_polygon[(filament[n+1], filament[n])] = r
                            left_polygon[(filament[n+1], filament[n])] = r

        #Fill in start_c and end_cc for external links
        for k, v in start_cc.iteritems():
            if k not in end_cc.keys():
                end_cc[k] = v
        for k, v in end_c.iteritems():
            if k not in start_c.keys():
                start_c[k] = v

        self.start_c = start_c
        self.start_cc = start_cc
        self.end_c = end_c
        self.end_cc = end_cc
        self.region_edge = region_edge
        self.node_edge = node_edge
        self.right_polygon = right_polygon
        self.left_polygon = left_polygon
        self.start_node = start_node
        self.end_node = end_node
        self.node_coords = coords_org
        self.node_list = [n for n in self.node_coords.keys()]

    @staticmethod
    def filament_pointers(filament, node_edge={}):
        """
        Define the edge pointers for a filament


        Arguments
        ---------

        filament:   list
                    ordered nodes defining a graph filament where a filament is
                    defined as a sequence of ordered nodes with at least one
                    internal node having incidence=2

        node_edge:  dict
                    key is a node, value is the edge the node is assigned to

        Returns
        -------

        ecc:    dict
                key is edge, value is first edge encountered when rotating
                counterclockwise around edge start end node

        ec:     dict
                key is edge, value is first edge encountered when rotating
                clockwise around edge start end node


        scc:    dict
                key is edge, value is first edge encountered when rotating
                counterclockwise around edge start node


        sc:     dict
                key is edge, value is first edge encountered when rotating
                clockwise around edge start node

        node_edge: dict
                key is a node, value is the edge the node is assigned to

        """

        nv = len(filament)
        ec = {}
        ecc = {}
        sc = {}
        scc = {}
        for i in range(nv - 2):
            s0 = filament[i]
            e0 = filament[i + 1]
            s1 = filament[i + 2]
            ecc[s0, e0] = e0, s1
            ecc[s1, e0] = e0, s0
            ec[s0, e0] = e0, s1
            sc[e0, s1] = s0, e0
            scc[e0, s1] = s0, e0
            if s0 not in node_edge:
                node_edge[s0] = s0, e0
            if e0 not in node_edge:
                node_edge[e0] = s0, e0
            if s1 not in node_edge:
                node_edge[s1] = e0, s1
        # wrapper pointers for first and last edges
        ecc[filament[-2], filament[-1]] = filament[-1], filament[-2]
        ec[filament[-2], filament[-1]] = filament[-2], filament[-1]
        ecc[filament[1], filament[0]] = filament[0], filament[1]
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

    @staticmethod
    def regions_from_graph(nodes, edges, remove_holes=False):
        """
        Extract regions from nodes and edges of a planar graph

        Arguments
        ---------

        nodes: dict
            vertex id as key, coordinates of vertex as value

        edges: list
            (head,tail), (tail, head) edges

        Returns
        ------

        regions: list
                lists of nodes defining a region. Includes the external region

        filaments:  list
                    lists of nodes defining filaments and isolated vertices



        Examples
        --------
        >>> vertices = {0: (1, 8), 1: (1, 7), 2: (4, 7), 3: (0, 4), 4: (5, 4), 5: (3, 5), 6: (2, 4.5), 7: (6.5, 9), 8: (6.2, 5), 9: (5.5, 3), 10: (7, 3), 11: (7.5, 7.25), 12: (8, 4), 13: (11.5, 7.25), 14: (9, 1), 15: (11, 3), 16: (12, 2), 17: (12, 5), 18: (13.5, 6), 19: (14, 7.25), 20: (16, 4), 21: (18, 8.5), 22: (16, 1), 23: (21, 1), 24: (21, 4), 25: (18, 3.5), 26: (17, 2), 27: (19, 2)}
        >>> edges = [(1, 2),(1, 3),(2, 1),(2, 4),(2, 7),(3, 1),(3, 4),(4, 2),(4, 3),(4, 5),(5, 4),(5, 6),(6, 5),(7, 2),(7, 11),(8, 9),(8, 10),(9, 8),(9, 10),(10, 8),(10, 9),(11, 7),(11, 12),(11, 13),(12, 11),(12, 13),(12, 20),(13, 11),(13, 12),(13, 18),(14, 15),(15, 14),(15, 16),(16, 15),(18, 13),(18, 19),(19, 18),(19, 20),(19, 21),(20, 12),(20, 19),(20, 21),(20, 22),(20, 24),(21, 19),(21, 20),(22, 20),(22, 23),(23, 22),(23, 24),(24, 20),(24, 23),(25, 26),(25, 27),(26, 25),(26, 27),(27, 25),(27, 26)]
        >>> r = WED.regions_from_graph(vertices, edges)
        >>> r['filaments']
        [[6, 5, 4], [2, 7, 11], [14, 15, 16]]
        >>> r['regions']
        [[3, 4, 2, 1, 3], [9, 10, 8, 9], [11, 12, 13, 11], [12, 20, 19, 18, 13, 12], [19, 20, 21, 19], [22, 23, 24, 20, 22], [26, 27, 25, 26]]

        Notes
        -----
        Based on
        Eberly http://www.geometrictools.com/Documentation/MinimalCycleBasis.pdf.
        """

        def adj_nodes(start_key, edges):
            """Finds all nodes adjacent to start_key.

            Parameters
            ----------
            start_key: int
                The id of the node to find neighbors of.

            edges: list
                All edges in the graph

            Returns
            -------
            vnext: list
                List of adjacent nodes.
            """
            start_key
            vnext = []
            for edge in edges:
                if edge[0] == start_key:
                    vnext.append(edge[1])
            if len(vnext) == 0:
                pass
                #print "Vertex is end point."
            return vnext

        def find_start_node(nodes, node_coord):
            start_node = []
            minx = float('inf')
            for key, node in nodes.items():
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

        def clockwise(nodes, vnext, start_key, v_prev, vertices=None):
            v_curr = np.asarray(nodes[start_key])
            v_next = None
            if v_prev is None:
                v_prev = np.asarray([0, -1])  #This should be a vertical tangent to the start node at initialization.
            else:
                pass
            d_curr = v_curr - v_prev

            for v_adj in vnext:
                #No backtracking
                if np.array_equal(np.asarray(nodes[v_adj]), v_prev) is True:
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
                    convex = d_next[0] * d_curr[1] - d_next[1] * d_curr[0]
                    if convex <= 0:
                        convex = True
                    else:
                        convex = False
                #Update if the next candidate is clockwise of the current clock-wise most
                if convex is True:
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
                #primitive = []
                #iscycle=True
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
                sorted_nodes, edges, nodes, node_coord, primitives, vertices, ext_edges = extractfilament(v0,v1,nodes, node_coord, sorted_nodes, edges, primitives,cycle_edge,vertices,ext_edges)

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

    def enum_links_node(self, node):
        return util.enum_links_node(self, node)

    def enum_edges_region(self, region):
        return util.enum_edges_region(self, region)

    def edge_length(self):
        return util.edge_length(self)

    def w_links(self):
        return networkw.w_links(self)
