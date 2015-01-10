from collections import defaultdict, OrderedDict, namedtuple
import math
import os
import copy

import numpy as np
import pysal as ps
from pysal.weights.util import get_ids

import random

import util

class Network:

    """
    A class to store two representations of a network extracted from a polyline shapefile.

    (1) A spatial retpresentation, stored in self.edges.  The spatial
        representation is used to generate edge lengths and used for snapping
        points to the network.

    (2) A graph representation, stored in self.graphedges.  The graph
        representation is used to generate adjacency measures as the
        connectivity of a street is a function of the number of
        intersections with other other streets and not the sinuiosity.

        The logic behind extraction is that an intersection must have
        neighbor cardinality >= 3, while a segment of an edge will have neighbor
        cardinality of 2.  The former are of interest, while the latter are
        considered artifacts of the digitization process.

    Performance Ideas:
    These are just, if needed ideas as I get this coded.
    (1) Edges are stored sorted.  Use skip-lists to speed iteration.
    """

    def __init__(self, in_shp):
        self.in_shp = in_shp

        self.adjacencylist = defaultdict(list)
        self.nodes = {}
        self.edge_lengths = {}
        self.edges = []

        self.pointpatterns = {}

        self.extractnetwork()
        self.node_coords = dict((value, key) for key, value in self.nodes.iteritems())

        #This is a spatial representation of the network.
        self.edges = sorted(self.edges)

        #Extract the graph
        self.extractgraph()

        self.node_list = sorted(self.nodes.values())

    def extractnetwork(self):
        """
        Using the same logic as the high efficiency areal unit weights creation
        extract the network from the edges / vertices.
        """
        nodecount = 0
        edgetpl = namedtuple('Edge', ['source', 'destination'])
        shps = ps.open(self.in_shp)
        for shp in shps:
            vertices = shp.vertices
            for i, v in enumerate(vertices[:-1]):
                try:
                    vid = self.nodes[v]
                except:
                    self.nodes[v] = vid = nodecount
                    nodecount += 1
                try:
                    nvid = self.nodes[vertices[i+1]]
                except:
                    self.nodes[vertices[i+1]] = nvid = nodecount
                    nodecount += 1

                self.adjacencylist[vid].append(nvid)
                self.adjacencylist[nvid].append(vid)

                #Sort the edges so that mono-directional keys can be stored.
                edgenodes = sorted([vid, nvid])
                edge = tuple(edgenodes)
                self.edges.append(edge)
                length = util.compute_length(v, vertices[i+1])
                self.edge_lengths[edge] = length

    def extractgraph(self):
        """
        Extract a graph representation of a network
        """
        self.graphedges = []
        self.edge_to_graph = {}
        self.graph_lengths = {}

        #Find all nodes with cardinality 2
        segment_nodes = []
        for k, v in self.adjacencylist.iteritems():
            #len(v) == 1 #cul-de-sac
            #len(v) == 2 #bridge segment
            #len(v) > 2 #intersection
            if len(v) == 2:
                segment_nodes.append(k)

        #Start with a copy of the spatial representation and iteratively
        # remove edges deemed to be segments
        self.graphedges = copy.deepcopy(self.edges)
        self.graph_lengths = copy.deepcopy(self.edge_lengths)
        self.graph_to_edges = {}  #Mapping all the edges contained within a single graph represented edge

        bridges = []
        for s in segment_nodes:
            bridge = [s]
            neighbors = self.yieldneighbor(s, segment_nodes, bridge)
            while neighbors:
                cnode = neighbors.pop()
                segment_nodes.remove(cnode)
                bridge.append(cnode)
                newneighbors = self.yieldneighbor(cnode, segment_nodes, bridge)
                neighbors += newneighbors
            bridges.append(bridge)

        for bridge in bridges:
            if len(bridge) == 1:
                n = self.adjacencylist[bridge[0]]
                newedge = tuple(sorted([n[0], n[1]]))
                #Identify the edges to be removed
                e1 = tuple(sorted([bridge[0], n[0]]))
                e2 = tuple(sorted([bridge[0], n[1]]))
                #Remove from the graph
                self.graphedges.remove(e1)
                self.graphedges.remove(e2)
                #Remove from the edge lengths
                length_e1 = self.edge_lengths[e1]
                length_e2 = self.edge_lengths[e2]
                self.graph_lengths.pop(e1, None)
                self.graph_lengths.pop(e2, None)
                self.graph_lengths[newedge] = length_e1 + length_e2
                #Update the pointers
                self.graph_to_edges[e1] = newedge
                self.graph_to_edges[e2] = newedge
            else:
                cumulative_length = 0
                startend = {}
                redundant = set([])
                for b in bridge:
                    for n in self.adjacencylist[b]:
                        if n not in bridge:
                            startend[b] = n
                        else:
                            redundant.add(tuple(sorted([b,n])))

                newedge = tuple(sorted(startend.values()))
                for k, v in startend.iteritems():
                    redundant.add(tuple(sorted([k,v])))

                for r in redundant:
                    self.graphedges.remove(r)
                    cumulative_length += self.edge_lengths[r]
                    self.graph_lengths.pop(r, None)
                    self.graph_to_edges[r] = newedge
                self.graph_lengths[newedge] = cumulative_length

            self.graphedges.append(newedge)
        self.graphedges = sorted(self.graphedges)

    def yieldneighbor(self, node, segment_nodes, bridge):
        n = []
        for i in self.adjacencylist[node]:
            if i in segment_nodes and i not in bridge:
                n.append(i)
        return n

    def contiguityweights(self, graph=True, weightings=None):
        """
        Create a contiguity based W object

        Parameters
        -----------
        graph           boolean
                        controls whether the W is generated using the spatial
                        representation or the graph representation (default True)
        weightings      dict of lists of weightings for each edge
        """

        self.wtype = 'Contiguity'
        neighbors = {}
        neighbors = OrderedDict()

        if graph:
            edges = self.graphedges
        else:
            edges = self.edges

        if weightings:
            weights = {}
        else:
            weights = None

        for key in edges:
            neighbors[key] = []
            if weightings:
                weights[key] = []

            for neigh in edges:
                if key == neigh:
                    continue
                if key[0] == neigh[0] or key[0] == neigh[1] or key[1] == neigh[0] or key[1] == neigh[1]:
                    neighbors[key].append(neigh)
                    if weightings:
                        weights[key].append(weightings[neigh])
                #TODO: Add a break condition - everything is sorted, so we know when we have stepped beyond a possible neighbor.
                #if key[1] > neigh[1]:  #NOT THIS
                    #break

        self.w = ps.weights.W(neighbors, weights=weights)

    def distancebandweights(self, threshold):
        """
        Create distance based weights
        """
        self.wtype='Distance: {}'.format(threshold)
        try:
            hasattr(self.alldistances)
        except:
            self.node_distance_matrix()

        neighbor_query = np.where(self.distancematrix < threshold)
        neighbors = defaultdict(list)
        for i, n in enumerate(neighbor_query[0]):
            neigh = neighbor_query[1][i]
            if n != neigh:
                neighbors[n].append(neighbor_query[1][i])

        self.w = ps.weights.W(neighbors)

    def snapobservations(self, shapefile, name, idvariable=None, attribute=None):
        #Explicitly defined kwargs, if it cleaner to just take **kwargs and
        # them to the Point Pattern constructor?
        """
        Snap a point pattern shapefile to a network shapefile.

        Parameters
        ----------
        shapefile   str The PATH to the shapefile
        name        str Name to be assigned to the point dataset
        idvariable  str Column name to be used as ID variable
        attribute   bool Defines whether attributes should be extracted
        """

        self.pointpatterns[name] = PointPattern(shapefile, idvariable=idvariable, attribute=attribute)
        self.snap_to_edge(self.pointpatterns[name])

    def compute_distance_to_nodes(self, x, y, edge):
        """
        Given an observation on a network edge, return the distance to the two
        nodes that bound that end.

        Parameters
        ----------
        network      obj    PySAL network object
        x            float  x-coordinate of the snapped point
        y            float  y-coordiante of the snapped point
        edge         tuple  (node0, node1) representation of the network edge

        Returns
        --------
        d1          float  the distance to node0, always the node with the lesser id
        d2          float  the distance to node1, always the node with the greater id
        """

        d1 = util.compute_length((x,y), self.node_coords[edge[0]])
        d2 = util.compute_length((x,y), self.node_coords[edge[1]])
        return d1, d2

    def snap_to_edge(self, pointpattern):
        """
        Snap point observations to network edges.

        Parameters
        -----------
        pointpattern  obj PySAL Point Pattern Object

        Returns
        obs_to_edge   dict with edge as key and list of points as value
        edge_to_obs   dict with point id as key and edge tuple as value
        dist_to_node  dict with edge as key and tuple of distances to nodes as value
        """

        obs_to_edge = {}
        dist_to_node = {}

        pointpattern.snapped_coordinates = {}

        for pt_index, point in pointpattern.points.iteritems():
            x0 = point['coordinates'][0]
            y0 = point['coordinates'][1]

            d = {}
            vectors = {}
            c = 0

            #Components of this for loop can be pre computed and cached, like denom to distance =
            for edge in self.edges:
                xi = self.node_coords[edge[0]][0]
                yi = self.node_coords[edge[0]][1]
                xi1 = self.node_coords[edge[1]][0]
                yi1 = self.node_coords[edge[1]][1]

                num = ((yi1 - yi)*(x0-xi)-(xi1-xi)*(y0-yi))
                denom = ((yi1-yi)**2 + (xi1-xi)**2)
                k = num / denom
                distance = abs(num) / math.sqrt(((yi1-yi)**2 + (xi1-xi)**2))
                vectors[c] = (xi, xi1, yi, yi1,k,edge)
                d[distance] = c
                c += 1

            min_dist = SortedEdges(sorted(d.items()))

            for dist, vector_id in min_dist.iteritems():
                value = vectors[vector_id]
                xi = value[0]
                xi1 = value[1]
                yi = value[2]
                yi1 = value[3]
                k = value[4]
                edge = value[5]

                #Okabe Method
                x = x0 - k * (yi1 - yi)
                y = y0 + k * (xi1 - xi)

                #Compute the distance from the new point to the nodes
                d1, d2 = self.compute_distance_to_nodes(x, y, edge)

                if xi <= x <= xi1 or xi1 <= x <= xi and yi <= y <= yi1 or yi1 <=y <= yi:
                    #print "{} intersections edge {} at {}".format(pt_index, edge, (x,y))
                    #We are assuming undirected - this should never be true.
                    if edge not in obs_to_edge.keys():
                        obs_to_edge[edge] = {pt_index: (x,y)}
                    else:
                        obs_to_edge[edge][pt_index] =  (x,y)
                    dist_to_node[pt_index] = {edge[0]:d1, edge[1]:d2}
                    pointpattern.snapped_coordinates[pt_index] = (x,y)

                    break
                else:
                    #either pi or pi+1 are the nearest point on that edge.
                    #If this point is closer than the next distance, we can break, the
                    # observation intersects the node with the shorter
                    # distance.
                    pi = (xi, yi)
                    pi1 = (xi1, yi1)
                    p0 = (x0,y0)
                    #Maybe this call to ps.cg should go as well - as per the call in the class above
                    dist_pi = ps.cg.standalone.get_points_dist(p0, pi)
                    dist_pi1 = ps.cg.standalone.get_points_dist(p0, pi1)

                    if dist_pi < dist_pi1:
                        node_dist = dist_pi
                        (x,y) = pi
                    else:
                        node_dist = dist_pi1
                        (x,y) = pi1

                    d1, d2 = self.compute_distance_to_nodes(x, y, edge)

                    if node_dist < min_dist.next_key(dist):
                        if edge not in obs_to_edge.keys():
                            obs_to_edge[edge] = {pt_index: (x, y)}
                        else:
                            obs_to_edge[edge][pt_index] =  (x, y)
                        dist_to_node[pt_index] = {edge[0]:d1, edge[1]:d2}
                        pointpattern.snapped_coordinates[pt_index] = (x,y)
                        break


        obs_to_node = defaultdict(list)
        for k, v in obs_to_edge.iteritems():
            keys = v.keys()
            obs_to_node[k[0]] = keys
            obs_to_node[k[1]] = keys

        edge_to_obs = {}
        pointpattern.obs_to_edge = obs_to_edge
        pointpattern.dist_to_node = dist_to_node
        pointpattern.obs_to_node = obs_to_node

    def count_per_edge(self, obs_on_network, graph=True):
        """
        Snaps observations to the nearest edge and then counts
            the number of observations per edge.

        Parameters
        ----------
        obs_on_network: dict of observations on the network
            {(edge): {pt_id: (coords)}} or {edge: [(coord), (coord), (coord)]}
        Returns
        -------
        counts: dict {(edge):count}
        """
        counts = {}
        if graph:
            for key, observations in obs_on_network.iteritems():
                cnt = len(observations)
                if key in self.graph_to_edges.keys():
                    key = self.graph_to_edges[key]
                try:
                    counts[key] += cnt
                except:
                    counts[key] = cnt
        else:
            for key in obs_on_network.iterkeys():
                counts[key] = len(obs_on_network[key])
        return counts

    def _newpoint_coords(self, edge, distance):
        x1 = self.node_coords[edge[0]][0]
        y1 = self.node_coords[edge[0]][1]
        x2 = self.node_coords[edge[1]][0]
        y2 = self.node_coords[edge[1]][1]
        m = (y2 - y1) / (x2 - x1)
        b1 = y1 - m * (x1)
        if x1 > x2:
            x0 = x1 - distance / math.sqrt(1 + m**2)
        elif x1 < x2:
            x0 = x1 + distance / math.sqrt(1 + m**2)
        y0 = m * (x0 - x1) + y1
        return x0, y0

    def simulate_observations(self, count, distribution='uniform'):
        """
        Generates simulated points

        Parameters
        ----------
        count: integer number of points to create
        distirbution: distribution of random points

        Returns
        -------
        random_pts: dict with {(edge):[(x,y), (x1, y1), ... , (xn,yn)]}
        """
        simpts = SimulatedPointPattern()

        #Cumulative Network Length
        edges = []
        lengths = np.zeros(len(self.edge_lengths))
        for i, key in enumerate(self.edge_lengths.iterkeys()):
            edges.append(key)
            lengths[i] = self.edge_lengths[key]
        stops = np.cumsum(lengths)
        totallength = stops[-1]

        if distribution is 'uniform':
            nrandompts = np.random.uniform(0, totallength, size=(count,))
        elif distribution is 'poisson':
            nrandompts = np.random.uniform(0, totallength, size=(np.random.poisson(count),))

        for i, r in enumerate(nrandompts):
            idx = np.where(r < stops)[0][0]
            assignment_edge = edges[idx]
            distance_from_start = stops[idx] - r
            #Populate the coordinates dict
            x0, y0 = self._newpoint_coords(assignment_edge, distance_from_start)
            simpts.snapped_coordinates[i] = (x0, y0)
            simpts.obs_to_node[assignment_edge[0]].append(i)
            simpts.obs_to_node[assignment_edge[1]].append(i)

            #Populate the distance to node
            simpts.dist_to_node[i] = {assignment_edge[0] : distance_from_start,
                    assignment_edge[1] : self.edge_lengths[edges[idx]] - distance_from_start}

            simpts.points = simpts.snapped_coordinates

        return simpts

    def enum_links_node(self, v0):
        links = []
        neighbornodes =  self.adjacencylist[v0]
        for n in neighbornodes:
            links.append(tuple(sorted([n, v0])))
        return links

    def node_distance_matrix(self):
        self.alldistances = {}
        nnodes = len(self.node_list)
        self.distancematrix = np.empty((nnodes, nnodes))
        for node in self.node_list:
            distance, pred = util.dijkstra(self, self.edge_lengths, node, n=float('inf'))
            pred = np.array(pred)
            tree = util.generatetree(pred)
            cumdist = util.cumulativedistances(np.array(distance), tree)
            self.alldistances[node] = (distance, tree)
            self.distancematrix[node] = distance

    def nearestneighbordistances(self, sourcepattern, destpattern=None):
        """
        Compute the interpattern nearest neighbor distances or the intrapattern
        nearest neight distances between a source pattern and a destination pattern.

        Parameters
        ----------
        sourcepattern   str The key of a point pattern snapped to the network.
        destpatter      str (Optional) The key of a point pattern snapped to the network.

        Returns
        -------
        nearest         ndarray (n,2) With column[:,0] containing the id of the nearest
                        neighbor and column [:,1] containing the distance.
        """

        if not sourcepattern in self.pointpatterns.keys():
            print "Key Error: Available point patterns are {}".format(self.pointpatterns.key())
            return

        try:
            hasattr(self.alldistances)
        except:
            self.node_distance_matrix()
        print sourcepattern
        pt_indices = self.pointpatterns[sourcepattern].points.keys()
        dist_to_node = self.pointpatterns[sourcepattern].dist_to_node
        nearest = np.zeros((len(pt_indices), 2), dtype=np.float32)
        nearest[:,1] = np.inf

        if destpattern == None:
            destpattern = sourcepattern
        obs_to_node = self.pointpatterns[destpattern].obs_to_node

        searchpts = copy.deepcopy(pt_indices)


        searchnodes = {}
        for s in searchpts:
            e1, e2 = dist_to_node[s].keys()
            searchnodes[s] = (e1, e2)

        for p1 in pt_indices:
            #Get the source nodes and dist to source nodes
            source1, source2 = searchnodes[p1]
            sdist1, sdist2 = dist_to_node[p1].values()

            searchpts.remove(p1)
            for p2 in searchpts:
                dest1, dest2 = searchnodes[p2]
                ddist1, ddist2 = dist_to_node[p2].values()
                source1_to_dest1 = sdist1 + self.alldistances[source1][0][dest1] + ddist1
                source1_to_dest2 = sdist1 + self.alldistances[source1][0][dest2] + ddist2
                source2_to_dest1 = sdist2 + self.alldistances[source2][0][dest1] + ddist1
                source2_to_dest2 = sdist2 + self.alldistances[source2][0][dest2] + ddist2


                if source1_to_dest1 < nearest[p1, 1]:
                    nearest[p1, 0] = p2
                    nearest[p1, 1] = source1_to_dest1
                if source1_to_dest1 < nearest[p2, 1]:
                    nearest[p2, 0] = p1
                    nearest[p2, 1] = source1_to_dest1

                if source1_to_dest2 < nearest[p1, 1]:
                    nearest[p1, 0] = p2
                    nearest[p1, 1] = source1_to_dest2
                if source1_to_dest1 < nearest[p2, 1]:
                    nearest[p2, 0] = p1
                    nearest[p2, 1] = source1_to_dest2

                if source2_to_dest1 < nearest[p1, 1]:
                    nearest[p1, 0] = p2
                    nearest[p1, 1] = source2_to_dest1
                if source2_to_dest1 < nearest[p2, 1]:
                    nearest[p2, 0] = p1
                    nearest[p2, 1] = source2_to_dest1

                if source2_to_dest2 < nearest[p1, 1]:
                    nearest[p1, 0] = p2
                    nearest[p1, 1] = source2_to_dest2
                if source2_to_dest2 < nearest[p2, 1]:
                    nearest[p2, 0] = p1
                    nearest[p2, 1] = source2_to_dest2

        return nearest


    def allneighbordistances(self, sourcepattern, destpattern=None):
        """
        Compute the distance between all observations points and either
         (a) all other observation points within the same set or
         (b) all other observation points from another set

        Parameters
        ----------
        sourcepattern   str The key of a point pattern snapped to the network.
        destpatter      str (Optional) The key of a point pattern snapped to the network.

        Returns
        -------
        nearest         ndarray (n,2) With column[:,0] containing the id of the nearest
                        neighbor and column [:,1] containing the distance.
        """

        try:
            hasattr(self.alldistances)
        except:
            self.node_distance_matrix()

        src_indices = sourcepattern.points.keys()
        nsource_pts = len(src_indices)
        dist_to_node = sourcepattern.dist_to_node
        if destpattern == None:
            destpattern = sourcepattern
        dest_indices = destpattern.points.keys()
        ndest_pts = len(dest_indices)

        searchpts = copy.deepcopy(dest_indices)
        nearest  = np.empty((nsource_pts, ndest_pts))
        nearest[:] = np.inf

        searchnodes = {}
        for s in searchpts:
            e1, e2 = dist_to_node[s].keys()
            searchnodes[s] = (e1, e2)

        for p1 in src_indices:
            #Get the source nodes and dist to source nodes
            source1, source2 = searchnodes[p1]
            sdist1, sdist2 = dist_to_node[p1].values()

            searchpts.remove(p1)
            for p2 in searchpts:
                dest1, dest2 = searchnodes[p2]
                ddist1, ddist2 = dist_to_node[p2].values()
                source1_to_dest1 = sdist1 + self.alldistances[source1][0][dest1] + ddist1
                source1_to_dest2 = sdist1 + self.alldistances[source1][0][dest2] + ddist2
                source2_to_dest1 = sdist2 + self.alldistances[source2][0][dest1] + ddist1
                source2_to_dest2 = sdist2 + self.alldistances[source2][0][dest2] + ddist2

                p1row = nearest[p1]
                p1col = nearest[:,p1]
                p2row = nearest[p2]
                p2col = nearest[:,p2]

                if source1_to_dest1 < nearest[p1, p2]:
                    nearest[p1, p2] = source1_to_dest1
                if source1_to_dest1 < nearest[p2, p1]:
                    nearest[p2, p1] = source1_to_dest1

                if source1_to_dest2 < nearest[p1, p2]:
                    nearest[p1, p2] = source1_to_dest2
                if source1_to_dest1 < nearest[p2, p1]:
                    nearest[p2, p1] = source1_to_dest2

                if source2_to_dest1 < nearest[p1, p2]:
                    nearest[p1, p2] = source2_to_dest1
                if source2_to_dest1 < nearest[p2, p1]:
                    nearest[p2, p2] = source2_to_dest1

                if source2_to_dest2 < nearest[p1, p2]:
                    nearest[p1, p2] = source2_to_dest2
                if source2_to_dest2 < nearest[p2, p1]:
                    nearest[p2, p1] = source2_to_dest2
        np.fill_diagonal(nearest, np.nan)
        return nearest

class PointPattern():
    def __init__(self, shapefile, idvariable=None, attribute=False):
        self.points = {}
        self.npoints = 0

        if idvariable:
            ids = get_ids(shapefile, idvariable)
        else:
            ids = None

        pts = ps.open(shapefile)

        #Get attributes if requested
        if attribute == True:
            dbname = os.path.splitext(shapefile)[0] + '.dbf'
            db = ps.open(dbname)
        else:
            db = None

        for i, pt in enumerate(pts):
            if ids and db:
                self.points[ids[i]] = {'coordinates':pt, 'properties':db[i]}
            elif ids and not db:
                self.points[ids[i]] = {'coordinates':pt, 'properties':None}
            elif not ids and db:
                self.points[i] = {'coordinates':pt, 'properties':db[i]}
            else:
                self.points[i] = {'coordinates':pt, 'properties':None}

        pts.close()
        if db:
            db.close()
        self.npoints = len(self.points.keys())


class SimulatedPointPattern():
    """
    Struct style class to mirror the Point Pattern Class.

    If the PointPattern class has methods, it might make sense to
    make this a child of that class.
    """
    def __init__(self):
        self.npoints = 0
        self.obs_to_edge = {}
        self.obs_to_node = defaultdict(list)
        self.dist_to_node = {}
        self.snapped_coordinates = {}


class SortedEdges(OrderedDict):
    def next_key(self, key):
        next = self._OrderedDict__map[key][1]
        if next is self._OrderedDict__root:
            raise ValueError("{!r} is the last key.".format(key))
        return next[2]
    def first_key(self):
        for key in self: return key
        raise ValueError("No sorted edges remain.")
