from collections import defaultdict, OrderedDict
import math
import os
import cPickle
import copy

import numpy as np
import pysal as ps
from pysal.weights.util import get_ids

from analysis import NetworkG, NetworkK, NetworkF
import util

__all__ = ["Network", "PointPattern", "NetworkG", "NetworkK", "NetworkF"  ]


class Network:

    """
    Spatially constrained network representation and analytical functionality.

    Parameters
    -----------
    in_shp : string
             input shapefile

    node_sig : int
               round the x and y coordinates of all nodes to node_sig
               significant digits (combined significant digits on left and right
               of decimal place); default is 11; set to None for no rounding

    unique_segs : boolean
                  if True (default), keep only unique segments (i.e., prune
                  out any duplicated segments); if False keep all segments

    Attributes
    ----------
    in_shp : string
             input shapefile name

    adjacencylist : list
                    of lists storing node adjacency

    nodes : dict
            key are tuple of node coords and value is the node ID

    edge_lengths : dict
                   key is a tuple of sorted node IDs representing an edge
                   value is the length

    pointpatterns : dict
                    key is a string name of the pattern
                    value is a point pattern class instance

    node_coords : dict
                  key is th node ID and value are the (x,y) coordinates
                  inverse to nodes

    edges : list
            of edges, where each edge is a sorted tuple of node IDs

    node_list : list
                node IDs

    alldistances : dict
                   key is the node ID
                   value is a tuple with two elements, first is a list of the
                   shortest path distances, second is a dict with
                   the key being the id of the destination node and the value
                   is a list of the shortest path.

    Examples
    --------

    Instantiate an instance of a network

    >>> ntw = ps.Network(ps.examples.get_path('streets.shp'))

    Snap point observations to the network with attribute information

    >>> ntw.snapobservations(ps.examples.get_path('crimes.shp'), 'crimes', attribute=True)

    And without attribute information

    >>> ntw.snapobservations(ps.examples.get_path('schools.shp'), 'schools', attribute=False)

    """

    def __init__(self, in_shp=None, node_sig=11, unique_segs=True):
        if in_shp:
            self.in_shp = in_shp
            self.node_sig = node_sig
            self.unique_segs = unique_segs

            self.adjacencylist = defaultdict(list)
            self.nodes = {}
            self.edge_lengths = {}
            self.edges = []

            self.pointpatterns = {}

            self._extractnetwork()
            self.node_coords = dict((value, key) for key, value in self.nodes.iteritems())

            #This is a spatial representation of the network.
            self.edges = sorted(self.edges)

            #Extract the graph
            self.extractgraph()

            self.node_list = sorted(self.nodes.values())

    def _round_sig(self, v):
        """
        Used internally to round vertex to a set number of significant
        digits. If sig is set to 4, then the following are some possible
        results for a coordinate: 0.0xxxx, 0.xxxx, x.xxx, xx.xx, xxx.x,
        xxxx.0, xxxx0.0
        """
        sig = self.node_sig
        if sig is None:
            return v
        out_v = [val if 0 \
                     else round(val, -int(math.floor(math.log10(math.fabs(val)))) + (sig-1)) \
                 for val in v]
        return tuple(out_v)

    def _extractnetwork(self):
        """
        Used internally, to extract a network from a polyline shapefile
        """
        nodecount = 0
        shps = ps.open(self.in_shp)
        for shp in shps:
            vertices = shp.vertices
            for i, v in enumerate(vertices[:-1]):
                v = self._round_sig(v)
                try:
                    vid = self.nodes[v]
                except:
                    self.nodes[v] = vid = nodecount
                    nodecount += 1
                v2 = self._round_sig(vertices[i+1])
                try:
                    nvid = self.nodes[v2]
                except:
                    self.nodes[v2] = nvid = nodecount
                    nodecount += 1

                self.adjacencylist[vid].append(nvid)
                self.adjacencylist[nvid].append(vid)

                #Sort the edges so that mono-directional keys can be stored.
                edgenodes = sorted([vid, nvid])
                edge = tuple(edgenodes)
                self.edges.append(edge)
                length = util.compute_length(v, vertices[i+1])
                self.edge_lengths[edge] = length
        if self.unique_segs == True:
            # remove duplicate edges and duplicate adjacent nodes
            self.edges = list(set(self.edges))
            for k, v in self.adjacencylist.iteritems():
                self.adjacencylist[k] = list(set(v))

    def extractgraph(self):
        """
        Using the existing network representation, create a graph based representation,
        by removing all nodes with neighbor incidence of two.  That is, we assume these
        nodes are bridges between nodes with higher incidence.
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
            neighbors = self._yieldneighbor(s, segment_nodes, bridge)
            while neighbors:
                cnode = neighbors.pop()
                segment_nodes.remove(cnode)
                bridge.append(cnode)
                newneighbors = self._yieldneighbor(cnode, segment_nodes, bridge)
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

    def _yieldneighbor(self, node, segment_nodes, bridge):
        """
        Used internally, this method traverses a bridge segement
        to find the source and destination nodes.
        """
        n = []
        for i in self.adjacencylist[node]:
            if i in segment_nodes and i not in bridge:
                n.append(i)
        return n

    def contiguityweights(self, graph=True, weightings=None):
        """
        Create a contiguity based W object

        Parameters
        ----------
        graph : boolean
                {True, False } controls whether the W is generated using the spatial
                representation or the graph representation

        weightings : dict
                     of lists of weightings for each edge

        Returns
        -------
         : W
           A PySAL W Object representing the binary adjacency of the network

        Examples
        --------
        >>> ntw = ps.Network(ps.examples.get_path('streets.shp'))
        >>> w = ntw.contiguityweights(graph=False)
        >>> ntw.snapobservations(ps.examples.get_path('crimes.shp'), 'crimes', attribute=True)
        >>> counts = ntw.count_per_edge(ntw.pointpatterns['crimes'].obs_to_edge, graph=False)

        Using the W object, access to ESDA functionality is provided.  First,
        a vector of attributes is created for all edges with observations.

        >>> w = ntw.contiguityweights(graph=False)
        >>> edges = w.neighbors.keys()
        >>> y = np.zeros(len(edges))
        >>> for i, e in enumerate(edges):
        ...     if e in counts.keys():
        ...         y[i] = counts[e]

        Next, a standard call ot Moran is made and the result placed into `res`

        >>> res = ps.esda.moran.Moran(y, w, permutations=99)

        """

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

        return ps.weights.W(neighbors, weights=weights)

    def distancebandweights(self, threshold):
        """
        Create distance based weights

        Parameters
        ----------
        threshold : float
                    Distance threshold value
        """
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

        return ps.weights.W(neighbors)

    def snapobservations(self, shapefile, name, idvariable=None, attribute=None):
        """
        Snap a point pattern shapefile to this network object.  The point pattern
        is the stored in the network.pointpattern['key'] attribute of the network
        object.

        Parameters
        ----------
        shapefile : str
                    The PATH to the shapefile

        name : str
               Name to be assigned to the point dataset

        idvariable : str
                     Column name to be used as ID variable

        attribute : bool
                    Defines whether attributes should be extracted

        Returns
        -------

        """

        self.pointpatterns[name] = PointPattern(shapefile, idvariable=idvariable, attribute=attribute)
        self._snap_to_edge(self.pointpatterns[name])

    def compute_distance_to_nodes(self, x, y, edge):
        """
        Given an observation on a network edge, return the distance to the two
        nodes that bound that end.

        Parameters
        ----------
        x : float
            x-coordinate of the snapped point

        y : float
            y-coordiante of the snapped point

        edge : tuple
              (node0, node1) representation of the network edge

        Returns
        -------
        d1 : float
             the distance to node0, always the node with the lesser id

        d2 : float
             the distance to node1, always the node with the greater id
        """

        d1 = util.compute_length((x,y), self.node_coords[edge[0]])
        d2 = util.compute_length((x,y), self.node_coords[edge[1]])
        return d1, d2

    def _snap_to_edge(self, pointpattern):
        """
        Used internally to snap point observations to network edges.

        Parameters
        -----------
        pointpattern : obj
                       PySAL Point Pattern Object

        Returns
        -------
        obs_to_edge : dict
                      with edge as key and list of points as value

        edge_to_obs : dict
                      with point id as key and edge tuple as value

        dist_to_node : dict
                       with point id as key and value as a dict with key for
                       node id, and value distance from point to node

        """

        obs_to_edge = {}
        dist_to_node = {}

        pointpattern.snapped_coordinates = {}
        segments = []
        s2e = {}
        for edge in self.edges:
            head = self.node_coords[edge[0]]
            tail = self.node_coords[edge[1]]
            segments.append(ps.cg.Chain([head,tail]))
            s2e[(head,tail)] = edge
            

        points = {}
        p2id = {}
        for pointIdx, point in pointpattern.points.iteritems(): 
            points[pointIdx] = point['coordinates']

        snapped = util.snapPointsOnSegments(points, segments)

        for pointIdx, snapInfo in snapped.iteritems():
            x,y = snapInfo[1].tolist()
            edge = s2e[tuple(snapInfo[0])]
            if edge not in obs_to_edge:
                obs_to_edge[edge] = {}
            obs_to_edge[edge][pointIdx] = (x,y)
            pointpattern.snapped_coordinates[pointIdx] = (x,y)
            d1,d2 = self.compute_distance_to_nodes(x, y, edge)
            dist_to_node[pointIdx] = {edge[0]:d1, edge[1]:d2}

        obs_to_node = defaultdict(list)
        for k, v in obs_to_edge.iteritems():
            keys = v.keys()
            obs_to_node[k[0]] = keys
            obs_to_node[k[1]] = keys

        pointpattern.obs_to_edge = obs_to_edge
        pointpattern.dist_to_node = dist_to_node
        pointpattern.obs_to_node = obs_to_node


    def count_per_edge(self, obs_on_network, graph=True):
        """
        Compute the counts per edge.

        Parameters
        ----------
        obs_on_network : dict
                         of observations on the network
                         {(edge): {pt_id: (coords)}} or {edge: [(coord), (coord), (coord)]}
        Returns
        -------
        counts: dict {(edge):count}

        Example
        -------

        Note that this passes the obs_to_edge attribute of a point pattern
        snapped to the network.

        >>> ntw = ps.Network(ps.examples.get_path('streets.shp'))
        >>> ntw.snapobservations(ps.examples.get_path('crimes.shp'), 'crimes', attribute=True)
        >>> counts = ntw.count_per_edge(ntw.pointpatterns['crimes'].obs_to_edge,graph=False)
        >>> s = sum([v for v in counts.itervalues()])
        >>> s
        287
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
        """
        Used internally to compute new point coordinates during snapping
        """
        x1 = self.node_coords[edge[0]][0]
        y1 = self.node_coords[edge[0]][1]
        x2 = self.node_coords[edge[1]][0]
        y2 = self.node_coords[edge[1]][1]
        if x1 == x2:  # vertical line case
            x0 = x1
            if y1 < y2:
                y0 = y1 + distance
            elif y1 > y2:
                y0 = y2 + distance
            else:    # zero length edge
                y0 = y1
            return x0, y0        
        m = (y2 - y1) / (x2 - x1)
        if x1 > x2:
            x0 = x1 - distance / math.sqrt(1 + m**2)
        elif x1 < x2:
            x0 = x1 + distance / math.sqrt(1 + m**2)
        y0 = m * (x0 - x1) + y1
        return x0, y0

    def simulate_observations(self, count, distribution='uniform'):
        """
        Generate a simulated point pattern on the network.

        Parameters
        ----------
        count : integer
                number of points to create or mean of the distribution
                if not 'uniform'

        distribution : string
                       {'uniform', 'poisson'} distribution of random points

        Returns
        -------
        random_pts : dict
                     key is the edge tuple
                     value is a list of new point coordinates

        Example
        -------

        >>> ntw = ps.Network(ps.examples.get_path('streets.shp'))
        >>> ntw.snapobservations(ps.examples.get_path('crimes.shp'), 'crimes', attribute=True)
        >>> npts = ntw.pointpatterns['crimes'].npoints
        >>> sim = ntw.simulate_observations(npts)
        >>> isinstance(sim, ps.network.network.SimulatedPointPattern)
        True
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
            simpts.npoints = len(simpts.points)

        return simpts

    def enum_links_node(self, v0):
        """
        Returns the edges (links) around node

        Parameters
        -----------
        v0 : int
             node id

        Returns
        -------
        links : list
                list of tuple edge adjacent to the node
        """
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
            #tree = util.generatetree(pred)
            tree = None
            self.alldistances[node] = (distance, tree)
            self.distancematrix[node] = distance

    def allneighbordistances(self, sourcepattern, destpattern=None):
        """
        Compute either all distances between i and j in a single point pattern
        or all distances between each i from a source pattern and all j
        from a destination pattern

        Parameters
        ----------
        sourcepattern : str
                        The key of a point pattern snapped to the network.

        destpattern :str
                    (Optional) The key of a point pattern snapped to the network.

        Returns
        -------
        nearest : array (n,n)
                  An array or shape n,n storing distances between all points

        """

        if not hasattr(self,'alldistances'):
            self.node_distance_matrix()
            
        # source setup
        src_indices = sourcepattern.points.keys()
        nsource_pts = len(src_indices)
        src_dist_to_node = sourcepattern.dist_to_node
        src_nodes = {}
        for s in src_indices:
            e1, e2 = src_dist_to_node[s].keys()
            src_nodes[s] = (e1, e2)

        # destination setup
        symmetric = False
        if destpattern is None:
            symmetric = True
            destpattern = sourcepattern
        dest_indices = destpattern.points.keys()
        ndest_pts = len(dest_indices)
        dest_dist_to_node = destpattern.dist_to_node
        dest_searchpts = copy.deepcopy(dest_indices)
        dest_nodes = {}
        for s in dest_indices:
            e1, e2 = dest_dist_to_node[s].keys()
            dest_nodes[s] = (e1, e2)
        
        # output setup
        nearest = np.empty((nsource_pts, ndest_pts))
        nearest[:] = np.inf

        for p1 in src_indices:
            #Get the source nodes and dist to source nodes
            source1, source2 = src_nodes[p1]
            set1 = set(src_nodes[p1])
            # distance from node1 to p, distance from node2 to p
            sdist1, sdist2 = src_dist_to_node[p1].values()

            if symmetric:
                # only compute the upper triangle if symmetric
                dest_searchpts.remove(p1)
            for p2 in dest_searchpts:
                dest1, dest2 = dest_nodes[p2]
                set2 = set(dest_nodes[p2])
                if set1 == set2: #same edge
                    x1,y1 = sourcepattern.snapped_coordinates[p1]
                    x2,y2 = destpattern.snapped_coordinates[p2]
                    xd = x1-x2
                    yd = y1-y2
                    nearest[p1,p2] = np.sqrt(xd*xd + yd*yd)

                else:
                    ddist1, ddist2 = dest_dist_to_node[p2].values()
                    d11 = self.alldistances[source1][0][dest1]
                    d21 = self.alldistances[source2][0][dest1]
                    d12 = self.alldistances[source1][0][dest2]
                    d22 = self.alldistances[source2][0][dest2]

                    # find shortest distance from path passing through each of two origin nodes
                    # to first destination node
                    sd_1 = d11 + sdist1
                    sd_21 = d21 + sdist2
                    if sd_1 > sd_21:
                        sd_1 = sd_21
                    # now add point to node one distance on destination edge
                    len_1 = sd_1 + ddist1

                    # repeat but now for paths entering at second node of second edge
                    sd_2 = d12 + sdist1
                    sd_22 = d22 + sdist2
                    b = 0
                    if sd_2 > sd_22:
                        sd_2 = sd_22
                        b = 1
                    len_2 = sd_2 + ddist2

                    # now find shortest length path between the point 1 on edge 1 and
                    # point 2 on edge 2, and assign
                    sp_12 = len_1
                    if len_1 > len_2:
                        sp_12 = len_2
                    nearest[p1, p2] = sp_12
                if symmetric:
                    # mirror the upper and lower triangle when symmetric
                    nearest[p2,p1] = nearest[p1,p2]                    
        if symmetric:
            # populate the main diagonal when symmetric
            #np.fill_diagonal(nearest, 0)
            np.fill_diagonal(nearest, np.nan)
        return nearest

    def nearestneighbordistances(self, sourcepattern, destpattern=None):
        """
        Compute the interpattern nearest neighbor distances or the intrapattern
        nearest neight distances between a source pattern and a destination pattern.

        Parameters
        ----------
        sourcepattern   str The key of a point pattern snapped to the network.

        destpattern      str (Optional) The key of a point pattern snapped to the network.

        Returns
        -------
        nearest         ndarray (n,2) With column[:,0] containing the id of the nearest
                        neighbor and column [:,1] containing the distance.
        """

        if not sourcepattern in self.pointpatterns.keys():
            raise KeyError("Available point patterns are {}".format(self.pointpatterns.keys()))

        if not hasattr(self,'alldistances'):
            self.node_distance_matrix()

        pt_indices = self.pointpatterns[sourcepattern].points.keys()
        dist_to_node = self.pointpatterns[sourcepattern].dist_to_node
        nearest = np.zeros((len(pt_indices), 2), dtype=np.float32)
        nearest[:,1] = np.inf

        if destpattern == None:
            destpattern = sourcepattern

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

    def NetworkF(self, pointpattern, nsteps=10, permutations=99,
                 threshold=0.2, distribution='uniform',
                 lowerbound=None, upperbound=None):
        """
        Computes a network constrained F-Function

        Parameters
        ----------
        pointpattern : object
                       A PySAL point pattern object

        nsteps : int
                 The number of steps at which the count of the nearest
                 neighbors is computed

        permutations : int
                       The number of permutations to perform (default 99)

        threshold : float
                    The level at which significance is computed.  0.5 would be 97.5% and 2.5%

        distribution : str
                       The distirbution from which random points are sampled: uniform or poisson

        lowerbound : float
                     The lower bound at which the G-function is computed. (default 0)

        upperbound : float
                     The upper bound at which the G-function is computed.
                     Defaults to the maximum pbserved nearest neighbor distance.

        Returns
        -------
        NetworkF : object
                   A network F class instance

        """
        return NetworkF(self, pointpattern, nsteps=nsteps,
                        permutations=permutations,threshold=threshold,
                        distribution=distribution,lowerbound=lowerbound,
                        upperbound=upperbound)

    def NetworkG(self, pointpattern, nsteps=10, permutations=99,
                 threshold=0.5, distribution='uniform',
                 lowerbound=None, upperbound=None):
        """
        Computes a network constrained G-Function

        Parameters
        ----------
        pointpattern : object
                       A PySAL point pattern object

        nsteps : int
                 The number of steps at which the count of the nearest
                 neighbors is computed

        permutations : int
                       The number of permutations to perform (default 99)

        threshold : float
                    The level at which significance is computed.  0.5 would be 97.5% and 2.5%

        distribution : str
                       The distirbution from which random points are sampled: uniform or poisson

        lowerbound : float
                     The lower bound at which the G-function is computed. (default 0)

        upperbound : float
                     The upper bound at which the G-function is computed.
                     Defaults to the maximum pbserved nearest neighbor distance.

        Returns
        -------
        NetworkG : object
                   A network G class object

        """

        return NetworkG(self, pointpattern, nsteps=nsteps,
                        permutations=permutations,threshold=threshold,
                        distribution=distribution,lowerbound=lowerbound,
                        upperbound=upperbound)

    def NetworkK(self, pointpattern, nsteps=10, permutations=99,
                 threshold=0.5, distribution='uniform',
                 lowerbound=None, upperbound=None):
        """
        Computes a network constrained G-Function

        Parameters
        ----------
        pointpattern : object
                       A PySAL point pattern object

        nsteps : int
                 The number of steps at which the count of the nearest
                 neighbors is computed

        permutations : int
                       The number of permutations to perform (default 99)

        threshold : float
                    The level at which significance is computed.  0.5 would be 97.5% and 2.5%

        distribution : str
                       The distirbution from which random points are sampled: uniform or poisson

        lowerbound : float
                     The lower bound at which the G-function is computed. (default 0)

        upperbound : float
                     The upper bound at which the G-function is computed.
                     Defaults to the maximum pbserved nearest neighbor distance.

        Returns
        -------
        NetworkK : object
                   A network K class object

        """
        return NetworkK(self, pointpattern, nsteps=nsteps,
                        permutations=permutations,threshold=threshold,
                        distribution=distribution,lowerbound=lowerbound,
                        upperbound=upperbound)

    def segment_edges(self, distance):
        """
        Segment all of the edges in the network at either
        a fixed distance or a fixed number of segments.

        Parameters
        -----------
        distance : float
                   The distance at which edges are split

        Returns
        -------
        sn : object
             PySAL Network Object

        Example
        -------

        >>> ntw = ps.Network(ps.examples.get_path('streets.shp'))
        >>> n200 = ntw.segment_edges(200.0)
        >>> len(n200.edges)
        688
        """

        sn = Network()
        sn.adjacencylist = copy.deepcopy(self.adjacencylist)
        sn.edge_lengths = copy.deepcopy(self.edge_lengths)
        sn.edges = set(copy.deepcopy(self.edges))
        sn.node_coords = copy.deepcopy(self.node_coords)
        sn.node_list = copy.deepcopy(self.node_list)
        sn.nodes = copy.deepcopy(self.nodes)
        sn.pointpatterns = copy.deepcopy(self.pointpatterns)
        sn.in_shp = self.in_shp

        current_node_id = max(self.nodes.values())

        newedges = set()
        removeedges = set()
        for e in sn.edges:
            length = sn.edge_lengths[e]
            interval = distance

            totallength = 0
            currentstart = startnode = e[0]
            endnode = e[1]

            #If the edge will be segmented, remove the
            # current edge from the adjacency list
            if interval < length:
                sn.adjacencylist[e[0]].remove(e[1])
                sn.adjacencylist[e[1]].remove(e[0])
                sn.edge_lengths.pop(e, None)
                removeedges.add(e)
            else:
                continue

            while totallength < length:
                currentstop = current_node_id
                if totallength + interval > length:
                    currentstop = endnode
                    interval = length - totallength
                    totallength = length
                else:
                    current_node_id += 1
                    currentstop = current_node_id
                    totallength += interval

                    #Compute the new node coordinate
                    newx, newy = self._newpoint_coords(e, totallength)

                    #Update node_list
                    if currentstop not in sn.node_list:
                        sn.node_list.append(currentstop)

                    #Update nodes and node_coords
                    sn.node_coords[currentstop] = newx, newy
                    sn.nodes[(newx, newy)] = currentstop

                #Update the adjacencylist
                sn.adjacencylist[currentstart].append(currentstop)
                sn.adjacencylist[currentstop].append(currentstart)


                #Add the new edge to the edge dict
                #Iterating over this, so we need to add after iterating
                newedges.add(tuple(sorted([currentstart, currentstop])))

                #Modify edge_lengths
                sn.edge_lengths[tuple(sorted([currentstart, currentstop]))] = interval

                #Increment the start to the stop
                currentstart = currentstop

        sn.edges.update(newedges)
        sn.edges.difference_update(removeedges)
        sn.edges = list(sn.edges)
        #Update the point pattern snapping
        for instance in sn.pointpatterns.itervalues():
            sn._snap_to_edge(instance)

        return sn

    def savenetwork(self, filename):
        """
        Save a network to disk as a binary file

        Parameters
        ----------
        filename : str
                   The filename where the network should be saved.
                   This should be a full PATH or the file is saved
                   whereever this method is called from.

        Example
        --------
        >>> ntw = ps.Network(ps.examples.get_path('streets.shp'))
        >>> ntw.savenetwork('mynetwork.pkl')

        """
        with open(filename, 'wb') as networkout:
            cPickle.dump(self, networkout, protocol=2)

    @staticmethod
    def loadnetwork(filename):
        with open(filename, 'rb') as networkin:
            self = cPickle.load(networkin)

        return self


class PointPattern():
    """
    A stub point pattern class used to store a point pattern.
    This class is monkey patched with network specific attributes
    when the points are snapped to a network.

    In the future this class may be replaced with a generic point
    pattern class.

    Parameters
    ----------
    shapefile : string
                input shapefile

    idvariable : string
                 field in the shapefile to use as an idvariable

    attribute : boolean
                {False, True} A flag to indicate whether all attributes
                are tagged to this class.

    Attributes
    ----------
    points : dict
             key is the point id
             value are the coordinates

    npoints : integer
              the number of points


    """
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

    This class is not intended to be used by the external user.

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

