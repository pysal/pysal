from collections import defaultdict, OrderedDict
import os
import pickle
import copy
import numpy as np
from .analysis import NetworkG, NetworkK, NetworkF
from . import util
from libpysal import cg, examples, weights
try:
    from libpysal import open
except ImportError:
    import libpysal
    open = libpysal.io.open

__all__ = ["Network", "PointPattern", "NetworkG", "NetworkK", "NetworkF"]


class Network:
    """Spatially-constrained network representation
    and analytical functionality.
    
    Parameters
    ----------
    
    in_data : geopandas.GeoDataFrame or str
        The input geographic data. Either (1) a path to a shapefile (str);
        or (2) a geopandas.GeoDataFrame.
    
    node_sig : int
        Round the x and y coordinates of all nodes to node_sig significant
        digits (combined significant digits on the left and right of the
        decimal place). Default is 11. Set to None for no rounding.
    
    unique_segs : bool
        If True (default), keep only unique segments (i.e., prune out any
        duplicated segments). If False keep all segments.
    
    extractgraph : bool
        If True, extract a graph-theoretic object with no degree 2 nodes.
        Default is True.
    
    Attributes
    ----------
    
    in_data : str
        The input shapefile. This must be in .shp format.
    
    adjacencylist : list
        List of lists storing node adjacency.
    
    nodes : dict
        Keys are tuples of node coords and values are the node ID.
    
    edge_lengths : dict
        Keys are tuples of sorted node IDs representing an edge and values
        are the length.
    
    pointpatterns : dict
        Keys are a string name of the pattern and values are point pattern
        class instances.
    
    node_coords : dict
        Keys are the node ID and values are the (x,y) coordinates
        inverse to nodes.
    
    edges : list
        List of edges, where each edge is a sorted tuple of node IDs.
    
    node_list : list
        List of node IDs.
    
    alldistances : dict
        Keys are the node IDs (int). Values are tuples with two elements as
        follows (1) a list of the shortest path distances; (2) a dict with the
        key being the id of the destination node and the value being a list of
        the shortest path.
    
    distancematrix : numpy.ndarray
        all network nodes (non-observations) distance matrix.
    
    graphedges : list
        tuples of graph edge ids.
    
    graph_lengths : dict
        Keys are the graph edge ids (tuple). Values are the graph edge lenght
        (float).
    
    Examples
    --------
    
    Instantiate an instance of a network.
    
    >>> import spaghetti as spgh
    >>> streets_file = examples.get_path('streets.shp')
    >>> ntw = spgh.Network(in_data=streets_file)
    
    Snap point observations to the network with attribute information.
    
    >>> crimes_file = examples.get_path('crimes.shp')
    >>> ntw.snapobservations(crimes_file, 'crimes', attribute=True)
   
    And without attribute information.
   
    >>> schools_file = examples.get_path('schools.shp')
    >>> ntw.snapobservations(schools_file, 'schools', attribute=False)
    
    """
    
    def __init__(self, in_data=None, node_sig=11,
                 unique_segs=True, extractgraph=True):
        """
        """
        if in_data is not None:
            self.in_data = in_data
            self.node_sig = node_sig
            self.unique_segs = unique_segs
            
            self.adjacencylist = defaultdict(list)
            self.nodes = {}
            self.edge_lengths = {}
            self.edges = []
            
            self.pointpatterns = {}
            
            self._extractnetwork()
            self.node_coords = dict((v, k) for k, v in self.nodes.items())
            
            # This is a spatial representation of the network.
            self.edges = sorted(self.edges)
            
            # Extract the graph.
            if extractgraph:
                self.extractgraph()
                
            self.node_list = sorted(self.nodes.values())


    def _round_sig(self, v):
        """Used internally to round the vertex to a set number of significant
        digits. If sig is set to 4, then the following are some possible
        results for a coordinate are as follows.
            (1) 0.0xxxx, (2) 0.xxxx, (3) x.xxx, (4) xx.xx,
            (5) xxx.x, (6) xxxx.0, (7) xxxx0.0
        """
        sig = self.node_sig
        if sig is None:
            return v
        out_v = [val if 0
                 else round(val, -int(np.floor(np.log10(np.fabs(val)))) +
                            (sig - 1))
                 for val in v]
        
        return tuple(out_v)


    def _extractnetwork(self):
        """Used internally to extract a network from a polyline shapefile.
        """
        nodecount = 0
        if isinstance(self.in_data, str):
            shps = open(self.in_data)
        else:
            shps = self.in_data.geometry
        for shp in shps:
            vertices = weights._contW_lists._get_verts(shp)
            for i, v in enumerate(vertices[:-1]):
                v = self._round_sig(v)
                try:
                    vid = self.nodes[v]
                except KeyError:
                    self.nodes[v] = vid = nodecount
                    nodecount += 1
                v2 = self._round_sig(vertices[i + 1])
                try:
                    nvid = self.nodes[v2]
                except KeyError:
                    self.nodes[v2] = nvid = nodecount
                    nodecount += 1
                    
                self.adjacencylist[vid].append(nvid)
                self.adjacencylist[nvid].append(vid)
                
                # Sort the edges so that mono-directional keys can be stored.
                edgenodes = sorted([vid, nvid])
                edge = tuple(edgenodes)
                self.edges.append(edge)
                length = util.compute_length(v, vertices[i + 1])
                self.edge_lengths[edge] = length
        if self.unique_segs:
            # Remove duplicate edges and duplicate adjacent nodes.
            self.edges = list(set(self.edges))
            for k, v in self.adjacencylist.items():
                self.adjacencylist[k] = list(set(v))


    def extractgraph(self):
        """Using the existing network representation, create a graph-theoretic
        representation by removing all nodes with a neighbor incidence of two 
        (non-articulation points). That is, we assume these nodes are bridges
        between nodes with higher incidence.
        """
        self.graphedges = []
        self.graph_lengths = {}
        
        # Find all nodes with cardinality 2. (non-articulation points)
        segment_nodes = []
        for k, v in self.adjacencylist.items():
            # len(v) == 1 #cul-de-sac
            # len(v) == 2 #bridge segment
            # len(v) > 2 #intersection
            if len(v) == 2:
                segment_nodes.append(k)
                
        # Start with a copy of the spatial representation and
        # iteratively remove edges deemed to be segments.
        self.graphedges = copy.deepcopy(self.edges)
        self.graph_lengths = copy.deepcopy(self.edge_lengths)
        # Mapping all the 'network edges' contained within a single
        # 'graph represented' edge.
        self.graph_to_edges = {}
        
        # build up bridges "rooted" on the initial non-articulation points
        bridges = []
        for s in segment_nodes:
            bridge = [s]
            neighbors = self._yieldneighbor(s, segment_nodes, bridge)
            while neighbors:
                cnode = neighbors.pop()
                segment_nodes.remove(cnode)
                bridge.append(cnode)
                newneighbors = self._yieldneighbor(cnode,
                                                   segment_nodes,
                                                   bridge)
                neighbors += newneighbors
            bridges.append(bridge)
            
        for bridge in bridges:
            if len(bridge) == 1:
                n = self.adjacencylist[bridge[0]]
                newedge = tuple(sorted([n[0], n[1]]))
                # Identify the edges to be removed.
                e1 = tuple(sorted([bridge[0], n[0]]))
                e2 = tuple(sorted([bridge[0], n[1]]))
                # Remove them from the graph.
                self.graphedges.remove(e1)
                self.graphedges.remove(e2)
                # Remove from the edge lengths.
                length_e1 = self.edge_lengths[e1]
                length_e2 = self.edge_lengths[e2]
                self.graph_lengths.pop(e1, None)
                self.graph_lengths.pop(e2, None)
                self.graph_lengths[newedge] = length_e1 + length_e2
                # Update the pointers.
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
                            redundant.add(tuple(sorted([b, n])))
                            
                newedge = tuple(sorted(startend.values()))
                for k, v in startend.items():
                    redundant.add(tuple(sorted([k, v])))
                    
                for r in redundant:
                    self.graphedges.remove(r)
                    cumulative_length += self.edge_lengths[r]
                    self.graph_lengths.pop(r, None)
                    self.graph_to_edges[r] = newedge
                self.graph_lengths[newedge] = cumulative_length
                
            self.graphedges.append(newedge)
        self.graphedges = sorted(self.graphedges)


    def _yieldneighbor(self, node, segment_nodes, bridge):
        """Used internally, this method traverses a bridge segement to find
        the source and destination nodes.
        
        Parameters
        ----------
        
        node : int
            node id
        
        segment_nodes : list
            all non-articulation points in the network (degree-2 nodes).
        
        bridge : list
            inital bridge list containing only `node`
        
        Returns
        -------
        
        n : list
            nodes to keep (articulation points)
        
        """
        n = []
        # get all nodes adjacent to `node`
        for i in self.adjacencylist[node]:
            if i in segment_nodes and i not in bridge:
                n.append(i)
                
        return n


    def contiguityweights(self, graph=True, weightings=None):
        """Create a contiguity based W object.
        
        Parameters
        ----------
        
        graph : bool
            {True, False} controls whether the W is generated using the spatial
            representation or the graph representation.
        
        weightings : dict
            Dict of lists of weightings for each edge.
        
        Returns
        -------
        
         W : libpysal.weights.weights.W
            A PySAL W Object representing the binary adjacency of the network.
        
        Examples
        --------
        
        Instantiate an instance of a network.
        
        >>> import spaghetti as spgh
        >>> from libpysal import examples
        >>> import esda
        >>> import numpy as np
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        
        Snap point observations to the network with attribute information.
        
        >>> ntw.snapobservations(examples.get_path('crimes.shp'),
        ...                      'crimes', attribute=True)
        
        Find counts per netowrk edge.
        
        >>> counts = ntw.count_per_edge(ntw.pointpatterns['crimes']
        ...                             .obs_to_edge, graph=False)
        >>> counts[(50, 165)]
        4
        
        Create a contiguity based W object.
        
        >>> w = ntw.contiguityweights(graph=False)
        
        Using the W object, access to ESDA functionality is provided.
        First, a vector of attributes is created for all edges
        with observations.
        
        >>> w = ntw.contiguityweights(graph=False)
        >>> edges = w.neighbors.keys()
        >>> y = np.zeros(len(edges))
        >>> for i, e in enumerate(edges):
        ...     if e in counts.keys():
        ...         y[i] = counts[e]
        >>> y[3]
        3.0
        
        Next, a standard call ot Moran is made and the
        result placed into `res`.
        
        >>> res = esda.moran.Moran(y, w, permutations=99)
        >>> type(res)
        <class 'esda.moran.Moran'>
        
        """
        
        neighbors = {}
        neighbors = OrderedDict()
        
        if graph:
            edges = self.graphedges
        else:
            edges = self.edges
            
        if weightings:
            _weights = {}
        else:
            _weights = None
            
        for key in edges:
            neighbors[key] = []
            if weightings:
                _weights[key] = []
                
            for neigh in edges:
                if key == neigh:
                    continue
                if key[0] == neigh[0] or key[0] == neigh[1]\
                        or key[1] == neigh[0] or key[1] == neigh[1]:
                    neighbors[key].append(neigh)
                    if weightings:
                        _weights[key].append(weightings[neigh])
                # TODO: Add a break condition - everything is sorted,
                #       so we know when we have stepped beyond
                #       a possible neighbor.
                # if key[1] > neigh[1]:  #NOT THIS
                    # break
        w = weights.W(neighbors, weights=_weights)
        
        return w


    def distancebandweights(self, threshold, n_proccess=None, gen_tree=False):
        """Create distance based weights.
        
        Parameters
        ----------
        
        threshold : float
            Distance threshold value.
        
        n_processes : int, str
            (Optional) Specify the number of cores to utilize. Default is 1
            core. Use (int) to specify an exact number or cores. Use ("all")
            to request all available cores.
        
        gen_tree : bool
            rebuild shortest path {True}, or skip {False}
        
        Returns
        -------
        
        w : libpysal.weights.weights.W
            A PySAL W Object representing the binary adjacency of the network.
        
        Examples
        --------
        
        >>> import spaghetti as spgh
        >>> streets_file = examples.get_path('streets.shp')
        >>> ntw = spgh.Network(in_data=streets_file)
        >>> w = ntw.distancebandweights(threshold=500)
        >>> w.n
        230
        >>> w.histogram
        [(1, 22), (2, 58), (3, 63), (4, 40), (5, 36), (6, 3), (7, 5), (8, 3)]
        
        """
        try:
            hasattr(self.alldistances)
        except AttributeError:
            self.node_distance_matrix(n_proccess, gen_tree=gen_tree)
            
        neighbor_query = np.where(self.distancematrix < threshold)
        neighbors = defaultdict(list)
        for i, n in enumerate(neighbor_query[0]):
            neigh = neighbor_query[1][i]
            if n != neigh:
                neighbors[n].append(neighbor_query[1][i])
        w = weights.W(neighbors)
        
        return w


    def snapobservations(self, in_data, name,
                         idvariable=None, attribute=None):
        """Snap a point pattern shapefile to this network object. The point
        pattern is stored in the network.pointpattern['key'] attribute of
        the network object.
        
        Parameters
        ----------
        
        in_data : geopandas.GeoDataFrame or str
            The input geographic data. Either (1) a path to a shapefile (str);
            or (2) a geopandas.GeoDataFrame.
        
        name : str
            Name to be assigned to the point dataset.
        
        idvariable : str
            Column name to be used as ID variable.
        
        attribute : bool
            Defines whether attributes should be extracted. True for attribute
            extraction. False for no attribute extraaction.
        
        Examples
        --------
        
        >>> import spaghetti as spgh
        >>> streets_file = examples.get_path('streets.shp')
        >>> ntw = spgh.Network(in_data=streets_file)
        >>> pt_str = 'crimes'
        >>> in_data = examples.get_path('{}.shp'.format(pt_str))
        >>> ntw.snapobservations(in_data, pt_str, attribute=True)
        >>> ntw.pointpatterns[pt_str].npoints
        287
        
        """
        
        self.pointpatterns[name] = PointPattern(in_data=in_data,
                                                idvariable=idvariable,
                                                attribute=attribute)
        self._snap_to_edge(self.pointpatterns[name])


    def compute_distance_to_nodes(self, x, y, edge):
        """Given an observation on a network edge, return the distance to
        the two nodes that bound that end.
        
        Parameters
        ----------
        
        x : float
            x-coordinate of the snapped point.
        
        y : float
            y-coordiante of the snapped point.
        
        edge : tuple
            (node0, node1) representation of the network edge.
        
        Returns
        -------
        
        d1 : float
            The distance to node0. Always the node with the lesser id.
        
        d2 : float
            The distance to node1. Always the node with the greater id.
        
        """
        
        d1 = util.compute_length((x, y), self.node_coords[edge[0]])
        d2 = util.compute_length((x, y), self.node_coords[edge[1]])
        
        return d1, d2

    def compute_snap_dist(self, pattern, idx):
        """Given an observation snapped to a network edge, calculate the
        distance from the original location to the snapped location.
        
        Parameters
        -----------
        
        pattern : spaghetti.network.PointPattern
            point pattern object
        idx : int
            point id
        
        Return
        ------
        dist : float
            euclidean distance from original location to snapped location.
        """
        loc = pattern.points[idx]['coordinates']
        snp = pattern.snapped_coordinates[idx]
        dist = util.compute_length(loc, snp)
        return dist

    def _snap_to_edge(self, pointpattern):
        """ Used internally to snap point observations to network edges.
        
        Parameters
        -----------
        
        pointpattern : spaghetti.network.PointPattern
            point pattern object
        
        Returns
        -------
        
        obs_to_edge : dict
            Dict with edges as keys and lists of points as values.
        
        edge_to_obs : dict
            Dict with point ids as keys and edge tuples as values.
        
        dist_to_node : dict
            Dict with point ids as keys and values as dicts with keys for node
            ids and values as distances from point to node.
        
        """
        
        obs_to_edge = {}
        dist_to_node = {}
        dist_snapped = {}
        obs_to_node = defaultdict(list)
        
        pointpattern.snapped_coordinates = {}
        segments = []
        s2e = {}
        for edge in self.edges:
            head = self.node_coords[edge[0]]
            tail = self.node_coords[edge[1]]
            segments.append(cg.Chain([head, tail]))
            s2e[(head, tail)] = edge
        
        # snap points
        points = {}
        p2id = {}
        for pointIdx, point in pointpattern.points.items():
            points[pointIdx] = point['coordinates']
        snapped = util.snap_points_on_segments(points, segments)
        
        # record obs_to_edge, dist_to_node, and dist_snapped
        for pointIdx, snapInfo in snapped.items():
            x, y = snapInfo[1].tolist()
            edge = s2e[tuple(snapInfo[0])]
            if edge not in obs_to_edge:
                obs_to_edge[edge] = {}
            obs_to_edge[edge][pointIdx] = (x, y)
            pointpattern.snapped_coordinates[pointIdx] = (x, y)
            d1, d2 = self.compute_distance_to_nodes(x, y, edge)
            dist_to_node[pointIdx] = {edge[0]: d1, edge[1]: d2}
            dist_snapped[pointIdx] = self.compute_snap_dist(pointpattern,
                                                            pointIdx)
        
        # record obs_to_node
        obs_to_node = defaultdict(list)
        for k, v in obs_to_edge.items():
            keys = v.keys()
            obs_to_node[k[0]] = keys
            obs_to_node[k[1]] = keys
        
        pointpattern.obs_to_edge = obs_to_edge
        pointpattern.dist_to_node = dist_to_node
        pointpattern.dist_snapped = dist_snapped
        pointpattern.obs_to_node = list(obs_to_node)


    def count_per_edge(self, obs_on_network, graph=True):
        """Compute the counts per edge.
        
        Parameters
        ----------
        
        obs_on_network : dict
            Dict of observations on the network. {(edge):{pt_id:(coords)}} or
            {edge:[(coord),(coord),(coord)]}
        
        Returns
        -------
        counts : dict
            {(edge):count}
        
        Examples
        --------
        
        Note that this passes the obs_to_edge attribute of a
        point pattern snapped to the network.
        
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.snapobservations(examples.get_path('crimes.shp'),
        ...                                           'crimes',
        ...                                           attribute=True)
        
        >>> counts = ntw.count_per_edge(ntw.pointpatterns['crimes']
        ...                             .obs_to_edge, graph=False)
        >>> counts[(140, 142)]
        10
        
        >>> s = sum([v for v in list(counts.values())])
        >>> s
        287
        
        """
        counts = {}
        if graph:
            for key, observations in obs_on_network.items():
                cnt = len(observations)
                if key in self.graph_to_edges.keys():
                    key = self.graph_to_edges[key]
                try:
                    counts[key] += cnt
                except KeyError:
                    counts[key] = cnt
        else:
            for key in obs_on_network.keys():
                counts[key] = len(obs_on_network[key])
        return counts


    def _newpoint_coords(self, edge, distance):
        """ Used internally to compute new point coordinates during snapping.
        """
        x1 = self.node_coords[edge[0]][0]
        y1 = self.node_coords[edge[0]][1]
        x2 = self.node_coords[edge[1]][0]
        y2 = self.node_coords[edge[1]][1]
        if x1 == x2:  # Vertical line case
            x0 = x1
            if y1 < y2:
                y0 = y1 + distance
            elif y1 > y2:
                y0 = y2 + distance
            else:    # Zero length edge
                y0 = y1
            return x0, y0
        m = (y2 - y1) / (x2 - x1)
        if x1 > x2:
            x0 = x1 - distance / np.sqrt(1 + m**2)
        elif x1 < x2:
            x0 = x1 + distance / np.sqrt(1 + m**2)
        y0 = m * (x0 - x1) + y1
        
        return x0, y0


    def simulate_observations(self, count, distribution='uniform'):
        """ Generate a simulated point pattern on the network.
        
        Parameters
        ----------
        
        count : int
            The number of points to create or mean of the distribution
            if not 'uniform'.
        
        distribution : str
            {'uniform', 'poisson'} distribution of random points. If poisson
            if the distrubtion the poisson is calculated from half the total
            network length
        
        Returns
        -------
        
        random_pts : dict
            Keys are the edge tuple. Values are lists of new point coordinates.
        
        Examples
        --------
       
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.snapobservations(examples.get_path('crimes.shp'),
        ...                                        'crimes',
        ...                                         attribute=True)
       
        >>> npts = ntw.pointpatterns['crimes'].npoints
        >>> sim = ntw.simulate_observations(npts)
        >>> isinstance(sim, spgh.network.SimulatedPointPattern)
        True
        
        """
        simpts = SimulatedPointPattern()
        
        # Cumulative Network Length.
        edges = []
        lengths = np.zeros(len(self.edge_lengths))
        for i, key in enumerate(self.edge_lengths.keys()):
            edges.append(key)
            lengths[i] = self.edge_lengths[key]
        stops = np.cumsum(lengths)
        totallength = stops[-1]
        
        if distribution is 'uniform':
            nrandompts = np.random.uniform(0, totallength, size=(count,))
        elif distribution is 'poisson':
            mid_length = totallength / 2.
            nrandompts = np.random.poisson(mid_length, size=(count,))
            
        for i, r in enumerate(nrandompts):
            idx = np.where(r < stops)[0][0]
            assignment_edge = edges[idx]
            distance_from_start = stops[idx] - r
            # Populate the coordinates dict.
            x0, y0 = self._newpoint_coords(assignment_edge,
                                           distance_from_start)
            simpts.snapped_coordinates[i] = (x0, y0)
            simpts.obs_to_node[assignment_edge[0]].append(i)
            simpts.obs_to_node[assignment_edge[1]].append(i)
            
            # Populate the distance to node.
            distance_from_end = self.edge_lengths[edges[idx]]\
                                                        - distance_from_start
            simpts.dist_to_node[i] = {assignment_edge[0]: distance_from_start,
                                      assignment_edge[1]: distance_from_end}
            
            simpts.points = simpts.snapped_coordinates
            simpts.npoints = len(simpts.points)
            
        return simpts


    def enum_links_node(self, v0):
        """Returns the edges (links) around node.
        
        Parameters
        -----------
        
        v0 : int
            Node id
        
        Returns
        -------
        
        links : list
            List of tuple edges adjacent to the node.
        
        Examples
        --------
        
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.enum_links_node(24)
        [(24, 48), (24, 25), (24, 26)]
        
        """
        links = []
        neighbornodes = self.adjacencylist[v0]
        for n in neighbornodes:
            links.append(tuple(sorted([n, v0])))
        
        return links


    def node_distance_matrix(self, n_processes, gen_tree=False):
        """ Called from within allneighbordistances(),
        nearestneighbordistances(), and distancebandweights().
        
        Parameters
        -----------
        
        n_processes : int
            cpu cores for multiprocessing.
       
        gen_tree : bool
            rebuild shortest path {True}, or skip {False}
        
        """
        self.alldistances = {}
        nnodes = len(self.node_list)
        self.distancematrix = np.empty((nnodes, nnodes))
        # Single-core processing
        if not n_processes:
            for node in self.node_list:
                distance, pred = util.dijkstra(self, self.edge_lengths, node)
                pred = np.array(pred)
                if gen_tree:
                    tree = util.generatetree(pred)
                else:
                    tree = None
                self.alldistances[node] = (distance, tree)
                self.distancematrix[node] = distance
        # Multiprocessing
        if n_processes:
            import multiprocessing as mp
            from itertools import repeat
            if n_processes == "all":
                cores = mp.cpu_count()
            else:
                cores = n_processes
            p = mp.Pool(processes=cores)
            distance_pred = p.map(util.dijkstra_multi,
                                  zip(repeat(self),
                                      repeat(self.edge_lengths),
                                      self.node_list))
            iterations = range(len(distance_pred))
            distance = [distance_pred[itr][0] for itr in iterations]
            pred = np.array([distance_pred[itr][1] for itr in iterations])
            for node in self.node_list:
                if gen_tree:
                    tree = util.generatetree(pred[node])
                else:
                    tree = None
                self.alldistances[node] = (distance[node], tree)
                self.distancematrix[node] = distance[node]


    def allneighbordistances(self, sourcepattern, destpattern=None,
                             fill_diagonal=None, n_processes=None,
                             gen_tree=False, snap_dist=False):
        """ Compute either all distances between i and j in a single point
        pattern or all distances between each i from a source pattern and all
        j from a destination pattern.
        
        Parameters
        ----------
        
        sourcepattern : str or spaghetti.network.PointPattern
            The key of a point pattern snapped to the network OR
            the full spaghetti.network.PointPattern object.
        
        destpattern : str
            (Optional) The key of a point pattern snapped to the network OR
            the full spaghetti.network.PointPattern object.
        
        fill_diagonal : float, int
            (Optional) Fill the diagonal of the cost matrix. Default in None
            and will populate the diagonal with numpy.nan Do not declare a
            destpattern for a custom fill_diagonal.
        
        n_processes : int, str
            (Optional) Specify the number of cores to utilize. Default is 1
            core. Use (int) to specify an exact number or cores.  Use ("all")
            to request all available cores.
        
        gen_tree : bool
            rebuild shortest path {True}, or skip {False}
        
        snap_dist : bool
            include the distance from the original location to the snapped
            location along the network. Default is False.
        
        Returns
        -------
        
        nearest : numpy.ndarray
            An array of shape (n,n) storing distances between all points.
        
        tree_nearest : dict
            nearest network node to point pattern node shortest path lookup
        
        Examples
        --------
        
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.snapobservations(examples.get_path('crimes.shp'),
        ...                                        'crimes',
        ...                                         attribute=True)
        
        
        >>> s2s_dist = ntw.allneighbordistances('crimes')
        >>> s2s_dist[0,0], s2s_dist[1,0]
        (nan, 3105.189475447081)
        
        
        >>> ntw.snapobservations(examples.get_path('schools.shp'),
        ...                                        'schools',
        ...                                        attribute=False)
        
        
        >>> s2d_dist = ntw.allneighbordistances('crimes',
        ...                                     destpattern='schools')
        >>> s2d_dist[0,0], s2d_dist[1,0]
        (4520.72353741989, 6340.422971967316)
        
        
        >>> s2d_dist, tree = ntw.allneighbordistances('schools',
        ...                                           gen_tree=True)
        >>> tree[(6, 7)]
        (173, 64)
        """
        
        if not hasattr(self, 'alldistances'):
            self.node_distance_matrix(n_processes, gen_tree=gen_tree)
        
        if type(sourcepattern) is str:
            sourcepattern = self.pointpatterns[sourcepattern]
            if destpattern:
                destpattern = self.pointpatterns[destpattern]
        
        # Source setup
        src_indices = list(sourcepattern.points.keys())
        src_d2n = copy.deepcopy(sourcepattern.dist_to_node)
        nsource_pts = len(src_indices)
        src_nodes = {}
        for s in src_indices:
            e1, e2 = src_d2n[s].keys()
            src_nodes[s] = (e1, e2)
            
        # Destination setup
        symmetric = False
        if destpattern is None:
            symmetric = True
            destpattern = sourcepattern
        dest_indices = list(destpattern.points.keys())
        dst_d2n = copy.deepcopy(destpattern.dist_to_node)
        ndest_pts = len(dest_indices)
        dest_searchpts = copy.deepcopy(dest_indices)
        dest_nodes = {}
        
        # add snapping distance to each pointpattern
        if snap_dist:
            patterns = [sourcepattern, destpattern]
            dist_copies = [src_d2n, dst_d2n]
            for elm, pp in enumerate(patterns):
                for pidx, dists_dict in dist_copies[elm].items():
                    for nidx, ndist in dists_dict.items():
                        dists_dict[nidx] = ndist + pp.dist_snapped[pidx]
        
        for s in dest_indices:
            e1, e2 = dst_d2n[s].keys()
            dest_nodes[s] = (e1, e2)
            
        # Output setup
        nearest = np.empty((nsource_pts, ndest_pts))
        nearest[:] = np.inf
        tree_nearest = {}
        
        for p1 in src_indices:
            # Get the source nodes and dist to source nodes.
            source1, source2 = src_nodes[p1]
            set1 = set(src_nodes[p1])
            # Distance from node1 to p, distance from node2 to p.
            sdist1, sdist2 = src_d2n[p1].values()
            
            if symmetric:
                # Only compute the upper triangle if symmetric.
                dest_searchpts.remove(p1)
            for p2 in dest_searchpts:
                dest1, dest2 = dest_nodes[p2]
                set2 = set(dest_nodes[p2])
                if set1 == set2:  # same edge
                    x1, y1 = sourcepattern.snapped_coordinates[p1]
                    x2, y2 = destpattern.snapped_coordinates[p2]
                    computed_length = util.compute_length((x1, y1), (x2, y2))
                    nearest[p1, p2] = computed_length
                    
                else:
                    ddist1, ddist2 = dst_d2n[p2].values()
                    
                    d11 = self.distancematrix[source1][dest1]
                    d21 = self.distancematrix[source2][dest1]
                    d12 = self.distancematrix[source1][dest2]
                    d22 = self.distancematrix[source2][dest2]
                    
                    # Find the shortest distance from the path passing
                    # through each of the two origin nodes to the first
                    # destination node.
                    sd_1 = d11 + sdist1
                    sd_21 = d21 + sdist2
                    sp_combo1 = source1, dest1
                    if sd_1 > sd_21:
                        sd_1 = sd_21
                        sp_combo1 = source2, dest1
                    
                    # Now add the point to node one distance on
                    # the destination edge.
                    len_1 = sd_1 + ddist1
                    
                    # Repeat the prior but now for the paths entering
                    # at the second node of the second edge.
                    sd_2 = d12 + sdist1
                    sd_22 = d22 + sdist2
                    sp_combo2 = source1, dest2
                    if sd_2 > sd_22:
                        sd_2 = sd_22
                        sp_combo2 = source2, dest2
                    len_2 = sd_2 + ddist2
                    
                    # Now find the shortest distance path between point
                    # 1 on edge 1 and point 2 on edge 2, and assign.
                    sp_12 = len_1
                    s_node, d_node = sp_combo1
                    if len_1 > len_2:
                        sp_12 = len_2
                        s_node, d_node = sp_combo2
                    
                    nearest[p1, p2] = sp_12
                    tree_nearest[p1, p2] = (s_node, d_node)
                    
                if symmetric:
                    # Mirror the upper and lower triangle
                    # when symmetric.
                    nearest[p2, p1] = nearest[p1, p2]
                    
        # Populate the main diagonal when symmetric.
        if symmetric:
            if fill_diagonal is None:
                np.fill_diagonal(nearest, np.nan)
            else:
                np.fill_diagonal(nearest, fill_diagonal)
        
        if gen_tree:
            return nearest, tree_nearest
        else:
            return nearest


    def nearestneighbordistances(self, sourcepattern, destpattern=None,
                                 n_processes=None, gen_tree=False,
                                 all_dists=None, snap_dist=False,
                                 keep_zero_dist=True):
        """Compute the interpattern nearest neighbor distances or the
        intrapattern nearest neighbor distances between a source
        pattern and a destination pattern.
        
        Parameters
        ----------
        
        sourcepattern : str
            The key of a point pattern snapped to the network.
        
        destpattern : str
            (Optional) The key of a point pattern snapped to the network.
        
        n_processes : int, str
            (Optional) Specify the number of cores to utilize. Default is 1
            core. Use (int) to specify an exact number or cores. Use ("all")
            to request all available cores.
        
        gen_tree : bool
            rebuild shortest path {True}, or skip {False}
        
        all_dists : numpy.ndarray
            An array of shape (n,n) storing distances between all points.
        
        snap_dist : bool
            include the distance from the original location to the snapped
            location along the network. Default is False.
        
        keep_zero_dist : bool
            Include zero values in minimum distance (True) or exclude (False).
            Default is True. If the source pattern is the same as the
            destination pattern the diagonal is filled with nans
        
        Returns
        -------
        nearest : dict
            key is source point id, value is tuple of list containing 
            nearest destination point ids and distance.
        
        Examples
        --------
        
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.snapobservations(examples.get_path('crimes.shp'), 'crimes')
        >>> nn = ntw.nearestneighbordistances('crimes', keep_zero_dist=True)
        >>> nn[11], nn[18]
        (([18, 19], 165.33982412719126), ([19], 0.0))
        
        >>> nn = ntw.nearestneighbordistances('crimes', keep_zero_dist=False)
        >>> nn[11], nn[18]
        (([18, 19], 165.33982412719126), ([11], 165.33982412719126))
        
        """
        if sourcepattern not in self.pointpatterns.keys():
            err_msg = "Available point patterns are {}"
            raise KeyError(err_msg.format(self.pointpatterns.keys()))
            
        if not hasattr(self, 'alldistances'):
            self.node_distance_matrix(n_processes, gen_tree=gen_tree)
        
        symmetric = sourcepattern != destpattern
        
        # (for source-to-source patterns) if zero-distance neighbors are
        # desired, keep the diagonal as NaN and take the minimum distance
        # neighbor(s), which may include zero distance neighors.
        fill_diagonal = None
        if not keep_zero_dist and symmetric:
            # (for source-to-source patterns) if zero-distance neighbors should
            # be ignored, convert the diagonal to 0.0 and take the minimum
            # distance neighbor(s) that is/are not 0.0 distance.
            fill_diagonal = 0.
        
        sourcepattern = self.pointpatterns[sourcepattern]
        if destpattern:
            destpattern = self.pointpatterns[destpattern]
        
        if all_dists is None:
            all_dists = self.allneighbordistances(sourcepattern,
                                                  destpattern=destpattern,
                                                  fill_diagonal=fill_diagonal,
                                                  n_processes=n_processes,
                                                  gen_tree=gen_tree,
                                                  snap_dist=snap_dist)
        nearest = {}
        
        for source_index in sourcepattern.points.keys():
            if keep_zero_dist and symmetric:
                val = np.nanmin(all_dists[source_index,:])
            else:
                val = np.min(all_dists[source_index,:]\
                                      [np.nonzero(all_dists[source_index,:])])
            # nearest destination (may be more than one if equal distance)
            dest_idxs = np.where(all_dists[source_index,:] == val)[0].tolist()
            nearest[source_index] = (dest_idxs, val)
            
        return nearest


    def NetworkF(self, pointpattern, nsteps=10, permutations=99, threshold=0.2,
                 distribution='uniform',  lowerbound=None, upperbound=None):
        """Computes a network constrained F-Function
        
        Parameters
        ----------
        
        pointpattern : spaghetti.network.PointPattern
            A spaghetti point pattern object.
        
        nsteps : int
            The number of steps at which the count of the nearest neighbors
            is computed.
        
        permutations : int
            The number of permutations to perform (default 99).
        
        threshold : float
            The level at which significance is computed.
            -- 0.5 would be 97.5% and 2.5%
        
        distribution : str
            The distribution from which random points are sampled
            -- uniform or poisson
        
        lowerbound : float
            The lower bound at which the F-function is computed. (Default 0).
        
        upperbound : float
            The upper bound at which the F-function is computed. Defaults to
            the maximum observed nearest neighbor distance.
        
        Returns
        -------
        
        NetworkF : spaghetti.analysis.NetworkF
            A network F class instance.
        
        Examples
        --------
        
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(in_data=examples.get_path('streets.shp'))
        >>> pt_str = 'crimes'
        >>> in_data = examples.get_path('{}.shp'.format(pt_str))
        >>> ntw.snapobservations(in_data, pt_str, attribute=True)
        >>> crimes = ntw.pointpatterns['crimes']
        >>> sim = ntw.simulate_observations(crimes.npoints)
        >>> fres = ntw.NetworkF(crimes, permutations=5, nsteps=10)
        >>> fres.lowerenvelope.shape[0]
        10
        """
        return NetworkF(self, pointpattern, nsteps=nsteps,
                        permutations=permutations, threshold=threshold,
                        distribution=distribution, lowerbound=lowerbound,
                        upperbound=upperbound)


    def NetworkG(self, pointpattern, nsteps=10, permutations=99,
                 threshold=0.5, distribution='uniform',
                 lowerbound=None, upperbound=None):
        """Computes a network constrained G-Function
        
        Parameters
        ----------
        
        pointpattern : spaghetti.network.PointPattern
            A spaghetti point pattern object.
        
        nsteps : int
            The number of steps at which the count of the nearest neighbors
            is computed.
        
        permutations : int
            The number of permutations to perform (default 99).
        
        threshold : float
            The level at which significance is computed.
            -- 0.5 would be 97.5% and 2.5%
        
        distribution : str
            The distribution from which random points are sampled
            -- uniform or poisson
        
        lowerbound : float
            The lower bound at which the G-function is computed. (Default 0).
        
        upperbound : float
            The upper bound at which the G-function is computed. Defaults to
            the maximum observed nearest neighbor distance.
        
        Returns
        -------
        
        NetworkG : spaghetti.analysis.NetworkG
            A network G class instance.
        
        Examples
        --------
        
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(in_data=examples.get_path('streets.shp'))
        >>> pt_str = 'crimes'
        >>> in_data = examples.get_path('{}.shp'.format(pt_str))
        >>> ntw.snapobservations(in_data, pt_str, attribute=True)
        >>> crimes = ntw.pointpatterns['crimes']
        >>> sim = ntw.simulate_observations(crimes.npoints)
        >>> gres = ntw.NetworkG(crimes, permutations=5, nsteps=10)
        >>> gres.lowerenvelope.shape[0]
        10
        """
        
        return NetworkG(self, pointpattern, nsteps=nsteps,
                        permutations=permutations, threshold=threshold,
                        distribution=distribution, lowerbound=lowerbound,
                        upperbound=upperbound)


    def NetworkK(self, pointpattern, nsteps=10, permutations=99,
                 threshold=0.5, distribution='uniform',
                 lowerbound=None, upperbound=None):
        """
        Computes a network constrained K-Function
        
        Parameters
        ----------
        
        pointpattern : spaghetti.network.PointPattern
            A spaghetti point pattern object.
        
        nsteps : int
            The number of steps at which the count of the nearest neighbors
            is computed.
        
        permutations : int
            The number of permutations to perform (default 99).
        
        threshold : float
            The level at which significance is computed.
            -- 0.5 would be 97.5% and 2.5%
        
        distribution : str
            The distribution from which random points are sampled
            -- uniform or poisson
        
        lowerbound : float
            The lower bound at which the K-function is computed. (Default 0).
        
        upperbound : float
            The upper bound at which the K-function is computed. Defaults to
            the maximum observed nearest neighbor distance.
        
        Returns
        -------
        
        NetworkK : spaghetti.analysis.NetworkK
            A network K class instance.
        
        Examples
        --------
        
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(in_data=examples.get_path('streets.shp'))
        >>> pt_str = 'crimes'
        >>> in_data = examples.get_path('{}.shp'.format(pt_str))
        >>> ntw.snapobservations(in_data, pt_str, attribute=True)
        >>> crimes = ntw.pointpatterns['crimes']
        >>> sim = ntw.simulate_observations(crimes.npoints)
        >>> kres = ntw.NetworkK(crimes, permutations=5, nsteps=10)
        >>> kres.lowerenvelope.shape[0]
        10
        """
        return NetworkK(self, pointpattern, nsteps=nsteps,
                        permutations=permutations, threshold=threshold,
                        distribution=distribution, lowerbound=lowerbound,
                        upperbound=upperbound)


    def segment_edges(self, distance):
        """Segment all of the edges in the network at either a
        fixed distance or a fixed number of segments.
        
        Parameters
        -----------
        
        distance : float
            The distance at which edges are split.
        
        Returns
        -------
        
        sn : spaghetti.Network
            spaghetti Network object.
        
       Examples
        --------
       
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
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
        sn.in_data = self.in_data
        
        current_node_id = max(self.nodes.values())
        
        newedges = set()
        removeedges = set()
        for e in sn.edges:
            length = sn.edge_lengths[e]
            interval = distance
            
            totallength = 0
            currentstart = startnode = e[0]
            endnode = e[1]
            
            # If the edge will be segmented remove the current
            # edge from the adjacency list.
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
                    
                    # Compute the new node coordinate.
                    newx, newy = self._newpoint_coords(e, totallength)
                    
                    # Update node_list.
                    if currentstop not in sn.node_list:
                        sn.node_list.append(currentstop)
                    
                    # Update nodes and node_coords.
                    sn.node_coords[currentstop] = newx, newy
                    sn.nodes[(newx, newy)] = currentstop
                
                # Update the adjacency list.
                sn.adjacencylist[currentstart].append(currentstop)
                sn.adjacencylist[currentstop].append(currentstart)
                
                # Add the new edge to the edge dict.
                # Iterating over this so we need to add after iterating.
                newedges.add(tuple(sorted([currentstart, currentstop])))
                
                # Modify edge_lengths.
                current_start_stop = tuple(sorted([currentstart, currentstop]))
                sn.edge_lengths[current_start_stop] = interval
                
                # Increment the start to the stop.
                currentstart = currentstop
        
        sn.edges.update(newedges)
        sn.edges.difference_update(removeedges)
        sn.edges = list(sn.edges)
        # Update the point pattern snapping.
        for instance in sn.pointpatterns.values():
            sn._snap_to_edge(instance)
            
        return sn


    def savenetwork(self, filename):
        """Save a network to disk as a binary file.
        
        Parameters
        ----------
        
        filename : str
            The filename where the network should be saved. This should be a
            full path or it will be save in the current directory.
        
        Examples
        --------
        
        >>> import spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.savenetwork('mynetwork.pkl')
        """
        with open(filename, 'wb') as networkout:
            pickle.dump(self, networkout, protocol=2)

    @staticmethod
    def loadnetwork(filename):
        """Load a network from a binary file saved on disk.
        
        Parameters
        ----------
        
        filename : str
            The filename where the network should be saved.
        
        Returns
        -------
        
        self : spaghetti.Network
            spaghetti Network object
            
        """
        with open(filename, 'rb') as networkin:
            self = pickle.load(networkin)
            
        return self


class PointPattern():
    """A stub point pattern class used to store a point pattern. This class is
    monkey patched with network specific attributes when the points are snapped
    to a network. In the future this class may be replaced with a generic point
    pattern class.
    
    Parameters
    ----------
    
    in_data : geopandas.GeoDataFrame or str
        The input geographic data. Either (1) a path to a shapefile (str);
        or (2) a geopandas.GeoDataFrame.
    
    idvariable : str
        Field in the shapefile to use as an id variable.
    
    attribute :  bool
        {False, True} A flag to indicate whether all attributes are tagged
        to this class.
    
    Attributes
    ----------
    
    points : dict
        Keys are the point ids (int). Values are the x,y coordinates (tuple).
    
    npoints : int
        The number of points.
    
    obs_to_edge : dict
        Keys are edge ids (tuple). Values are snapped point information (dict).
        Withing the snapped point information (dict) keys are observation id
        (int), and values are snapped coordinates.
    
    obs_to_node : list
       list of incident network nodes to snapped observation points converted
       from default_dict. Originally in the form of paired left/right nearest
       network nodes - {netnode1: obs_id1, netnode2: obs_id1,
       netnode1: obs_id2... netnode1: obs_idn}, then simplified to a list in
       the form - [netnode1, netnode2, netnode1, netnode2, ...].
       
    dist_to_node : dict
        Keys are observations ids (int). Values are distance lookup (dict).
        Within distance lookup (dict) keys are the two incident nodes of the
        edge and values are distance to each of those edges.
    
    snapped_coordinates : dict
        Keys are the point ids (int). Values are the snapped x,y
        coordinates (tuple).
    
    snap_dist : bool
        include the distance from the original location to the snapped
        location along the network. Default is False.
    
    """
    def __init__(self, in_data=None, idvariable=None, attribute=False):
        self.points = {}
        self.npoints = 0
        
        if isinstance(in_data, str):
            from_shp = True
        else:
            from_shp = False
            
        if idvariable:
            ids = weights.util.get_ids(in_data, idvariable)
        else:
            ids = None
            
        if from_shp:
            pts = open(in_data)
        else:
            pts_objs = list(in_data.geometry)
            pts = [cg.shapes.Point((p.x, p.y)) for p in pts_objs]
            
        # Get attributes if requested
        if attribute:
            if from_shp:
                dbname = os.path.splitext(in_data)[0] + '.dbf'
                db = open(dbname)
            else:
                db = in_data.drop(in_data.geometry.name,
                                   axis=1).values.tolist()
                db = [[d] for d in db]
        else:
            db = None
            
        for i, pt in enumerate(pts):
            # ids, attributes
            if ids and db is not None:
                self.points[ids[i]] = {'coordinates': pt, 'properties': db[i]}
            # ids, no attributes
            elif ids and db is None:
                self.points[ids[i]] = {'coordinates': pt, 'properties': None}
            # no ids, attributes
            elif not ids and db is not None:
                self.points[i] = {'coordinates': pt, 'properties': db[i]}
            # no ids, no attributes
            else:
                self.points[i] = {'coordinates': pt, 'properties': None}
        if from_shp:
            pts.close()
            if db:
                db.close()
                
        self.npoints = len(self.points.keys())


class SimulatedPointPattern():
    """Struct style class to mirror the Point Pattern Class. If the
    PointPattern class has methods, it might make sense to make this a child of
    that class. This class is not intended to be used by the external user.
    
    Attributes
    ----------
    
    npoints : int
        The number of points.
    
    obs_to_edge : dict
        Keys are edge ids (tuple). Values are snapped point information (dict).
        Withing the snapped point information (dict) keys are observation id
        (int), and values are snapped coordinates.
    
    obs_to_node : list
       list of incident network nodes to snapped observation points converted
       from default_dict. Originally in the form of paired left/right nearest
       network nodes - {netnode1: obs_id1, netnode2: obs_id1,
       netnode1: obs_id2... netnode1: obs_idn}, then simplified to a list in
       the form - [netnode1, netnode2, netnode1, netnode2, ...].
       
    dist_to_node : dict
        Keys are observations ids (int). Values are distance lookup (dict).
        Within distance lookup (dict) keys are the two incident nodes of the
        edge and values are distance to each of those edges.
    
    snapped_coordinates : dict
        Keys are the point ids (int). Values are the snapped x,y
        coordinates (tuple).
    """
    def __init__(self):
        self.npoints = 0
        self.obs_to_edge = {}
        self.obs_to_node = defaultdict(list)
        self.dist_to_node = {}
        self.snapped_coordinates = {}


class SortedEdges(OrderedDict):
    """
    Parameters
    ----------
    OrderedDict : collections.OrderedDict
    
    """
    def next_key(self, key):
        """
        Parameters
        ----------
        
        key : 
            
        Returns
        -------
        
        n : 
            
        """
        next = self._OrderedDict__map[key][1]
        if next is self._OrderedDict__root:
            raise ValueError("{!r} is the last key.".format(key))
        n = next[2]
        return n

    def first_key(self):
        """
        
        Returns
        -------
        
        key : 
            
        """
        for key in self:
            return key
        raise ValueError("No sorted edges remain.")
