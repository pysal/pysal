from collections import defaultdict, OrderedDict
import copy, os, pickle

import numpy as np

from .analysis import NetworkG, NetworkK, NetworkF
from . import util
from pysal.lib import cg, examples, weights
try:
    from pysal.lib import open
except ImportError:
    import pysal.lib
    open = pysal.lib.io.open


__all__ = ["Network", "PointPattern", "NetworkG", "NetworkK", "NetworkF"]


class Network:
    """Spatially-constrained network representation and analytical
    functionality. Naming conventions are as follows, (1) arcs and
    vertices for the full network object, and (2) edges and nodes for
    the simplified graph-theoretic object. The term 'link' is used to
    refer to a network arc or a graph edge.
    
    Parameters
    ----------
    
    in_data : {geopandas.GeoDataFrame, str}
        The input geographic data. Either (1) a path to a shapefile
        (str); or (2) a `geopandas.GeoDataFrame 
        <http://geopandas.org/data_structures.html#geodataframe>`_.
    
    vertex_sig : int
        Round the x and y coordinates of all vertices to ``vertex_sig``
        significant digits (combined significant digits on the left and
        right of the decimal place). Default is 11. Set to ``None`` for
        no rounding.
    
    unique_arcs : bool
        If ``True`` (default), keep only unique arcs (i.e., prune
        out any duplicated arcs). If ``False`` keep all segments.
    
    extractgraph : bool
        If ``True``, extract a graph-theoretic object with no degree 2
        nodes. Default is ``True``.
    
    w_components : bool
        Set to ``False`` to not record connected components from a
        `pysal.lib.weights.weights.W 
        <https://pysal.lib.readthedocs.io/en/latest/generated/
        pysal.lib.weights.W.html#pysal.lib.weights.W>`_
        object. Default is True.
        
        
    weightings : {dict, bool}
        If ``dict``, lists of weightings for each arc. If ``bool``,
        ``True`` flags ``self.arc_lengths`` as the weightings,
        ``False`` sets to no weightings. Default is ``False``.
    
    Attributes
    ----------
    
    adjacencylist : list
        List of lists storing vertex adjacency.
    
    vertex_coords : dict
        Keys are the vertex ID and values are the (x,y) coordinates
        inverse to vertices.
    
    vertex_list : list
        List of vertex IDs.
    
    vertices : dict
        Keys are tuples of vertex coords and values are the vertex ID.
    
    arcs : list
        List of arcs, where each arc is a sorted tuple
        of vertex IDs.
    
    arc_lengths : dict
        Keys are tuples of sorted vertex IDs representing an arc and
        values are the length.
    
    pointpatterns : dict
        Keys are a string name of the pattern and values are
        ``PointPattern`` class instances.
    
    alldistances : dict
        Keys are the vertex IDs (``int``). Values are tuples with two
        elements as follows (1) a list of the shortest path distances;
        (2) a dict with the key being the id of the destination vertex
        and the value being a list of the shortest path.
    
    distancematrix : `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
        all network vertices (non-observations) distance matrix.
    
    edges : list
        tuples of graph edge ids.
    
    edge_lengths : dict
        Keys are the graph edge ids (tuple). Values are the graph edge
        length (``float``).
    
    non_articulation_points : list
        All vertices with degree 2 that are not in an isolated
        island ring (loop) component
    
    w_network : `pysal.lib.weights.weights.W <https://pysal.lib.readthedocs.io/en/latest/generated/pysal.lib.weights.W.html#pysal.lib.weights.W>`_
        Weights object created from the network arcs
    
    network_n_components : int
        Count of connected components in the network.
    
    network_component_labels : `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
        Component labels for networks arc
    
    network_component2arc : dict
        Lookup ``{int: list}`` for arcs comprising network
        connected components keyed by component labels with arcs in
        a ``list`` as values.
    
    network_component_is_ring : dict
        Lookup ``{int: bool}`` keyed by component labels with values
        as ``True`` if the component is a closed ring, otherwise
        ``False``.
    
    w_graph : `pysal.lib.weights.weights.W <https://pysal.lib.readthedocs.io/en/latest/generated/pysal.lib.weights.W.html#pysal.lib.weights.W>`_
        Weights object created from the graph edges
    
    graph_n_components : int
        Count of connected components in the network.
    
    graph_component_labels : `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
        Component labels for graph edges
    
    graph_component2edge : dict
        Lookup ``{int: list}`` for edges comprising graph connected
        components keyed by component labels with edges in a list
        as values.
    
    graph_component_is_ring : dict
        Lookup ``{int: bool}`` keyed by component labels with values as
        ``True`` if the component is a closed ring, otherwise ``False``.
    
    Examples
    --------
    
    Instantiate an instance of a network.
    
    >>> import pysal.explore.spaghetti as spgh
    >>> streets_file = examples.get_path('streets.shp')
    >>> ntw = spgh.Network(in_data=streets_file)
    
    Snap point observations to the network with attribute information.
    
    >>> crimes_file = examples.get_path('crimes.shp')
    >>> ntw.snapobservations(crimes_file, 'crimes', attribute=True)
   
    And without attribute information.
   
    >>> schools_file = examples.get_path('schools.shp')
    >>> ntw.snapobservations(schools_file, 'schools', attribute=False)
    
    """
    
    def __init__(self, in_data=None, vertex_sig=11, unique_arcs=True,
                 extractgraph=True, w_components=True, weightings=False):
        
        # do this when creating a clean network instance from a
        # shapefile or a geopandas.GeoDataFrame, otherwise a shell
        # network instance is created (see `split_arcs()` method)
        if in_data is not None:
            
            # set parameters as attributes
            self.in_data = in_data
            self.vertex_sig = vertex_sig
            self.unique_arcs = unique_arcs
            
            self.adjacencylist = defaultdict(list)
            self.vertices = {}
            
            # initialize network arcs and arc_lengths
            self.arcs = []
            self.arc_lengths = {}
            
            # initialize pointpatterns
            self.pointpatterns = {}
            
            # spatial representation of the network
            self._extractnetwork()
            self.arcs = sorted(self.arcs)
            self.vertex_coords = dict((v, k) for k, v in self.vertices.items())
            
            # extract connected components
            if w_components:
                as_graph = False
                network_weightings = False
                
                if weightings is True:
                    # set network arc weights to length if weights are
                    # desired, but no other input in given
                    weightings = self.arc_lengths
                    network_weightings = True
                
                # extract contiguity weights from pysal.lib
                self.w_network = self.contiguityweights(graph=as_graph,
                                                        weightings=weightings)
                # extract connected components from the `w_network`
                self.extract_components(self.w_network, graph=as_graph)
            
            # extract the graph -- repeat similar as above
            # for extracting the network
            if extractgraph:
                self.extractgraph()
                
                if w_components:
                    as_graph = True
                    
                    if network_weightings:
                        weightings = self.edge_lengths
                    
                    self.w_graph = self.contiguityweights(\
                                                        graph=as_graph,
                                                        weightings=weightings)
                    self.extract_components(self.w_graph, graph=as_graph)
            
            # sorted list of vertex ids
            self.vertex_list = sorted(self.vertices.values())
    
    
    def _round_sig(self, v):
        """Used internally to round the vertex to a set number of
        significant digits. If ``sig`` is set to 4, then the following
        are some possible results for a coordinate are as follows.
        (1) 0.0xxxx, (2) 0.xxxx, (3) x.xxx, (4) xx.xx,
        (5) xxx.x, (6) xxxx.0, (7) xxxx0.0
        
        Parameters
        ----------
        
        v : tuple
            X,Y coordinate of the vertex
        
        """
        
        # set the number of significant digits
        sig = self.vertex_sig
        
        # simply return vertex (x,y) coordinates
        if sig is None:
            return v
        
        # for each coordinate in a coordinate pair
        # if the coordinate location is (0.0) simply return zero
        # else -- (1) take the absolute value of `val`; (2) take the
        # base 10 log for [1]; (3) take the floor of [2]; (4) convert
        # [3] into a negative integer; (5) add `sig - 1` to [4];
        # (6) round `val` by [5]
        out_v = [val if val == 0 \
                 else round(val, -int(np.floor(np.log10(np.fabs(val))))\
                            + (sig - 1))\
                 for val in v]
        
        return tuple(out_v)
    
    
    def extract_components(self, w, graph=False):
        """Extract connected component information from a
        ``pysal.lib.weights.weights.W`` object
        
        Parameters
        ----------
        
        w : `pysal.lib.weights.weights.W <https://pysal.lib.readthedocs.io/en/latest/generated/pysal.lib.weights.W.html#pysal.lib.weights.W>`_
            Weights object created from the network segments (either
            raw or graph-theoretic)
        
        graph : bool
            Flag for raw network [False] or graph-theoretic network
            ``True``. Default is ``False``.
        
        """
        
        # flag network (arcs) or graph (edges)
        if graph:
            links = self.edges
            obj_type = 'graph_'
        else:
            links = self.arcs
            obj_type = 'network_'
        
        # connected component count and labels
        n_components = w.n_components
        component_labels = w.component_labels
        
        # link to component lookup
        link2component = dict(zip(links, component_labels))
        
        # component ID to links lookup
        component2link = {}
        cp_labs = set(w.component_labels)
        for cpl in cp_labs:
            component2link[cpl] = sorted([k for k,v\
                                          in link2component.items()\
                                          if v == cpl])
        
        # component to ring lookup
        component_is_ring = {}
        for k,vs in component2link.items():
            component_is_ring[k] = True
            for v in vs:
                if len(w.neighbors[v]) != 2:
                    component_is_ring[k] = False
        
        # attribute label name depends on object type
        if graph:
            c2l_attr_name = 'component2edge'
        else:
            c2l_attr_name = 'component2arc'
        
        # set all new variables into list
        extracted_attrs = [['n_components', n_components],
                           ['component_labels', component_labels],
                           [c2l_attr_name, component2link],
                           ['component_is_ring', component_is_ring]]
        
        # iterate over list and set attribute with
        # either "network" or "graph" extension
        for (attr_str, attr) in extracted_attrs:
            setattr(self, obj_type+attr_str, attr)
    
    
    def _extractnetwork(self):
        """Used internally to extract a network from a polyline
        shapefile of a ``geopandas.GeoDataFrame``.
        """
        
        # initialize vertex count
        vertex_count = 0
        
        # determine if input network data is coming from
        # shapefile or a geopandas.GeoDataFrame
        if isinstance(self.in_data, str):
            shps = open(self.in_data)
        else:
            shps = self.in_data.geometry
        
        # iterate over each record of the network lines
        for shp in shps:
            
            # fetch all vertices between euclidean segments
            # in the line record -- these vertices are
            # coordinates in an (x, y) tuple.
            vertices = weights._contW_lists._get_verts(shp)
            
            # iterate over each vertex (v)
            for i, v in enumerate(vertices[:-1]):
                
                # -- For vertex 1
                # adjust precision -- this was originally
                # implemented to handle high-precision
                # network network vertices
                v = self._round_sig(v)
                
                # when the vertex already exists in lookup
                # set it as the current `vid`
                try:
                    vid = self.vertices[v]
                # when the vertex is not present in the lookup
                # add it and adjust vertex count
                except KeyError:
                    self.vertices[v] = vid = vertex_count
                    vertex_count += 1
                
                # -- For vertex 2
                # repeat the steps above for vertex 1
                v2 = self._round_sig(vertices[i + 1])
                try:
                    nvid = self.vertices[v2]
                except KeyError:
                    self.vertices[v2] = nvid = vertex_count
                    vertex_count += 1
                
                # records vertex 1 and vertex 2 adjacency
                self.adjacencylist[vid].append(nvid)
                self.adjacencylist[nvid].append(vid)
                
                # Sort the edges so that mono-directional
                # keys can be stored.
                arc_vertices = sorted([vid, nvid])
                arc = tuple(arc_vertices)
                
                # record the euclidean arc within the network
                self.arcs.append(arc)
                
                # record length
                length = util.compute_length(v,
                                             vertices[i + 1])
                self.arc_lengths[arc] = length
        
        if self.unique_arcs:
            # Remove duplicate edges and duplicate adjacent nodes.
            self.arcs = list(set(self.arcs))
            for k, v in self.adjacencylist.items():
                self.adjacencylist[k] = list(set(v))
    
    
    def extractgraph(self):
        """Using the existing network representation, create a
        graph-theoretic representation by removing all vertices with a
        neighbor incidence of two (non-articulation points). That is, we
        assume these vertices are bridges between vertices with higher
        or lower incidence.
        """
        
        # initialize edges and edge_lengths
        self.edges = []
        self.edge_lengths = {}
        
        # find all vertices with degree 2 that are not in an isolated
        # island ring (loop) component. These are non-articulation
        # points on the graph representation
        non_articulation_points = self._yield_napts()
        # retain non_articulation_points as an attribute
        self.non_articulation_points = list(non_articulation_points)
        
        # start with a copy of the spatial representation and
        # iteratively remove edges deemed to be segments
        self.edges = copy.deepcopy(self.arcs)
        self.edge_lengths = copy.deepcopy(self.arc_lengths)
        
        # mapping all the 'network arcs' contained within a single
        # 'graph represented' edge
        self.arcs_to_edges = {}
        
        # build up bridges "rooted" on the initial
        # non-articulation points
        bridge_roots = []
        
        # iterate over all vertices that are not contained within
        # isolated loops that have a degree of 2
        for s in non_articulation_points:
        
            # initialize bridge with an articulation point
            bridge = [s]
            
            # fetch all vertices adjacent to point `s`
            # that are also degree 2
            neighbors = self._yieldneighbor(s,
                                            non_articulation_points,
                                            bridge)
            while neighbors:
                
                # extract the current node in `neighbors`
                cnode = neighbors.pop()
                # remove it from `non_articulation_points`
                non_articulation_points.remove(cnode)
                # add it to bridge
                bridge.append(cnode)
                # fetch neighbors for the current node
                newneighbors = self._yieldneighbor(cnode,
                                                   non_articulation_points,
                                                   bridge)
                # add the new neighbors back into `neighbors`
                neighbors += newneighbors
            
            # once all potential neighbors are exhausted add the
            # current bridge of non-articulation points to the
            # list of rooted bridges
            bridge_roots.append(bridge)
        
        # iterate over the list of newly created rooted bridges
        for bridge in bridge_roots:
            
            # if the vertex is only one non-articulation
            # point in the bridge
            if len(bridge) == 1:
                
                # that the singular element of the bridge
                n = self.adjacencylist[bridge[0]]
                # and create a new graph edge from it
                new_edge = tuple(sorted([n[0], n[1]]))
                
                # identify the arcs to be removed
                e1 = tuple(sorted([bridge[0], n[0]]))
                e2 = tuple(sorted([bridge[0], n[1]]))
                
                # remove the network arcs (spatial) from the
                # graph-theoretic representation
                self.edges.remove(e1)
                self.edges.remove(e2)
                
                # remove the former network arc lengths from the
                # graph edge lengths lookup
                length_e1 = self.edge_lengths[e1]
                length_e2 = self.edge_lengths[e2]
                self.edge_lengths.pop(e1, None)
                self.edge_lengths.pop(e2, None)
                
                # and add the new edge length in their place
                self.edge_lengths[new_edge] = length_e1 + length_e2
                
                # update the pointers
                self.arcs_to_edges[e1] = new_edge
                self.arcs_to_edges[e2] = new_edge
            
            # if there are more than one vertices in the bridge
            else:
                cumulative_length = 0
                start_end = {}
                
                # initialize a redundant set of bridge edges
                redundant = set([])
                
                # iterate over the current bridge
                for b in bridge:
                    # iterate over each node in the bridge
                    for n in self.adjacencylist[b]:
                        # start the bridge with this node
                        if n not in bridge:
                            start_end[b] = n
                        # or create a redundant edge with the current
                        # node and `b`
                        else:
                            redundant.add(tuple(sorted([b, n])))
                
                # initialize a new graph edge
                new_edge = tuple(sorted(start_end.values()))
                
                # add start_end redundant edge
                for k, v in start_end.items():
                    redundant.add(tuple(sorted([k, v])))
                
                # remove all redundant network arcs while
                # adjusting the graph edge lengths lookup
                # and the edges_to_arcs lookup
                for r in redundant:
                    self.edges.remove(r)
                    cumulative_length += self.edge_lengths[r]
                    self.edge_lengths.pop(r, None)
                    self.arcs_to_edges[r] = new_edge
                
                # finally, add the new cumulative edge length
                self.edge_lengths[new_edge] = cumulative_length
            
            # add the updated graph edge
            self.edges.append(new_edge)
        
        # converted the graph edges into a sorted set to prune out
        # duplicate graph edges created during simplification
        self.edges = sorted(set(self.edges))
    
    
    def _yield_napts(self):
        """Find all nodes with degree 2 that are not in an isolated
        island ring (loop) component. These are non-articulation
        points on the graph representation.
        
        Returns
        -------
        
        napts : list
            non-articulation points on a graph representation
        
        """
        
        # non-articulation points
        napts = set()
        
        # network vertices remaining to evaluate
        unvisted = set(self.vertices.values())
        
        while unvisted:
            
            # iterate over each component
            for component_id, ring in self.network_component_is_ring.items():
                
                # evaluate for non-articulation points
                napts, unvisted = self._evaluate_napts(napts,
                                                       unvisted,
                                                       component_id,
                                                       ring)
        
        # convert set of non-articulation points into list
        napts = list(napts)
        
        return napts
    
    
    def _evaluate_napts(self, napts, unvisited, component_id, ring):
        """Evaluate one connected component in a network for
        non-articulation points (napts) and return an updated set of
        napts and unvisted vertices.
        
        Parameters
        ----------
        
        napts : set
            Non-articulation points (napts) in the network. The
            'napts' here do not include those within an isolated
            loop island.
        
        unvisited : set
            Vertices left to evaluate in the network.
        
        component_id : int
            ID for the network connected component for the
            current iteration of the algorithm.
        
        ring : bool
            Network component is isolated island loop ``True`` or
            not ``False``.
        
        Returns
        -------
        
        napts : set
            Updated 'napts' object.
        
        unvisited : set
            Updated 'napts' object.
        
        """
        
        # iterate over each `edge` of the `component`
        for component in self.network_component2arc[component_id]:
            
            # each `component` has two vertices
            for vertex in component:
                
                # if `component` is not an isolated island
                # and `vertex` has exactly 2 neighbors,
                # add `vertex` to `napts`
                if not ring:
                    if len(self.adjacencylist[vertex]) == 2:
                        napts.add(vertex)
                
                # remove `vertex` from `unvisited` if
                # it is still in the set else move along to
                # the next iteration
                try:
                    unvisited.remove(vertex)
                except KeyError:
                    pass
        
        return napts, unvisited
    
    
    def _yieldneighbor(self, vtx, arc_vertices, bridge):
        """Used internally, this method traverses a bridge arc
        to find the source and destination nodes.
        
        Parameters
        ----------
        
        vtx : int
            vertex id
        
        arc_vertices : list
            All non-articulation points in the network. These are
            referred to as degree-2 vertices.
        
        bridge : list
            Inital bridge list containing only ``vtx``.
        
        Returns
        -------
        
        nodes : list
            Vertices to keep (articulation points). These elements are 
            referred to as nodes.
        """
        
        # instantiate empty lis to fill with network articulation
        # points (nodes with a degree of 1 [endpoints] or greater
        # than 2 [intersections])
        nodes = []
        
        # get all nodes adjacent to `vtx` that are not in the
        # set of 'bridge' vertices
        for i in self.adjacencylist[vtx]:
            
            if i in arc_vertices and i not in bridge:
                nodes.append(i)
        
        return nodes
    
    
    def contiguityweights(self, graph=True, weightings=None):
        """Create a contiguity-based pysal.lib W object.
        
        Parameters
        ----------
        
        graph : bool
            ``{True, False}`` controls whether the W is generated using
            the spatial representation or the graph representation.
            Default is ``True``.
        
        weightings : dict
            dictionary of lists of weightings for each arc/edge.
        
        Returns
        -------
        
         W : `pysal.lib.weights.weights.W <https://pysal.lib.readthedocs.io/en/latest/generated/pysal.lib.weights.W.html#pysal.lib.weights.W>`_
            A ``pysal`` W Object representing the binary adjacency of
            the network.
        
        Examples
        --------
        
        Instantiate an instance of a network.
        
        >>> import pysal.explore.spaghetti as spgh
        >>> from pysal.lib import examples
        >>> import pysal.explore.esda
        >>> import numpy as np
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        
        Snap point observations to the network with
        attribute information.
        
        >>> ntw.snapobservations(examples.get_path('crimes.shp'),
        ...                      'crimes', attribute=True)
        
        Find counts per network arc.
        
        >>> counts = ntw.count_per_link(ntw.pointpatterns['crimes']
        ...                             .obs_to_arc, graph=False)
        >>> counts[(50, 165)]
        4
        
        Create a contiguity based W object.
        
        >>> w = ntw.contiguityweights(graph=False)
        
        Using the W object, access to ESDA functionality is provided.
        First, a vector of attributes is created for all edges
        with observations.
        
        >>> w = ntw.contiguityweights(graph=False)
        >>> arcs = w.neighbors.keys()
        >>> y = np.zeros(len(arcs))
        >>> for i, e in enumerate(arcs):
        ...     if e in counts.keys():
        ...         y[i] = counts[e]
        >>> y[3]
        3.0
        
        Next, a standard call ot Moran is made and the
        result placed into ``res``.
        
        >>> res = pysal.explore.esda.moran.Moran(y, w, permutations=99)
        >>> type(res)
        <class 'pysal.explore.esda.moran.Moran'>
        
        """
        
        # instantiate OrderedDict to record network link
        # adjacency which will be keyed by the link id (a tuple)
        # with values being lists of tuples (contiguous links)
        neighbors = OrderedDict()
        
        # flag network (arcs) or graph (edges)
        if graph:
            links = self.edges
        else:
            links = self.arcs
        
        # if weightings are desired instantiate a dictionary
        # other ignore weightings
        if weightings:
            _weights = {}
        else:
            _weights = None
        
        # iterate over all links until all possibilities
        # for network link adjacency are exhausted
        working = True
        while working:
            
            # for each network link (1)
            for key in links:
                
                # instantiate a slot in the OrderedDict
                neighbors[key] = []
                
                if weightings:
                    _weights[key] = []
                
                # for each network link (2)
                for neigh in links:
                    
                    # skip if comparing link to itself
                    if key == neigh:
                        continue
                    
                    # if link(1) and link(2) share any vertex
                    # update neighbors adjacency
                    if key[0] == neigh[0] or key[0] == neigh[1]\
                    or key[1] == neigh[0] or key[1] == neigh[1]:
                        neighbors[key].append(neigh)
                        
                        # and add weights if desired
                        if weightings:
                            _weights[key].append(weightings[neigh])
                    
                    # break condition
                    # -- everything is sorted, so we know when we have
                    # stepped beyond a possible neighbor
                    if key[1] > neigh[1]:
                        working = False
        
        # call pysal.lib for `W` instance
        w = weights.W(neighbors, weights=_weights)
        
        return w
    
    
    def distancebandweights(self, threshold,
                            n_proccess=None, gen_tree=False):
        """Create distance based weights.
        
        Parameters
        ----------
        
        threshold : float
            Distance threshold value.
        
        n_processes : {int, str}
            (Optional) Specify the number of cores to utilize. Default
            is 1 core. Use ``int`` to specify an exact number or cores.
            Use ``"all"`` to request all available cores.
        
        gen_tree : bool
            Rebuild shortest path with ``True``, or skip with ``False``.
        
        Returns
        -------
        
        w : `pysal.lib.weights.weights.W <https://pysal.lib.readthedocs.io/en/latest/generated/pysal.lib.weights.W.html#pysal.lib.weights.W>`_
            A ``pysal`` W Object representing the binary adjacency of
            the network.
        
        Examples
        --------
        
        >>> import pysal.explore.spaghetti as spgh
        >>> streets_file = examples.get_path('streets.shp')
        >>> ntw = spgh.Network(in_data=streets_file)
        >>> w = ntw.distancebandweights(threshold=500)
        >>> w.n
        230
        >>> w.histogram[-1]
        (8, 3)
        
        """
        
        # if the a vertex-to-vertex network distance matrix is
        # not present in the `network.Network` object; calculate
        # one at this point
        if not hasattr(self, 'alldistances'):
            self.full_distance_matrix(n_proccess, gen_tree=gen_tree)
        
        # identify all network vertices which are within the
        # `threshold` parameter
        neighbor_query = np.where(self.distancematrix < threshold)
        
        # create an instance for recording neighbors which 
        # inserts a new key if not present in object
        neighbors = defaultdict(list)
        
        # iterate over neighbors within the `threshold`
        # and record all network vertices as neighbors
        # if the vertex is not being compared to itself
        for i, n in enumerate(neighbor_query[0]):
            neigh = neighbor_query[1][i]
            if n != neigh:
                neighbors[n].append(neigh)
        
        # call pysal.lib for `W` instance
        w = weights.W(neighbors)
        
        return w
    
    
    def snapobservations(self, in_data, name,
                         idvariable=None, attribute=None):
        """Snap a point pattern shapefile to network object. The
        point pattern is stored in the ``network.pointpattern['key']``
        attribute of the network object.
        
        Parameters
        ----------
        
        in_data : {geopandas.GeoDataFrame, str}
            The input geographic data. Either (1) a path to a
            shapefile (``str``); or (2) a ``geopandas.GeoDataFrame``.
        
        name : str
            Name to be assigned to the point dataset.
        
        idvariable : str
            Column name to be used as ID variable.
        
        attribute : bool
            Defines whether attributes should be extracted. ``True`` for
            attribute extraction. ``False`` for no attribute extraction.
        
        Examples
        --------
        
        >>> import pysal.explore.spaghetti as spgh
        >>> streets_file = examples.get_path('streets.shp')
        >>> ntw = spgh.Network(in_data=streets_file)
        >>> pt_str = 'crimes'
        >>> in_data = examples.get_path('{}.shp'.format(pt_str))
        >>> ntw.snapobservations(in_data, pt_str, attribute=True)
        >>> ntw.pointpatterns[pt_str].npoints
        287
        
        """
        
        # create attribute of `pointpattern` but instantiating a
        # `network.PointPattern` class
        self.pointpatterns[name] = PointPattern(in_data=in_data,
                                                idvariable=idvariable,
                                                attribute=attribute)
        
        # allocate the point observations to the nework
        self._snap_to_link(self.pointpatterns[name])
    
    
    def compute_distance_to_vertices(self, x, y, arc):
        """Given an observation on a network arc, return the distance
        to the two vertices that bound that end.
        
        Parameters
        ----------
        
        x : float
            x-coordinate of the snapped point.
        
        y : float
            y-coordinate of the snapped point.
        
        arc : tuple
            (vtx0, vtx1) representation of the network arc.
        
        Returns
        -------
        
        d1 : float
            The distance to vtx0. Always the vertex with the lesser id.
        
        d2 : float
            The distance to vtx1. Always the vertex with the greater id.
        
        """
        
        # distance to vertex 1
        d1 = util.compute_length((x, y), self.vertex_coords[arc[0]])
        
        # distance to vertex 2
        d2 = util.compute_length((x, y), self.vertex_coords[arc[1]])
        
        return d1, d2
    
    
    def compute_snap_dist(self, pattern, idx):
        """Given an observation snapped to a network arc, calculate the
        distance from the original location to the snapped location.
        
        Parameters
        -----------
        
        pattern : spaghetti.network.PointPattern
            point pattern object
        
        idx : int
            point id
        
        Returns
        -------
        dist : float
            euclidean distance from original location to snapped
            location.
        """
        
        # set of original (x,y) point coordinates
        loc = pattern.points[idx]['coordinates']
        
        # set of snapped (x,y) point coordinate
        snp = pattern.snapped_coordinates[idx]
        
        # distance from the original location to
        # the snapped location along the network
        dist = util.compute_length(loc, snp)
        
        return dist
    
    
    def _snap_to_link(self, pointpattern):
        """Used internally to snap point observations to network arcs.
        
        Parameters
        -----------
        
        pointpattern : spaghetti.network.PointPattern
            point pattern object
        
        Returns
        -------
        
        obs_to_arc : dict
            Dictionary with arcs as keys and lists of points as values.
        
        arc_to_obs : dict
            Dictionary with point ids as keys and arc tuples as values.
        
        dist_to_vertex : dict
            Dictionary with point ids as keys and values as dicts
            with keys for vertex ids and values as distances from point
            to vertex.
        
        dist_snapped : dict
            Dictionary with point ids as keys and distance from point
            to the network arc which it is snapped.
        
        """
        
        # instantiate observations snapped coordinates lookup
        pointpattern.snapped_coordinates = {}
        
        # record throw-away arcs (pysal.cg.Chain) enumerator
        arcs_ = []
        
        # snapped(point)-to-arc lookup
        s2a = {}
        
        # iterate over network arc ids
        for arc in self.arcs:
            
            # record the start and end of the arc
            head = self.vertex_coords[arc[0]]
            tail = self.vertex_coords[arc[1]]
            
            # create a pysal.cg.Chain object of the arc
            # and add it to the arcs enumerator
            arcs_.append(cg.Chain([head, tail]))
            
            # add the arc into the snapped(point)-to-arc lookup
            s2a[(head, tail)] = arc
        
        # instantiate crosswalks
        points = {}             # point id to coordinates lookup
        obs_to_arc = {}         # observations to arcs lookup
        dist_to_vertex = {}     # distance to vertices lookup
        dist_snapped = {}       # snapped distance lookup
        
        # fetch and records point coordinates keyed by id
        for point_idx, point in pointpattern.points.items():
            points[point_idx] = point['coordinates']
        
        # snap point observations to the network
        snapped = util.snap_points_to_links(points, arcs_)
        
        # record obs_to_arc, dist_to_vertex, and dist_snapped
        
        # iterate over the snapped observation points
        for point_idx, snap_info in snapped.items():
            
            # fetch the x and y coordinate 
            x, y = snap_info[1].tolist()
            
            # look up the arc from snapped(point)-to-arc
            arc = s2a[tuple(snap_info[0])]
            
            # add the arc key to observations to arcs lookup
            if arc not in obs_to_arc:
                obs_to_arc[arc] = {}
            
            # add the (x,y) coordinates of the original observation
            # point location to the observations to arcs lookup
            obs_to_arc[arc][point_idx] = (x, y)
            
            # add the (x,y) coordinates of the snapped observation
            # point location to the snapped coordinates lookup
            pointpattern.snapped_coordinates[point_idx] = (x, y)
            
            # calculate the distance to the left and right vertex
            # along the network link from the snapped point location
            d1, d2 = self.compute_distance_to_vertices(x, y, arc)
            
            # record the distances in the distance to vertices lookup
            dist_to_vertex[point_idx] = {arc[0]: d1,
                                         arc[1]: d2}
            
            # record the snapped distance
            dist_snapped[point_idx] = self.compute_snap_dist(pointpattern,
                                                             point_idx)
        
        # instantiate observations to network vertex lookup
        obs_to_vertex = defaultdict(list)
        
        # iterate over the observations to arcs lookup
        for k, v in obs_to_arc.items():
            
            # record the left and right vertex ids
            keys = v.keys()
            obs_to_vertex[k[0]] = keys
            obs_to_vertex[k[1]] = keys
        
        # set crosswalks as attributes of the `pointpattern` class
        pointpattern.obs_to_arc = obs_to_arc
        pointpattern.dist_to_vertex = dist_to_vertex
        pointpattern.dist_snapped = dist_snapped
        pointpattern.obs_to_vertex = list(obs_to_vertex)
    
    
    def count_per_link(self, obs_on, graph=True):
        """Compute the counts per arc or edge (link).
        
        Parameters
        ----------
        
        obs_on_network : dict
            Dictionary of observations on the network.
            Either {(link):{pt_id:(coords)}} or 
            {link:[(coord),(coord),(coord)]}
        
        Returns
        -------
        counts : dict
            {(link):count}
        
        Examples
        --------
        
        Note that this passes the obs_to_arc or obs_to_edge attribute
        of a point pattern snapped to the network.
        
        >>> import pysal.explore.spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.snapobservations(examples.get_path('crimes.shp'),
        ...                                        'crimes',
        ...                                         attribute=True)
        
        >>> counts = ntw.count_per_link(ntw.pointpatterns['crimes']
        ...                             .obs_to_arc, graph=False)
        >>> counts[(140, 142)]
        10
        
        >>> s = sum([v for v in list(counts.values())])
        >>> s
        287
        
        """
        
        # instantiate observation counts by link lookup
        counts = {}
        
        # graph-theoretic object of nodes and edges
        if graph:
            
            # iterate the links-to-observations lookup
            for key, observations in obs_on.items():
                
                # isolate observation count for the link
                cnt = len(observations)
                
                # extract link (edges) key
                if key in self.arcs_to_edges.keys():
                    key = self.arcs_to_edges[key]
                
                # either add to current count or a dictionary
                # entry or create new dictionary entry
                try:
                    counts[key] += cnt
                except KeyError:
                    counts[key] = cnt
        
        # network object of arcs and vertices
        else:
            
            # simplified version of the above process
            for key in obs_on.keys():
                counts[key] = len(obs_on[key])
        
        return counts
    
    
    def _newpoint_coords(self, arc, distance):
        """Used internally to compute new point
        coordinates during snapping.
        """
        
        # extract coordinates for vertex 1 of arc
        x1 = self.vertex_coords[arc[0]][0]
        y1 = self.vertex_coords[arc[0]][1]
        
        # extract coordinates for vertex 2 of arc
        x2 = self.vertex_coords[arc[1]][0]
        y2 = self.vertex_coords[arc[1]][1]
        
        # if the network are is vertical set the (x) coordinate
        # and proceed to calculating the (y) coordinate
        if x1 == x2:
            x0 = x1
            
            # if the vertical direction is positive from
            # vertex 1 to vertex 2 on the euclidean plane
            if y1 < y2:
                y0 = y1 + distance
            
            # if the vertical direction is negative from
            # vertex 1 to vertex 2 on the euclidean plane
            elif y1 > y2:
                y0 = y2 + distance
            
            # otherwise the link is zero-length
            # -- this should never happen
            else:
                y0 = y1
            
            return x0, y0
        
        # calculate the slope of the arc, `m`
        m = (y2 - y1) / (x2 - x1)
        
        # if the horizontal direction is negative from
        # vertex 1 to vertex 2 on the euclidean plane
        if x1 > x2:
            x0 = x1 - distance / np.sqrt(1 + m**2)
        
        # if the horizontal direction is positive from
        # vertex 1 to vertex 2 on the euclidean plane
        elif x1 < x2:
            x0 = x1 + distance / np.sqrt(1 + m**2)
        
        # calculate the (y) coordinate
        y0 = m * (x0 - x1) + y1
        
        
        # the new (x,y) coordinates for the snapped observation
        return x0, y0
    
    
    def simulate_observations(self, count, distribution='uniform'):
        """Generate a simulated point pattern on the network.
        
        Parameters
        ----------
        
        count : int
            The number of points to create or mean of the distribution
            if not 'uniform'.
        
        distribution : str
            ``{'uniform', 'poisson'}`` distribution of random points.
            If ``"poisson"``, the distribution is calculated from half
            the total network length.
        
        Returns
        -------
        
        random_pts : dict
            Keys are the edge tuple. Values are lists of new
            point coordinates.
        
        Examples
        --------
       
        >>> import pysal.explore.spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.snapobservations(examples.get_path('crimes.shp'),
        ...                                        'crimes',
        ...                                         attribute=True)
       
        >>> npts = ntw.pointpatterns['crimes'].npoints
        >>> sim = ntw.simulate_observations(npts)
        >>> isinstance(sim, spgh.network.SimulatedPointPattern)
        True
        
        """
        
        # instantiate an empty `SimulatedPointPattern()`
        simpts = SimulatedPointPattern()
        
         # record throw-away arcs enumerator
        arcs_ = []
        
        # create array and fill each entry as length of network arc
        lengths = np.zeros(len(self.arc_lengths))
        for i, key in enumerate(self.arc_lengths.keys()):
            arcs_.append(key)
            lengths[i] = self.arc_lengths[key]
        
        # cumulative network length
        stops = np.cumsum(lengths)
        totallength = stops[-1]
        
        # create lengths with a uniform distribution
        if distribution is 'uniform':
            nrandompts = np.random.uniform(0,
                                           totallength,
                                           size=(count,))
        
        # create lengths with a poisson distribution
        elif distribution is 'poisson':
            # calculate poisson from half the network length
            mid_length = totallength / 2.
            nrandompts = np.random.poisson(mid_length,
                                           size=(count,))
        
        # iterate over random distances created above
        for i, r in enumerate(nrandompts):
            
            # take the first element of the index array (arc id)
            # where the random distance is less than that that of
            # its value in `stops`
            idx = np.where(r < stops)[0][0]
            
            # assign the simulated point to the ar
            assignment_arc = arcs_[idx]
            
            # calculate and set the distance from the arc start
            distance_from_start = stops[idx] - r
            
            # populate the coordinates dict
            x0, y0 = self._newpoint_coords(assignment_arc,
                                           distance_from_start)
            
            # record the snapped coordinates and associated vertices
            simpts.snapped_coordinates[i] = (x0, y0)
            simpts.obs_to_vertex[assignment_arc[0]].append(i)
            simpts.obs_to_vertex[assignment_arc[1]].append(i)
            
            # calculate and set the distance from the arc end
            distance_from_end = self.arc_lengths[arcs_[idx]]\
                                - distance_from_start
            
            # populate the distances to vertices
            simpts.dist_to_vertex[i] = {assignment_arc[0]: distance_from_start,
                                        assignment_arc[1]: distance_from_end}
            
            # set snapped coordinates and point count attributes 
            simpts.points = simpts.snapped_coordinates
            simpts.npoints = len(simpts.points)
            
        return simpts
    
    
    def enum_links_vertex(self, v0):
        """Returns the arcs (links) around vertices.
        
        Parameters
        -----------
        
        v0 : int
            vertex id
        
        Returns
        -------
        
        links : list
            List of tuple arcs adjacent to the vertex.
        
        Examples
        --------
        
        >>> import pysal.explore.spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.enum_links_vertex(24)
        [(24, 48), (24, 25), (24, 26)]
        
        """
        
        # instantiate links list
        links = []
        
        neighbor_vertices = self.adjacencylist[v0]
        
        # enumerate links associated with the current vertex
        for n in neighbor_vertices:
            links.append(tuple(sorted([n, v0])))
        
        return links
    
    
    def full_distance_matrix(self, n_processes, gen_tree=False):
        """All vertex-to-vertex distances on a network. This function
        is called from within ``allneighbordistances()``,
        ``nearestneighbordistances()``, and ``distancebandweights()``.
        
        Parameters
        -----------
        
        n_processes : int
            Cpu cores for multiprocessing.
       
        gen_tree : bool
            Rebuild shortest path ``True``, or skip ``False``.
        
        Notes
        -----
        
        Based on :cite:`Dijkstra1959a`.
        
        """
        
        # create `alldistances` attribute which will store
        # the distance cost matrix and path tree
        self.alldistances = {}
        
        # create an empty matrix which will store shortest path distance
        nvtx = len(self.vertex_list)
        self.distancematrix = np.empty((nvtx, nvtx))
        
        # single-core processing
        if not n_processes:
            
            # iterate over each network vertex
            for vtx in self.vertex_list:
                
                # calculate the shortest path and preceding
                # vertices for traversal route
                distance, pred = util.dijkstra(self, vtx)
                pred = np.array(pred)
                
                # generate the shortest path tree
                if gen_tree:
                    tree = util.generatetree(pred)
                else:
                    tree = None
                
                # populate distances and paths
                self.alldistances[vtx] = (distance, tree)
                self.distancematrix[vtx] = distance
        
        # multiprocessing
        if n_processes:
            
            # set up multiprocessing schema
            import multiprocessing as mp
            from itertools import repeat
            if n_processes == 'all':
                cores = mp.cpu_count()
            else:
                cores = n_processes
            p = mp.Pool(processes=cores)
            
            # calculate the shortest path and preceding
            # vertices for traversal route by mapping each process
            distance_pred = p.map(util.dijkstra_mp,
                                  zip(repeat(self), self.vertex_list))
            
            # set range of iterations
            iterations = range(len(distance_pred))
            
            # fill shortest paths
            distance = [distance_pred[itr][0] for itr in iterations]
            
            # fill preceding vertices
            pred = np.array([distance_pred[itr][1] for itr in iterations])
            
            # iterate of network vertices and generate
            # the shortest path tree for each
            for vtx in self.vertex_list:
                if gen_tree:
                    tree = util.generatetree(pred[vtx])
                else:
                    tree = None
                
                # populate distances and paths
                self.alldistances[vtx] = (distance[vtx], tree)
                self.distancematrix[vtx] = distance[vtx]
    
    
    def allneighbordistances(self, sourcepattern, destpattern=None,
                             fill_diagonal=None, n_processes=None,
                             gen_tree=False, snap_dist=False):
        """Compute either all distances between ``i`` and ``j`` in a
        single point pattern or all distances between each ``i`` from a
        source pattern and all ``j`` from a destination pattern.
        
        Parameters
        ----------
        
        sourcepattern : {str, spaghetti.network.PointPattern}
            The key of a point pattern snapped to the network OR
            the full ``spaghetti.network.PointPattern`` object.
        
        destpattern : str
            (Optional) The key of a point pattern snapped to the network
            OR the full ``spaghetti.network.PointPattern`` object.
        
        fill_diagonal : {float, int}
            (Optional) Fill the diagonal of the cost matrix. Default is
            ``None`` and will populate the diagonal with ``numpy.nan``.
            Do not declare a ``destpattern`` for a custom
            ``fill_diagonal``.
        
        n_processes : {int, str}
            (Optional) Specify the number of cores to utilize. Default
            is 1 core. Use ``int`` to specify an exact number or cores.
            Use ``"all"`` to request all available cores.
        
        gen_tree : bool
            Rebuild shortest path ``True``, or skip ``False``.
        
        snap_dist : bool
            Flag as ``True`` to include the distance from the original
            location to the snapped location along the network. Default
            is ``False``.
        
        Returns
        -------
        
        nearest : `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
            An array of shape (n,n) storing distances between all
            points.
        
        tree_nearest : dict
            Nearest network node to point pattern vertex shortest
            path lookup. The values of the dictionary are a ``tuple``
            of the nearest source vertex and the near destination
            vertex to query the lookup tree. If two observations are
            snapped to the same network arc a flag of -.1 is set for
            both the source and destination network vertex
            indicating the same arc is used while also raising an
            ``IndexError`` when rebuilding the path.
        
        Examples
        --------
        
        >>> import pysal.explore.spaghetti as spgh
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
        
        # calculate the network vertex to vertex distance matrix
        # if it is not already an attribute
        if not hasattr(self, 'alldistances'):
            self.full_distance_matrix(n_processes, gen_tree=gen_tree)
        
        # set the source and destination observation point patterns
        if type(sourcepattern) is str:
            sourcepattern = self.pointpatterns[sourcepattern]
            if destpattern:
                destpattern = self.pointpatterns[destpattern]
        
        # source pattern setup
        # set local copy of source pattern index
        src_indices = list(sourcepattern.points.keys())
        # set local copy of source distance to vertex lookup
        src_d2v = copy.deepcopy(sourcepattern.dist_to_vertex)
        # source point count
        nsource_pts = len(src_indices)
        # create source point to network vertex lookup
        src_vertices = {}
        for s in src_indices:
            v1, v2 = src_d2v[s].keys()
            src_vertices[s] = (v1, v2)
        
        # destination pattern setup
        # if only a source pattern is specified, also set it as
        # the destination pattern
        symmetric = False
        if destpattern is None:
            symmetric = True
            destpattern = sourcepattern
        # set local copy of destination pattern index
        dest_indices = list(destpattern.points.keys())
        # set local copy of destination distance to vertex lookup
        dst_d2v = copy.deepcopy(destpattern.dist_to_vertex)
        # destination point count
        ndest_pts = len(dest_indices)
        # create `deepcopy` of destination points to
        # consider for searching
        dest_searchpts = copy.deepcopy(dest_indices)
        # create destination point to network vertex lookup
        dest_vertices = {}
        for s in dest_indices:
            v1, v2 = dst_d2v[s].keys()
            dest_vertices[s] = (v1, v2)
        
        # add snapping distance to each pointpattern
        if snap_dist:
            # declare both point patterns and both
            # distance to vertex lookup in single lists
            patterns = [sourcepattern, destpattern]
            dist_copies = [src_d2v, dst_d2v]
            # iterate over each point pattern
            for elm, pp in enumerate(patterns):
                # extract associated vertex distances
                for pidx, dists_dict in dist_copies[elm].items():
                    # add snapped distance to each point
                    for vidx, vdist in dists_dict.items():
                        dists_dict[vidx] = vdist + pp.dist_snapped[pidx]
        
        # output setup
        # create empty source x destination array
        # and fill with infinity values
        nearest = np.empty((nsource_pts, ndest_pts))
        nearest[:] = np.inf
        # create empty dictionary to store path trees
        tree_nearest = {}
        
        # iterate over each point in sources
        for p1 in src_indices:
            
            # get the source vertices and dist to source vertices
            source1, source2 = src_vertices[p1]
            set1 = set(src_vertices[p1])
            
            # distance from source vertex1 to point and 
            # distance from source vertex2 to point
            sdist1, sdist2 = src_d2v[p1].values()
            
            if symmetric:
                
                # only compute the upper triangle if symmetric
                dest_searchpts.remove(p1)
            
            # iterate over each point remaining in destinations
            for p2 in dest_searchpts:
                
                # get the destination vertices and
                # dist to destination vertices
                dest1, dest2 = dest_vertices[p2]
                set2 = set(dest_vertices[p2])
                
                # when the observations are snapped to the same arc
                if set1 == set2:
                    
                    # calculate only the length between points along
                    # that arc
                    x1, y1 = sourcepattern.snapped_coordinates[p1]
                    x2, y2 = destpattern.snapped_coordinates[p2]
                    
                    computed_length = util.compute_length((x1, y1),
                                                          (x2, y2))
                    nearest[p1, p2] = computed_length
                    
                    # set the nearest network vertices to a flag of -.1
                    # indicating the same arc is used while also raising
                    # and indexing error when rebuilding the path
                    tree_nearest[p1, p2] = (-.1, -.1)
                
                # otherwise lookup distance between the source and
                # destination vertex from the `distancematrix`
                else:
                    
                    # distance from destination vertex1 to point and
                    # distance from destination vertex2 to point
                    ddist1, ddist2 = dst_d2v[p2].values()
                    
                    # set the four possible combinations of 
                    # source to destination shortest path traversal
                    d11 = self.distancematrix[source1][dest1]
                    d21 = self.distancematrix[source2][dest1]
                    d12 = self.distancematrix[source1][dest2]
                    d22 = self.distancematrix[source2][dest2]
                    
                    # find the shortest distance from the path passing
                    # through each of the two origin vertices to the
                    # first destination vertex
                    sd_1 = d11 + sdist1
                    sd_21 = d21 + sdist2
                    sp_combo1 = source1, dest1
                    if sd_1 > sd_21:
                        sd_1 = sd_21
                        sp_combo1 = source2, dest1
                    
                    # now add the point to vertex1 distance on
                    # the destination arc
                    len_1 = sd_1 + ddist1
                    
                    # repeat the prior but now for the paths entering
                    # at the second vertex of the second arc
                    sd_2 = d12 + sdist1
                    sd_22 = d22 + sdist2
                    sp_combo2 = source1, dest2
                    if sd_2 > sd_22:
                        sd_2 = sd_22
                        sp_combo2 = source2, dest2
                    len_2 = sd_2 + ddist2
                    
                    # now find the shortest distance path between point
                    # 1 on arc 1 and point 2 on arc 2, and assign
                    sp_12 = len_1
                    s_vertex, d_vertex = sp_combo1
                    if len_1 > len_2:
                        sp_12 = len_2
                        s_vertex, d_vertex = sp_combo2
                    
                    # set distance and path tree
                    nearest[p1, p2] = sp_12
                    tree_nearest[p1, p2] = (s_vertex, d_vertex)
                    
                if symmetric:
                    
                    # mirror the upper and lower triangle
                    # when symmetric
                    nearest[p2, p1] = nearest[p1, p2]
                    
        # populate the main diagonal when symmetric
        if symmetric:
            
            # fill the matrix diagonal with NaN values is no fill
            # value is specified
            if fill_diagonal is None:
                np.fill_diagonal(nearest, np.nan)
            
            # otherwise fill with specified value
            else:
                np.fill_diagonal(nearest, fill_diagonal)
        
        # if the nearest path tree is desired return it along
        # with the cost matrix
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
            (Optional) The key of a point pattern snapped to the
            network.
        
        n_processes : {int, str}
            (Optional) Specify the number of cores to utilize. Default
            is 1 core. Use ``int`` to specify an exact number or cores.
            Use ``"all"`` to request all available cores.
        
        gen_tree : bool
            Rebuild shortest path ``True``, or skip ``False``.
        
        all_dists : `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
            An array of shape (n,n) storing distances between all
            points.
        
        snap_dist : bool
            Flag as ``True`` to include the distance from the original
            location to the snapped location along the network. Default
            is ``False``.
        
        keep_zero_dist : bool
            Include zero values in minimum distance ``True`` or exclude
            ``False``. Default is ``True``. If the source pattern is the
            same as the destination pattern the diagonal is filled with
            ``numpy.nan``.
        
        Returns
        -------
        nearest : dict
            key is source point id, value is tuple of list containing
            nearest destination point ids and distance.
        
        Examples
        --------
        
        >>> import pysal.explore.spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> ntw.snapobservations(examples.get_path('crimes.shp'),
        ...                      'crimes')
        >>> nn = ntw.nearestneighbordistances('crimes',
        ...                                   keep_zero_dist=True)
        >>> nn[11], nn[18]
        (([18, 19], 165.33982412719126), ([19], 0.0))
        
        >>> nn = ntw.nearestneighbordistances('crimes',
        ...                                   keep_zero_dist=False)
        >>> nn[11], nn[18]
        (([18, 19], 165.33982412719126), ([11], 165.33982412719126))
        
        """
        
        # raise exception is the specified point pattern does not exist
        if sourcepattern not in self.pointpatterns.keys():
            err_msg = 'Available point patterns are {}'
            raise KeyError(err_msg.format(self.pointpatterns.keys()))
        
        # calculate the network vertex to vertex distance matrix
        # if it is not already an attribute
        if not hasattr(self, 'alldistances'):
            self.full_distance_matrix(n_processes, gen_tree=gen_tree)
        
        # determine if the source and destination patterns are equal
        symmetric = sourcepattern != destpattern
        
        # (for source-to-source patterns) if zero-distance neighbors are
        # desired, keep the diagonal as NaN and take the minimum
        # distance neighbor(s), which may include zero distance
        # neighors.
        fill_diagonal = None
        if not keep_zero_dist and symmetric:
            # (for source-to-source patterns) if zero-distance neighbors
            # should be ignored, convert the diagonal to 0.0 and take
            # the minimum distance neighbor(s) that is/are not 0.0
            # distance.
            fill_diagonal = 0.
        
        # set the source and destination observation point patterns
        sourcepattern = self.pointpatterns[sourcepattern]
        if destpattern:
            destpattern = self.pointpatterns[destpattern]
        
        # if the full source to destination is not calculated,
        # do that at this time
        if all_dists is None:
            all_dists = self.allneighbordistances(sourcepattern,
                                                  destpattern=destpattern,
                                                  fill_diagonal=fill_diagonal,
                                                  n_processes=n_processes,
                                                  gen_tree=gen_tree,
                                                  snap_dist=snap_dist)
        
        # create empty nearest neighbors lookup
        nearest = {}
        
        # iterate over each source point
        for source_index in sourcepattern.points.keys():
            
            # this considers all zero-distance neighbors
            if keep_zero_dist and symmetric:
                val = np.nanmin(all_dists[source_index,:])
            
            # this does not consider zero-distance neighbors
            else:
                val = np.min(all_dists[source_index,:]\
                                      [np.nonzero(all_dists[source_index,:])])
            
            # nearest destination (may be more than one if
            # observations are equal distances away)
            dest_idxs = np.where(all_dists[source_index,:] == val)[0].tolist()
            
            # set nearest destination point(s) and distance
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
            The number of steps at which the count of the nearest
            neighbors is computed.
        
        permutations : int
            The number of permutations to perform. Default 99.
        
        threshold : float
            The level at which significance is computed.
            (0.5 would be 97.5% and 2.5%).
        
        distribution : str
            The distribution from which random points are sampled.
            Either ``"uniform"`` or ``"poisson"``.
        
        lowerbound : float
            The lower bound at which the F-function is computed.
            Default 0.
        
        upperbound : float
            The upper bound at which the F-function is computed.
            Defaults to the maximum observed nearest neighbor distance.
        
        Returns
        -------
        
        NetworkF : spaghetti.analysis.NetworkF
            A network F class instance.
        
        Examples
        --------
        
        >>> import pysal.explore.spaghetti as spgh
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
        
        # call analysis.NetworkF
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
            The number of steps at which the count of the nearest
            neighbors is computed.
        
        permutations : int
            The number of permutations to perform. Default 99.
        
        threshold : float
            The level at which significance is computed.
            (0.5 would be 97.5% and 2.5%).
        
        distribution : str
            The distribution from which random points are sampled
            Either ``"uniform"`` or ``"poisson"``.
        
        lowerbound : float
            The lower bound at which the G-function is computed.
            Default 0.
        
        upperbound : float
            The upper bound at which the G-function is computed.
            Defaults to the maximum observed nearest neighbor distance.
        
        Returns
        -------
        
        NetworkG : spaghetti.analysis.NetworkG
            A network G class instance.
        
        Examples
        --------
        
        >>> import pysal.explore.spaghetti as spgh
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
        
        # call analysis.NetworkG
        return NetworkG(self, pointpattern, nsteps=nsteps,
                        permutations=permutations, threshold=threshold,
                        distribution=distribution, lowerbound=lowerbound,
                        upperbound=upperbound)
    
    
    def NetworkK(self, pointpattern, nsteps=10, permutations=99,
                 threshold=0.5, distribution='uniform',
                 lowerbound=None, upperbound=None):
        """Computes a network constrained K-Function
        
        Parameters
        ----------
        
        pointpattern : spaghetti.network.PointPattern
            A spaghetti point pattern object.
        
        nsteps : int
            The number of steps at which the count of the nearest
            neighbors is computed.
        
        permutations : int
            The number of permutations to perform. Default is 99.
        
        threshold : float
            The level at which significance is computed.
            (0.5 would be 97.5% and 2.5%).
        
        distribution : str
            The distribution from which random points are sampled
            Either ``"uniform"`` or ``"poisson"``.
        
        lowerbound : float
            The lower bound at which the K-function is computed.
            Default is 0.
        
        upperbound : float
            The upper bound at which the K-function is computed.
            Defaults to the maximum observed nearest neighbor distance.
        
        Returns
        -------
        
        NetworkK : spaghetti.analysis.NetworkK
            A network K class instance.
        
        Notes
        -----
        
        Based on :cite:`Okabe2001`.
        
        Examples
        --------
        
        >>> import pysal.explore.spaghetti as spgh
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
        
        # call analysis.NetworkK
        return NetworkK(self, pointpattern, nsteps=nsteps,
                        permutations=permutations, threshold=threshold,
                        distribution=distribution, lowerbound=lowerbound,
                        upperbound=upperbound)
    
    
    def split_arcs(self, distance):
        """Split all of the arcs in the network at either a
        fixed distance or a fixed number of arcs.
        
        Parameters
        -----------
        
        distance : float
            The distance at which arcs are split.
        
        Returns
        -------
        
        split_network : spaghetti.Network
            newly instantiated ``spaghetti.Network`` object.
        
       Examples
        --------
       
        >>> import pysal.explore.spaghetti as spgh
        >>> ntw = spgh.Network(examples.get_path('streets.shp'))
        >>> n200 = ntw.split_arcs(200.0)
        >>> len(n200.arcs)
        688
        
        """
        
        # create new shell network instance
        split_network = Network()
        
        # duplicate input network attributes
        split_network.adjacencylist = copy.deepcopy(self.adjacencylist)
        split_network.arc_lengths = copy.deepcopy(self.arc_lengths)
        split_network.arcs = set(copy.deepcopy(self.arcs))
        split_network.vertex_coords = copy.deepcopy(self.vertex_coords)
        split_network.vertex_list = copy.deepcopy(self.vertex_list)
        split_network.vertices = copy.deepcopy(self.vertices)
        split_network.pointpatterns = copy.deepcopy(self.pointpatterns)
        split_network.in_data = self.in_data
        
        # set vertex id to start iterations
        current_vertex_id = max(self.vertices.values())
        
        # instantiate sets for newly created network arcs and
        # input network arcs to remove
        new_arcs = set()
        remove_arcs = set()
        
        # iterate over all network arcs
        for arc in split_network.arcs:
            
            # fetch network arc length
            length = split_network.arc_lengths[arc]
            
            # set initial segmentation interval
            interval = distance
            
            # initialize arc new arc length at zero
            totallength = 0
            
            # initialize the starting vertex, current vertex, and
            # ending vertex
            currentstart = start_vertex = arc[0]
            end_vertex = arc[1]
            
            # if the arc will be split remove the current
            # arc from the adjacency list
            if interval < length:
                
                # remove old arc adjacency information
                split_network.adjacencylist[arc[0]].remove(arc[1])
                split_network.adjacencylist[arc[1]].remove(arc[0])
                
                # remove old arc length information
                split_network.arc_lengths.pop(arc, None)
                
                # add old arc to set of arcs to remove
                remove_arcs.add(arc)
            
            # if the arc will not be split, do nothing and continue
            else:
                continue
            
            # traverse the length of the arc
            while totallength < length:
                
                # set/update the current vertex id
                currentstop = current_vertex_id
                
                # once an can not be split further
                if totallength + interval > length:
                    # record the ending vertex
                    currentstop = end_vertex
                    # set the length remainder
                    interval = length - totallength
                    # full old length reached
                    totallength = length
                
                else:
                    # set the current vertex id
                    current_vertex_id += 1
                    # set the current stopping id
                    currentstop = current_vertex_id
                    # add the interval distance to the traversed length
                    totallength += interval
                    
                    # compute the new vertex coordinate
                    newx, newy = self._newpoint_coords(arc, totallength)
                    
                    # update the vertex
                    if currentstop not in split_network.vertex_list:
                        split_network.vertex_list.append(currentstop)
                    
                    # update vertex coordinates and vertex id
                    split_network.vertex_coords[currentstop] = newx, newy
                    split_network.vertices[(newx, newy)] = currentstop
                
                # update the new network adjacency list
                split_network.adjacencylist[currentstart].append(currentstop)
                split_network.adjacencylist[currentstop].append(currentstart)
                
                # add the new arc to the arc dictionary
                # iterating over this so we need to add after iterating
                new_arcs.add(tuple(sorted([currentstart, currentstop])))
                
                # modify arc_lengths
                current_start_stop = tuple(sorted([currentstart,
                                                   currentstop]))
                
                # set the length of the arc
                split_network.arc_lengths[current_start_stop] = interval
                
                # increment the starting vertex to the stopping vertex
                currentstart = currentstop
        
        # add the newly created arcs to the network
        split_network.arcs.update(new_arcs)
        
        # remove the old arcs the network
        split_network.arcs.difference_update(remove_arcs)
        split_network.arcs = list(split_network.arcs)
        
        # update the snapped point pattern
        for instance in split_network.pointpatterns.values():
            split_network._snap_to_link(instance)
        
        return split_network
    
    
    def savenetwork(self, filename):
        """Save a network to disk as a binary file.
        
        Parameters
        ----------
        
        filename : str
            The filename where the network should be saved. This should
            be a full path or it will be save in the current directory.
        
        Examples
        --------
        
        >>> import pysal.explore.spaghetti as spgh
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


def element_as_gdf(net, vertices=False, arcs=False, pp_name=None,
                   snapped=False, id_col='id', geom_col='geometry'):
    """Return a `geopandas.GeoDataFrame 
    <http://geopandas.org/data_structures.html#geodataframe>`_ of
    network elements. This can be (a) the vertices of a network; (b) the
    arcs of a network; (c) both the vertices and arcs of the network;
    (d) raw point pattern associated with the network; or (e) snapped
    point pattern of (d).
    
    Parameters
    ----------
    
    net : spaghetti.Network
        network object
    
    vertices : bool
        Extract the network vertices. Default is ``False``.
    
    arcs : bool
        Extract the network arcs. Default is ``False``.
    
    pp_name : str
        Name of the network ``PointPattern`` to extract.
        Default is ``None``.
    
    snapped : bool
        If extracting a network ``PointPattern``, set to ``True`` for
        snapped point locations along the network. Default is ``False``.
    
    id_col : str
        GeoDataFrame column name for IDs. Default is ``'id'``.
    
    geom_col : str
        GeoDataFrame column name for geometry. Default is
        ``'geometry'``.
    
    Raises
    ------
    
    KeyError
        In order to extract a ``PointPattern`` it must already be a part
        of the ``spaghetti.Network`` object. This exception is raised
        when a ``PointPattern`` is being extracted that does not exist
        within the ``spaghetti.Network`` object.
    
    Returns
    -------
    
    points : geopandas.GeoDataFrame
        Network point elements (either vertices or ``PointPattern``
        points) as a `geopandas.GeoDataFrame` of ``shapely.Point``
        objects with an ``id`` column and ``geometry`` column.
    
    lines : geopandas.GeoDataFrame
        Network arc elements as a ``geopandas.GeoDataFrame`` of
        ``shapely.LineString`` objects with an ``id`` column and
        ``geometry`` column.
    
    Notes
    -----
    
    This function requires `geopandas <http://geopandas.org>`_.
    
    """
    
    # need vertices place holder to create network segment LineStrings
    # even if only network edges are desired.
    vertices_for_arcs = False
    if arcs and not vertices:
        vertices_for_arcs = True

    # vertices/nodes/points
    if vertices or vertices_for_arcs or pp_name:
        points = util._points_as_gdf(net,
                                     vertices,
                                     vertices_for_arcs,
                                     pp_name, snapped,
                                     id_col=id_col,
                                     geom_col=geom_col)
        
        # return points geodataframe if arcs not specified or
        # if extracting `PointPattern` points
        if not arcs or pp_name:
            return points
    
    # arcs
    arcs = util._arcs_as_gdf(net,
                             points,
                             id_col=id_col,
                             geom_col=geom_col)
    
    if vertices_for_arcs:
        return arcs
    
    else:
        return points, arcs


class PointPattern():
    """A stub point pattern class used to store a point pattern. This
    class is monkey patched with network specific attributes when the
    points are snapped to a network. In the future this class may be
    replaced with a generic point pattern class.
    
    Parameters
    ----------
    
    in_data : {geopandas.GeoDataFrame, str}
        The input geographic data. Either (1) a path to a shapefile
        ``str``; or (2) a `geopandas.GeoDataFrame 
        <http://geopandas.org/data_structures.html#geodataframe>`_.
        
    idvariable : str
        Field in the shapefile to use as an id variable.
    
    attribute :  bool
        A flag to indicate whether all attributes are tagged to this
        class (``True``) or excluded (``False``). Default is ``False``.
    
    Attributes
    ----------
    
    points : dict
        Keys are the point ids (int). Values are the x,y
        coordinates (tuple).
    
    npoints : int
        The number of points.
    
    obs_to_arc : dict
        Keys are arc ids (tuple). Values are snapped point information
        (``dict``).  Within the snapped point information (``dict``)
        keys are observation ids (``int``), and values are snapped
        coordinates.
    
    obs_to_vertex : list
       List of incident network vertices to snapped observation points
       converted from a ``default_dict``. Originally in the form of
       paired left/right nearest network vertices {netvtx1: obs_id1,
       netvtx2: obs_id1, netvtx1: obs_id2... netvtx1: obs_idn}, then
       simplified to a list in the form
       [netvtx1, netvtx2, netvtx1, netvtx2, ...].
       
    dist_to_vertex : dict
        Keys are observations ids (``int``). Values are distance lookup
        (``dict``). Within distance lookup (``dict``) keys are the two
        incident vertices of the arc and values are distance to each of
        those arcs.
    
    snapped_coordinates : dict
        Keys are the point ids (``int``). Values are the snapped x,y
        coordinates (tuple).
    
    snap_dist : bool
            Flag as ``True`` to include the distance from the original
            location to the snapped location along the network. Default
            is ``False``.
    
    """
    
    def __init__(self,
                 in_data=None,
                 idvariable=None,
                 attribute=False):
        
        # initialize points dictionary and counter
        self.points = {}
        self.npoints = 0
        
        # flag for points from a shapefile
        if isinstance(in_data, str):
            from_shp = True
        else:
            from_shp = False
        
        # either set native point id from dataset or create new ids
        if idvariable:
            ids = weights.util.get_ids(in_data,
                                       idvariable)
        else:
            ids = None
        
        # extract the point geometries
        if from_shp:
            pts = open(in_data)
        else:
            pts_objs = list(in_data.geometry)
            pts = [cg.shapes.Point((p.x, p.y)) for p in pts_objs]
        
        
        # fetch attributes if requested
        if attribute:
            
            # open the database file if data is from shapefile
            if from_shp:
                dbname = os.path.splitext(in_data)[0] + '.dbf'
                db = open(dbname)
            
            # if data is from a GeoDataFrame, drop the geometry column
            # and declare attribute values as a list of lists
            else:
                db = in_data.drop(in_data.geometry.name,
                                   axis=1).values.tolist()
                db = [[d] for d in db]
        else:
            db = None
        
        # iterate over all points
        for i, pt in enumerate(pts):
            
            # ids, attributes
            if ids and db is not None:
                self.points[ids[i]] = {'coordinates': pt,
                                       'properties': db[i]}
            
            # ids, no attributes
            elif ids and db is None:
                self.points[ids[i]] = {'coordinates': pt,
                                       'properties': None}
            
            # no ids, attributes
            elif not ids and db is not None:
                self.points[i] = {'coordinates': pt,
                                  'properties': db[i]}
            
            # no ids, no attributes
            else:
                self.points[i] = {'coordinates': pt,
                                  'properties': None}
        
        # close the shapefile and database file
        # if the input data is a .shp
        if from_shp:
            pts.close()
            if db:
                db.close()
        
        # record number of points
        self.npoints = len(self.points.keys())


class SimulatedPointPattern():
    """Struct style class to mirror the ``PointPattern`` class. If the
    ``PointPattern`` class has methods, it might make sense to make this
    a child of that class. This class is not intended to be used by the
    external user.
    
    Attributes
    ----------
    
    npoints : int
        The number of points.
    
    obs_to_arc : dict
        Keys are arc ids (tuple). Values are snapped point information
        (``dict``).  Within the snapped point information (``dict``)
        keys are observation ids (``int``), and values are snapped
        coordinates.
    
    obs_to_vertex : list
       List of incident network vertices to snapped observation points
       converted from a ``default_dict``. Originally in the form of
       paired left/right nearest network vertices {netvtx1: obs_id1,
       netvtx2: obs_id1, netvtx1: obs_id2... netvtx1: obs_idn}, then
       simplified to a list in the form
       [netvtx1, netvtx2, netvtx1, netvtx2, ...].
       
    dist_to_vertex : dict
        Keys are observations ids (``int``). Values are distance lookup
        (``dict``). Within distance lookup (``dict``) keys are the two
        incident vertices of the arc and values are distance to each of
        those arcs.
    
    snapped_coordinates : dict
        Keys are the point ids (``int``). Values are the snapped x,y
        coordinates (tuple).
    
    snap_dist : bool
            Flag as ``True`` to include the distance from the original
            location to the snapped location along the network. Default
            is ``False``.
    
    """
    
    def __init__(self):
        
        # duplicate post-snapping PointPattern class structure
        self.npoints = 0
        self.obs_to_arc = {}
        self.obs_to_vertex = defaultdict(list)
        self.dist_to_vertex = {}
        self.snapped_coordinates = {}


