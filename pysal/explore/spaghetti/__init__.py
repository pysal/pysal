__version__ = "1.1.0"
"""
:mod:`spaghetti` --- Spatial Graphs: Networks, Topology, & Inference
====================================================================

"""
from .network import Network, PointPattern, SimulatedPointPattern, SortedEdges
from .analysis import NetworkBase, NetworkG, NetworkK, NetworkF
from .analysis import gfunction, kfunction, ffunction
from .util import compute_length, get_neighbor_distances, generatetree
from .util import dijkstra, dijkstra_mp
from .util import squared_distance_point_segment, snap_points_on_segments