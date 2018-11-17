.. _api_ref:

.. currentmodule:: spaghetti

API reference
=============

.. _network_api:


spaghetti.Network
-----------------

.. autosummary::
   :toctree: generated/

    Network
    Network.extractgraph
    Network.contiguityweights
    Network.distancebandweights
    Network.snapobservations
    Network.compute_distance_to_nodes
    Network.compute_snap_dist
    Network.count_per_edge
    Network.simulate_observations
    Network.enum_links_node
    Network.node_distance_matrix
    Network.allneighbordistances
    Network.nearestneighbordistances
    Network.NetworkF
    Network.NetworkG
    Network.NetworkK
    Network.segment_edges
    Network.savenetwork
    Network.loadnetwork


spaghetti.NetworkBase
---------------------

.. autosummary::
   :toctree: generated/

    NetworkBase
    NetworkBase.computeenvelope
    NetworkBase.setbounds
    NetworkBase.validatedistribution


spaghetti.NetworkF
---------------------

.. autosummary::
   :toctree: generated/

    NetworkF
    NetworkF.computeenvelope
    NetworkF.setbounds
    NetworkF.validatedistribution
    NetworkF.computeobserved
    NetworkF.computepermutations

spaghetti.NetworkG
---------------------

.. autosummary::
   :toctree: generated/

    NetworkG
    NetworkG.computeenvelope
    NetworkG.setbounds
    NetworkG.validatedistribution
    NetworkG.computeobserved
    NetworkG.computepermutations

spaghetti.NetworkK
---------------------

.. autosummary::
   :toctree: generated/

    NetworkK
    NetworkK.computeenvelope
    NetworkK.setbounds
    NetworkK.validatedistribution
    NetworkK.computeobserved
    NetworkK.computepermutations

spaghetti.PointPattern
----------------------

.. autosummary::
   :toctree: generated/
   
   PointPattern
   

spaghetti.SimulatedPointPattern
-------------------------------

.. autosummary::
   :toctree: generated/
   
    SimulatedPointPattern

spaghetti
-------------------------------

.. autosummary::
   :toctree: generated/
   
    spaghetti.compute_length
    spaghetti.dijkstra
    spaghetti.dijkstra_mp
    spaghetti.generatetree
    spaghetti.get_neighbor_distances
    spaghetti.snap_points_on_segments
    spaghetti.squared_distance_point_segment
    spaghetti.ffunction
    spaghetti.gfunction
    spaghetti.kfunction

