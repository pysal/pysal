.. _api_ref:
=============
API Reference
=============

This is the class and function reference of pysal.



:mod:`pysal.explore`: Exploratory spatial data analysis
=======================================================

.. automodule:: pysal.explore
    :no-members:
    :no-inherited-members:

pysal.explore.esda: Spatial Autocorrelation Analysis
++++++++++++++++++++++++++++++++++++++
.. currentmodule:: pysal.explore.esda


Gamma Statistic
---------------

.. autosummary::
   :toctree: generated/

    pysal.explore.esda.Gamma

.. _geary_api:

Geary Statistic
---------------

.. autosummary::
   :toctree: generated/

    pysal.explore.esda.Geary


.. _getis_api:

Getis-Ord Statistics
--------------------

.. autosummary::
   :toctree: generated/

    pysal.explore.esda.G
    pysal.explore.esda.G_Local

.. _join_api:

Join Count Statistics
---------------------

.. autosummary::
   :toctree: generated/

    pysal.explore.esda.Join_Counts

Moran Statistics
----------------

.. autosummary::
   :toctree: generated/

    pysal.explore.esda.Moran
    pysal.explore.esda.Moran_BV
    pysal.explore.esda.Moran_BV_matrix
    pysal.explore.esda.Moran_Local
    pysal.explore.esda.Moran_Local_BV
    pysal.explore.esda.Moran_Rate
    pysal.explore.esda.Moran_Local_Rate



pysal.explore.giddy: Geospatial Distribution Dynamics
+++++++++++++++++++++++++++++++++++++++
.. currentmodule:: pysal.explore.giddy


.. _markov_api:

Markov Methods
--------------

.. autosummary::
   :toctree: generated/

    pysal.explore.giddy.markov.Markov
    pysal.explore.giddy.markov.Spatial_Markov
    pysal.explore.giddy.markov.LISA_Markov
    pysal.explore.giddy.markov.kullback
    pysal.explore.giddy.markov.prais
    pysal.explore.giddy.markov.homogeneity
    pysal.explore.giddy.ergodic.steady_state
    pysal.explore.giddy.ergodic.fmpt
    pysal.explore.giddy.ergodic.var_fmpt


.. _directional_api:

Directional LISA
----------------

.. autosummary::
   :toctree: generated/

    pysal.explore.giddy.directional.Rose


.. _mobility_api:

Economic Mobility Indices
-------------------------
.. autosummary::
   :toctree: generated/

    pysal.explore.giddy.mobility.markov_mobility

.. _rank_api:

Exchange Mobility Methods
-------------------------
.. autosummary::
   :toctree: generated/

    pysal.explore.giddy.rank.Theta
    pysal.explore.giddy.rank.Tau
    pysal.explore.giddy.rank.SpatialTau
    pysal.explore.giddy.rank.Tau_Local
    pysal.explore.giddy.rank.Tau_Local_Neighbor
    pysal.explore.giddy.rank.Tau_Local_Neighborhood
    pysal.explore.giddy.rank.Tau_Regional



pysal.explore.spaghetti: 
+++++++++++++++++++++++++
.. currentmodule:: pysal.explore.spaghetti



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



:mod:`pysal.viz`: Geovisualization
=======================================================

.. automodule:: pysal.viz
    :no-members:
    :no-inherited-members:


pysal.explore.pysal.viz.mapclassify: Choropleth map classification
+++++++++++++++++++++++++++++++++++++++
.. currentmodule:: pysal.viz.mapclassify

.. _classifiers_api:

Classifiers
-----------

.. autosummary::
   :toctree: generated/

    pysal.viz.mapclassify.Box_Plot
    pysal.viz.mapclassify.Equal_Interval
    pysal.viz.mapclassify.Fisher_Jenks
    pysal.viz.mapclassify.Fisher_Jenks_Sampled
    pysal.viz.mapclassify.HeadTail_Breaks
    pysal.viz.mapclassify.Jenks_Caspall
    pysal.viz.mapclassify.Jenks_Caspall_Forced
    pysal.viz.mapclassify.Jenks_Caspall_Sampled
    pysal.viz.mapclassify.Max_P_Classifier
    pysal.viz.mapclassify.Maximum_Breaks
    pysal.viz.mapclassify.Natural_Breaks
    pysal.viz.mapclassify.Quantiles
    pysal.viz.mapclassify.Percentiles
    pysal.viz.mapclassify.Std_Mean
    pysal.viz.mapclassify.User_Defined

Utilities
---------

.. autosummary::
   :toctree: generated/

    pysal.viz.mapclassify.K_classifiers
    pysal.viz.mapclassify.gadf
	
