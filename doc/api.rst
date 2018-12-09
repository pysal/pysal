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

    rank.Theta
    rank.Tau
    rank.SpatialTau
    rank.Tau_Local
    rank.Tau_Local_Neighbor
    rank.Tau_Local_Neighborhood
    rank.Tau_Regional


pysal.explore.inequality: Spatial Inequality Analysis
+++++++++++++++++++++++++
.. currentmodule:: pysal.explore.inequality

 .. _inequality_api:

Theil Inequality Measures
-------------------------

.. autosummary::
   :toctree: generated/

    pysal.explore.inequality.theil.Theil 
    pysal.explore.inequality.theil.TheilD
    pysal.explore.inequality.theil.TheilDSim


Gini Inequality Measures
------------------------

.. autosummary::
   :toctree: generated/

    pysal.explore.inequality.gini.Gini
    pysal.explore.inequality.gini.Gini_Spatial




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


	
:mod:`pysal.model`: Linear models for spatial data analysis
===========================================================

.. automodule:: pysal.model
    :no-members:
    :no-inherited-members:

pysal.model.spreg: Spatial Econometrics
++++++++++++++++++++++++++++++++++++++
.. currentmodule:: pysal.model.spreg

Spatial Regression Models
-------------------------

These are the standard spatial regression models supported by the `spreg` package. Each of them contains a significant amount of detail in their docstring discussing how they're used, how they're fit, and how to interpret the results. 

.. autosummary::
   :toctree: generated/
    
    pysal.model.spreg.OLS
    pysal.model.spreg.ML_Lag
    pysal.model.spreg.ML_Error
    pysal.model.spreg.GM_Lag
    pysal.model.spreg.GM_Error
    pysal.model.spreg.GM_Error_Het
    pysal.model.spreg.GM_Error_Hom
    pysal.model.spreg.GM_Combo
    pysal.model.spreg.GM_Combo_Het
    pysal.model.spreg.GM_Combo_Hom
    pysal.model.spreg.GM_Endog_Error
    pysal.model.spreg.GM_Endog_Error_Het
    pysal.model.spreg.GM_Endog_Error_Hom
    pysal.model.spreg.TSLS
    pysal.model.spreg.ThreeSLS

Regimes Models
---------------

Regimes models are variants of spatial regression models which allow for structural instability in parameters. That means that these models allow different coefficient values in distinct subsets of the data. 

.. autosummary::
    :toctree: generated/

    pysal.model.spreg.OLS_Regimes
    pysal.model.spreg.ML_Lag_Regimes
    pysal.model.spreg.ML_Error_Regimes
    pysal.model.spreg.GM_Lag_Regimes
    pysal.model.spreg.GM_Error_Regimes
    pysal.model.spreg.GM_Error_Het_Regimes
    pysal.model.spreg.GM_Error_Hom_Regimes
    pysal.model.spreg.GM_Combo_Regimes
    pysal.model.spreg.GM_Combo_Hom_Regimes
    pysal.model.spreg.GM_Combo_Het_Regimes
    pysal.model.spreg.GM_Endog_Error_Regimes
    pysal.model.spreg.GM_Endog_Error_Hom_Regimes
    pysal.model.spreg.GM_Endog_Error_Het_Regimes

Seemingly-Unrelated Regressions
--------------------------------

Seeimingly-unrelated regression models are a generalization of linear regression. These models (and their spatial generalizations) allow for correlation in the residual terms between groups that use the same model. In spatial Seeimingly-Unrelated Regressions, the error terms across groups are allowed to exhibit a structured type of correlation: spatail correlation. 

.. autosummary::
   :toctree: generated/
    
    pysal.model.spreg.SUR
    pysal.model.spreg.SURerrorGM
    pysal.model.spreg.SURerrorML
    pysal.model.spreg.SURlagIV
    pysal.model.spreg.ThreeSLS



.. currentmodule:: pysal.model.mgwr


.. _gwr_api:

GWR Model Estimation and Inference
----------------------------------

.. autosummary::
   :toctree: generated/

    pysal.model.mgwr.gwr.GWR
    pysal.model.mgwr.gwr.GWRResults
    pysal.model.mgwr.gwr.GWRResultsLite


MGWR Estimation and Inference
-----------------------------

.. autosummary::
   :toctree: generated/

    pysal.model.mgwr.gwr.MGWR
    pysal.model.mgwr.gwr.MGWRResults

Utility Functions
-----------------

Kernel Specification
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    pysal.model.mgwr.kernels.fix_gauss
    pysal.model.mgwr.kernels.adapt_gauss
    pysal.model.mgwr.kernels.fix_bisquare
    pysal.model.mgwr.kernels.adapt_bisquare
    pysal.model.mgwr.kernels.fix_exp
    pysal.model.mgwr.kernels.adapt_exp

Bandwidth Selection
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    pysal.model.mgwr.sel_bw.Sel_BW


Visualization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pysal.model.mgwr.utils.shift_colormap
   pysal.model.mgwr.utils.truncate_colormap
   pysal.model.mgwr.utils.compare_surfaces
