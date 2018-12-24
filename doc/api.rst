.. _api_ref:
=============
API Reference
=============

This is the class and function reference of pysal.

.. currentmodule:: pysal.lib


:mod:`pysal.lib`: PySAL Core 
=============================


Spatial Weights
---------------

.. autosummary::
   :toctree: generated/

   weights.W

Distance Weights
++++++++++++++++
.. autosummary::
   :toctree: generated/

   weights.DistanceBand
   weights.Kernel
   weights.KNN

Contiguity Weights
++++++++++++++++++

.. autosummary::
   :toctree: generated/

   weights.Queen
   weights.Rook
   weights.Voronoi
   weights.W

spint Weights
+++++++++++++

.. autosummary::
   :toctree: generated/

   weights.WSP
   weights.netW
   weights.mat2L
   weights.ODW
   weights.vecW


Weights Util Classes and Functions
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   weights.block_weights
   weights.lat2W
   weights.comb
   weights.order
   weights.higher_order
   weights.shimbel
   weights.remap_ids
   weights.full2W
   weights.full
   weights.WSP2W
   weights.get_ids
   weights.get_points_array_from_shapefile

Weights user Classes and Functions
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   weights.min_threshold_distance
   weights.lat2SW
   weights.w_local_cluster
   weights.higher_order_sp
   weights.hexLat2W
   weights.attach_islands
   weights.nonplanar_neighbors
   weights.fuzzy_contiguity
   weights.min_threshold_dist_from_shapefile
   weights.build_lattice_shapefile
   weights.spw_from_gal


Set Theoretic Weights
+++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   weights.w_union
   weights.w_intersection
   weights.w_difference
   weights.w_symmetric_difference
   weights.w_subset
   weights.w_clip


Spatial Lag
+++++++++++

.. autosummary::
   :toctree: generated/

   weights.lag_spatial
   weights.lag_categorical
          

cg: Computational Geometry
--------------------------

alpha_shapes
++++++++++++

.. autosummary::
   :toctree: generated/

   cg.alpha_shape
   cg.alpha_shape_auto

voronoi
+++++++

.. autosummary::
   :toctree: generated/

   cg.voronoi_frames


sphere
++++++

.. autosummary::
   :toctree: generated/

   cg.RADIUS_EARTH_KM
   cg.RADIUS_EARTH_MILES
   cg.arcdist
   cg.arcdist2linear
   cg.brute_knn
   cg.fast_knn
   cg.fast_threshold
   cg.linear2arcdist
   cg.toLngLat
   cg.toXYZ
   cg.lonlat
   cg.harcdist
   cg.geointerpolate
   cg.geogrid

shapes
++++++

.. autosummary::
   :toctree: generated/

   cg.Point
   cg.LineSegment
   cg.Line
   cg.Ray
   cg.Chain
   cg.Polygon
   cg.Rectangle
   cg.asShape

standalone
++++++++++

.. autosummary::
   :toctree: generated/

   cg.bbcommon
   cg.get_bounding_box
   cg.get_angle_between
   cg.is_collinear
   cg.get_segments_intersect
   cg.get_segment_point_intersect
   cg.get_polygon_point_intersect
   cg.get_rectangle_point_intersect
   cg.get_ray_segment_intersect
   cg.get_rectangle_rectangle_intersection
   cg.get_polygon_point_dist
   cg.get_points_dist
   cg.get_segment_point_dist
   cg.get_point_at_angle_and_dist
   cg.convex_hull
   cg.is_clockwise
   cg.point_touches_rectangle
   cg.get_shared_segments
   cg.distance_matrix


locators
++++++++

.. autosummary::
   :toctree: generated/

   cg.Grid
   cg.PointLocator
   cg.PolygonLocator


kdtree
++++++

.. autosummary::
   :toctree: generated/

   cg.KDTree


io
-- 

.. autosummary::
   :toctree: generated/

   io.open
   io.fileio.FileIO


examples
--------


.. autosummary::
   :toctree: generated/

   examples.available
   examples.explain
   examples.get_path

.. currentmodule:: pysal.explore

:mod:`pysal.explore`: Exploratory spatial data analysis
=======================================================

pysal.explore.esda: Spatial Autocorrelation Analysis
----------------------------------------------------

.. autosummary::
   :toctree: /generated

Gamma Statistic
+++++++++++++++

.. autosummary::
   :toctree: generated/

   esda.Gamma

.. _geary_api:

Geary Statistic
+++++++++++++++

.. autosummary::
   :toctree: generated/

   esda.Geary


.. _getis_api:

Getis-Ord Statistics
++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   esda.G
   esda.G_Local

.. _join_api:

Join Count Statistics
+++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   esda.Join_Counts

Moran Statistics
++++++++++++++++

.. autosummary::
   :toctree: generated/

   esda.Moran
   esda.Moran_BV
   esda.Moran_BV_matrix
   esda.Moran_Local
   esda.Moran_Local_BV
   esda.Moran_Rate
   esda.Moran_Local_Rate



pysal.explore.giddy: Geospatial Distribution Dynamics
-----------------------------------------------------


.. _markov_api:

Markov Methods
++++++++++++++

.. autosummary::
   :toctree: generated/

   giddy.markov.Markov
   giddy.markov.Spatial_Markov
   giddy.markov.LISA_Markov
   giddy.markov.kullback
   giddy.markov.prais
   giddy.markov.homogeneity
   giddy.ergodic.steady_state
   giddy.ergodic.fmpt
   giddy.ergodic.var_fmpt


.. _directional_api:

Directional LISA
++++++++++++++++

.. autosummary::
   :toctree: generated/

   giddy.directional.Rose


.. _mobility_api:

Economic Mobility Indices
+++++++++++++++++++++++++
.. autosummary::
   :toctree: generated/

    giddy.mobility.markov_mobility

.. _rank_api:

Exchange Mobility Methods
+++++++++++++++++++++++++
.. autosummary::
   :toctree: generated/

    giddy.rank.Theta
    giddy.rank.Tau
    giddy.rank.SpatialTau
    giddy.rank.Tau_Local
    giddy.rank.Tau_Local_Neighbor
    giddy.rank.Tau_Local_Neighborhood
    giddy.rank.Tau_Regional


pysal.explore.inequality: Spatial Inequality Analysis
-----------------------------------------------------

 .. _inequality_api:

Theil Inequality Measures
+++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

    inequality.theil.Theil 
    inequality.theil.TheilD
    inequality.theil.TheilDSim


Gini Inequality Measures
++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

    inequality.gini.Gini_Spatial




pysal.explore.spaghetti: 
-------------------------



.. _network_api:


spaghetti.Network
+++++++++++++++++

.. autosummary::
   :toctree: generated/

   spaghetti.Network
   spaghetti.Network.extractgraph
   spaghetti.Network.contiguityweights
   spaghetti.Network.distancebandweights
   spaghetti.Network.snapobservations
   spaghetti.Network.compute_distance_to_nodes
   spaghetti.Network.compute_snap_dist
   spaghetti.Network.count_per_edge
   spaghetti.Network.simulate_observations
   spaghetti.Network.enum_links_node
   spaghetti.Network.node_distance_matrix
   spaghetti.Network.allneighbordistances
   spaghetti.Network.nearestneighbordistances
   spaghetti.Network.NetworkF
   spaghetti.Network.NetworkG
   spaghetti.Network.NetworkK
   spaghetti.Network.segment_edges
   spaghetti.Network.savenetwork
   spaghetti.Network.loadnetwork


spaghetti.NetworkBase
+++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   spaghetti.NetworkBase
   spaghetti.NetworkBase.computeenvelope
   spaghetti.NetworkBase.setbounds
   spaghetti.NetworkBase.validatedistribution


spaghetti.NetworkF
++++++++++++++++++

.. autosummary::
   :toctree: generated/

   spaghetti.NetworkF
   spaghetti.NetworkF.computeenvelope
   spaghetti.NetworkF.setbounds
   spaghetti.NetworkF.validatedistribution
   spaghetti.NetworkF.computeobserved
   spaghetti.NetworkF.computepermutations

spaghetti.NetworkG
++++++++++++++++++

.. autosummary::
   :toctree: generated/

   spaghetti.NetworkG
   spaghetti.NetworkG.computeenvelope
   spaghetti.NetworkG.setbounds
   spaghetti.NetworkG.validatedistribution
   spaghetti.NetworkG.computeobserved
   spaghetti.NetworkG.computepermutations

spaghetti.NetworkK
++++++++++++++++++

.. autosummary::
   :toctree: generated/

   spaghetti.NetworkK
   spaghetti.NetworkK.computeenvelope
   spaghetti.NetworkK.setbounds
   spaghetti.NetworkK.validatedistribution
   spaghetti.NetworkK.computeobserved
   spaghetti.NetworkK.computepermutations

spaghetti.PointPattern
++++++++++++++++++++++

.. autosummary::
   :toctree: generated/
   
   spaghetti.PointPattern
   

spaghetti.SimulatedPointPattern
+++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/
   
   spaghetti.SimulatedPointPattern

spaghetti
+++++++++

.. autosummary::
   :toctree: generated/
   
   spaghetti.spaghetti.compute_length
   spaghetti.spaghetti.dijkstra
   spaghetti.spaghetti.dijkstra_mp
   spaghetti.spaghetti.generatetree
   spaghetti.spaghetti.get_neighbor_distances
   spaghetti.spaghetti.snap_points_on_segments
   spaghetti.spaghetti.squared_distance_point_segment
   spaghetti.spaghetti.ffunction
   spaghetti.spaghetti.gfunction
   spaghetti.spaghetti.kfunction


.. currentmodule:: pysal.viz

:mod:`pysal.viz`: Geovisualization
==================================


pysal.viz.mapclassify: Choropleth map classification
----------------------------------------------------

.. _classifiers_api:

Classifiers
+++++++++++

.. autosummary::
   :toctree: generated/

   mapclassify.Box_Plot
   mapclassify.Equal_Interval
   mapclassify.Fisher_Jenks
   mapclassify.Fisher_Jenks_Sampled
   mapclassify.HeadTail_Breaks
   mapclassify.Jenks_Caspall
   mapclassify.Jenks_Caspall_Forced
   mapclassify.Jenks_Caspall_Sampled
   mapclassify.Max_P_Classifier
   mapclassify.Maximum_Breaks
   mapclassify.Natural_Breaks
   mapclassify.Quantiles
   mapclassify.Percentiles
   mapclassify.Std_Mean
   mapclassify.User_Defined

Utilities
+++++++++

.. autosummary::
   :toctree: generated/

   mapclassify.K_classifiers
   mapclassify.gadf

pysal.viz.splot: Lightweight visualization interface
----------------------------------------------------


Giddy
+++++

.. autosummary::
   :toctree: generated/

   splot.giddy.dynamic_lisa_heatmap
   splot.giddy.dynamic_lisa_rose
   splot.giddy.dynamic_lisa_vectors
   splot.giddy.lisa_composite
   splot.giddy.lisa_composite_explore


ESDA
++++

.. autosummary::
   :toctree: generated/

   splot.esda.moran_scatterplot
   splot.esda.plot_moran
   splot.esda.plot_moran_simulation
   splot.esda.plot_moran_bv
   splot.esda.plot_moran_bv_simulation
   splot.esda.lisa_cluster
   splot.esda.plot_local_autocorrelation
   splot.esda.moran_facet

Weights
+++++++

.. autosummary::
   :toctree: generated/

   splot.libpysal.plot_spatial_weights


mapping
+++++++

.. autosummary::
   :toctree: generated/

   splot.mapping.value_by_alpha_cmap
   splot.mapping.vba_choropleth
   splot.mapping.vba_legend
   splot.mapping.mapclassify_bin


 
.. currentmodule:: pysal.model
	
:mod:`pysal.model`: Linear models for spatial data analysis
===========================================================


pysal.model.spreg: Spatial Econometrics
---------------------------------------

These are the standard spatial regression models supported by the `spreg` package. Each of them contains a significant amount of detail in their docstring discussing how they're used, how they're fit, and how to interpret the results. 

.. autosummary::
   :toctree: generated/
    
   spreg.OLS
   spreg.ML_Lag
   spreg.ML_Error
   spreg.GM_Lag
   spreg.GM_Error
   spreg.GM_Error_Het
   spreg.GM_Error_Hom
   spreg.GM_Combo
   spreg.GM_Combo_Het
   spreg.GM_Combo_Hom
   spreg.GM_Endog_Error
   spreg.GM_Endog_Error_Het
   spreg.GM_Endog_Error_Hom
   spreg.TSLS
   spreg.ThreeSLS

Regimes Models
++++++++++++++

Regimes models are variants of spatial regression models which allow for structural instability in parameters. That means that these models allow different coefficient values in distinct subsets of the data. 

.. autosummary::
    :toctree: generated/

   spreg.OLS_Regimes
   spreg.ML_Lag_Regimes
   spreg.ML_Error_Regimes
   spreg.GM_Lag_Regimes
   spreg.GM_Error_Regimes
   spreg.GM_Error_Het_Regimes
   spreg.GM_Error_Hom_Regimes
   spreg.GM_Combo_Regimes
   spreg.GM_Combo_Hom_Regimes
   spreg.GM_Combo_Het_Regimes
   spreg.GM_Endog_Error_Regimes
   spreg.GM_Endog_Error_Hom_Regimes
   spreg.GM_Endog_Error_Het_Regimes

Seemingly-Unrelated Regressions
+++++++++++++++++++++++++++++++

Seeimingly-unrelated regression models are a generalization of linear regression. These models (and their spatial generalizations) allow for correlation in the residual terms between groups that use the same model. In spatial Seeimingly-Unrelated Regressions, the error terms across groups are allowed to exhibit a structured type of correlation: spatail correlation. 

.. autosummary::
   :toctree: generated/
    
   spreg.SUR
   spreg.SURerrorGM
   spreg.SURerrorML
   spreg.SURlagIV
   spreg.ThreeSLS


pysal.model.mgwr: Multiscale Geographically Weighted Regression
---------------------------------------------------------------

.. _mgwr_api:

GWR Model Estimation and Inference
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   mgwr.gwr.GWR
   mgwr.gwr.GWRResults
   mgwr.gwr.GWRResultsLite


MGWR Estimation and Inference
+++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

    mgwr.gwr.MGWR
    mgwr.gwr.MGWRResults


Kernel Specification
++++++++++++++++++++

.. autosummary::
   :toctree: generated/

    mgwr.kernels.fix_gauss
    mgwr.kernels.adapt_gauss
    mgwr.kernels.fix_bisquare
    mgwr.kernels.adapt_bisquare
    mgwr.kernels.fix_exp
    mgwr.kernels.adapt_exp

Bandwidth Selection
+++++++++++++++++++

.. autosummary::
   :toctree: generated/

   mgwr.sel_bw.Sel_BW


Visualization
+++++++++++++

.. autosummary::
   :toctree: generated/

   mgwr.utils.shift_colormap
   mgwr.utils.truncate_colormap
   mgwr.utils.compare_surfaces
