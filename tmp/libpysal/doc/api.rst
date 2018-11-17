.. _api_ref:

.. currentmodule:: libpysal


libpysal API reference
======================

Spatial Weights
---------------

.. autosummary::
   :toctree: generated/

   libpysal.weights.W

Distance Weights
++++++++++++++++
.. autosummary::
   :toctree: generated/

   libpysal.weights.DistanceBand
   libpysal.weights.Kernel
   libpysal.weights.KNN

Contiguity Weights
++++++++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.Queen
   libpysal.weights.Rook
   libpysal.weights.Voronoi
   libpysal.weights.W

spint Weights
+++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.WSP
   libpysal.weights.netW
   libpysal.weights.mat2L
   libpysal.weights.ODW
   libpysal.weights.vecW


Weights Util Classes and Functions
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.block_weights
   libpysal.weights.lat2W
   libpysal.weights.comb
   libpysal.weights.order
   libpysal.weights.higher_order
   libpysal.weights.shimbel
   libpysal.weights.remap_ids
   libpysal.weights.full2W
   libpysal.weights.full
   libpysal.weights.WSP2W
   libpysal.weights.get_ids
   libpysal.weights.get_points_array_from_shapefile

Weights user Classes and Functions
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.min_threshold_distance
   libpysal.weights.lat2SW
   libpysal.weights.w_local_cluster
   libpysal.weights.higher_order_sp
   libpysal.weights.hexLat2W
   libpysal.weights.attach_islands
   libpysal.weights.nonplanar_neighbors
   libpysal.weights.fuzzy_contiguity
   libpysal.weights.min_threshold_dist_from_shapefile
   libpysal.weights.build_lattice_shapefile
   libpysal.weights.spw_from_gal


Set Theoretic Weights
+++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.w_union
   libpysal.weights.w_intersection
   libpysal.weights.w_difference
   libpysal.weights.w_symmetric_difference
   libpysal.weights.w_subset
   libpysal.weights.w_clip


Spatial Lag
+++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.lag_spatial
   libpysal.weights.lag_categorical
          

cg: Computational Geometry
--------------------------

alpha_shapes
++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.cg.alpha_shape
   libpysal.cg.alpha_shape_auto

voronoi
+++++++

.. autosummary::
   :toctree: generated/

   libpysal.cg.voronoi_frames


sphere
++++++

.. autosummary::
   :toctree: generated/

   libpysal.cg.RADIUS_EARTH_KM
   libpysal.cg.RADIUS_EARTH_MILES
   libpysal.cg.arcdist
   libpysal.cg.arcdist2linear
   libpysal.cg.brute_knn
   libpysal.cg.fast_knn
   libpysal.cg.fast_threshold
   libpysal.cg.linear2arcdist
   libpysal.cg.toLngLat
   libpysal.cg.toXYZ
   libpysal.cg.lonlat
   libpysal.cg.harcdist
   libpysal.cg.geointerpolate
   libpysal.cg.geogrid

shapes
++++++

.. autosummary::
   :toctree: generated/

   libpysal.cg.Point
   libpysal.cg.LineSegment
   libpysal.cg.Line
   libpysal.cg.Ray
   libpysal.cg.Chain
   libpysal.cg.Polygon
   libpysal.cg.Rectangle
   libpysal.cg.asShape

standalone
++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.cg.bbcommon
   libpysal.cg.get_bounding_box
   libpysal.cg.get_angle_between
   libpysal.cg.is_collinear
   libpysal.cg.get_segments_intersect
   libpysal.cg.get_segment_point_intersect
   libpysal.cg.get_polygon_point_intersect
   libpysal.cg.get_rectangle_point_intersect
   libpysal.cg.get_ray_segment_intersect
   libpysal.cg.get_rectangle_rectangle_intersection
   libpysal.cg.get_polygon_point_dist
   libpysal.cg.get_points_dist
   libpysal.cg.get_segment_point_dist
   libpysal.cg.get_point_at_angle_and_dist
   libpysal.cg.convex_hull
   libpysal.cg.is_clockwise
   libpysal.cg.point_touches_rectangle
   libpysal.cg.get_shared_segments
   libpysal.cg.distance_matrix


locators
++++++++

.. autosummary::
   :toctree: generated/

   libpysal.cg.Grid
   libpysal.cg.PointLocator
   libpysal.cg.PolygonLocator


kdtree
++++++

.. autosummary::
   :toctree: generated/

   libpysal.cg.KDTree


io
-- 

.. autosummary::
   :toctree: generated/

   libpysal.io.open
   libpysal.io.fileio.FileIO


examples
--------


.. autosummary::
   :toctree: generated/

   libpysal.examples.available
   libpysal.examples.explain
   libpysal.examples.get_path
