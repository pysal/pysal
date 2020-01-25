.. _api_ref:

=============
API Reference
=============

This is the class and function reference of pysal.

:mod:`pysal.lib`: PySAL Core 
=============================

Weights
+++++++

Spatial Weights
---------------

:py:class:`libpysal.weights.W`
:py:class:`libpysal.weights.WSP`

Contiguity Weights
------------------

:py:class:`libpysal.weights.Queen`
:py:class:`libpysal.weights.Rook`
:py:class:`libpysal.weights.Voronoi`


Distance Weights
----------------

:py:class:`libpysal.weights.DistanceBand`
:py:class:`libpysal.weights.Kernel`
:py:class:`libpysal.weights.KNN`

Spatial Interaction Weights
---------------------------

:py:func:`libpysal.weights.netW`
:py:func:`libpysal.weights.mat2L`
:py:func:`libpysal.weights.ODW`
:py:func:`libpysal.weights.vecW`


Weights Utilities
-----------------

:py:func:`libpysal.weights.block_weights`
:py:func:`libpysal.weights.lat2W`
:py:func:`libpysal.weights.comb`
:py:func:`libpysal.weights.order`
:py:func:`libpysal.weights.higher_order`
:py:func:`libpysal.weights.shimbel`
:py:func:`libpysal.weights.remap_ids`
:py:func:`libpysal.weights.full2W`
:py:func:`libpysal.weights.full`
:py:func:`libpysal.weights.WSP2W`
:py:func:`libpysal.weights.get_ids`
:py:func:`libpysal.weights.get_points_array_from_shapefile`



