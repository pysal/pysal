"""
``splot.giddy``
===============

Provides visualisations for the Geospatial Distribution Dynamics - `giddy` module.
`giddy` provides a tool for space–time analytics that consider the role of space in the evolution of distributions over time.

Directional LISA analytics
--------------------------

.. autosummary::
   :toctree: generated/

   dynamic_lisa_heatmap
   dynamic_lisa_rose
   dynamic_lisa_vectors
   dynamic_lisa_composite
   dynamic_lisa_composite_explore

"""

from ._viz_giddy_mpl import (dynamic_lisa_heatmap,
                             dynamic_lisa_rose,
                             dynamic_lisa_vectors,
                             dynamic_lisa_composite,
                             dynamic_lisa_composite_explore)