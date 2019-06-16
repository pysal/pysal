"""
``splot.esda``
===============

Provides visualisations for the `esda` subpackage.
`esda` provides tools for exploratory spatial data analysis that consider the role of space in a distribution of attribute values.

Moran analytics
---------------

.. autosummary::
   :toctree: generated/
   
   moran_scatterplot
   plot_moran_simulation
   plot_moran
   plot_moran_bv_simulation
   plot_moran_bv
   lisa_cluster
   plot_local_autocorrelation
   moran_facet

"""

from ._viz_esda_mpl import (moran_scatterplot,
                            plot_moran_simulation,
                            plot_moran,
                            plot_moran_bv_simulation,
                            plot_moran_bv,
                            lisa_cluster,
                            plot_local_autocorrelation,
                            moran_facet)