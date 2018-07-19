"""
``splot.esda``
===============

Provides visualisations for the `esda` subpackage.
`esda` provides tools for exploratory spatial data analysis that consider the role of space in a distribution of attribute values.

Moran Local analytics
---------------------

.. autosummary::
   :toctree: generated/
   
   moran_scatterplot
   plot_moran_simulation
   plot_moran
   moran_bv_scatterplot
   plot_moran_bv_simulation
   plot_moran_bv
   moran_loc_scatterplot
   lisa_cluster
   plot_local_autocorrelation

"""

from ._viz_esda_mpl import (moran_scatterplot,
                            plot_moran_simulation,
                            plot_moran,
                            moran_bv_scatterplot,
                            plot_moran_bv_simulation,
                            plot_moran_bv,
                            moran_loc_scatterplot,
                            lisa_cluster,
                            plot_local_autocorrelation,
                            moran_loc_bv_scatterplot)