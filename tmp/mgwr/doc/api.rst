.. _api_ref:

.. currentmodule:: mgwr

API reference
=============

.. _gwr_api:

GWR Model Estimation and Inference
----------------------------------

.. autosummary::
   :toctree: generated/

    mgwr.gwr.GWR
    mgwr.gwr.GWRResults
    mgwr.gwr.GWRResultsLite


MGWR Estimation and Inference
-----------------------------

.. autosummary::
   :toctree: generated/

    mgwr.gwr.MGWR
    mgwr.gwr.MGWRResults


Utility Functions
-----------------

Kernel Specification
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    mgwr.kernels.fix_gauss
    mgwr.kernels.adapt_gauss
    mgwr.kernels.fix_bisquare
    mgwr.kernels.adapt_bisquare
    mgwr.kernels.fix_exp
    mgwr.kernels.adapt_exp

Bandwidth Selection
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    mgwr.sel_bw.Sel_BW


Visualization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   utils.shift_colormap
   utils.truncate_colormap
   utils.compare_surfaces