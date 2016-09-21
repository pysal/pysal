**G**eneralized **L**inear **M**odeling
=======================================

This module is an adaptation of a portion of [GLM functionality from the
Statsmodels](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/genmod/generalized_linear_model.py) package, this it has been simplified and customized for the purposes of serving
as the base for several other PySAL modules, namely SpInt and GWR. Currently, it
supports the estimation of Gaussian, Poisson, and Logistic regression using only
iteratively weighted least squares estimation (IWLS). One of the large differences this
module and the functions avaialble in the Statsmodels package is that the custom IWLS routine is fully sparse compatible, which was necesary for the very sparse design matrices that arise in constrained spatial interaction models. The somewhat limited functionality and computation of only a subset of GLM diagnostics also decreases the computational overhead. Another difference is that this module also supports the estimation of QuasiPoisson models. One caveat is that this custom IWLS routine currently generates estimates by directly solves the least squares normal equations rather than using a more robust method like the pseudo inverse. For more robust estimation of ill conditioned data and a fuller GLM framework we suggest using the original GLM functionality from [Statsmodels](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/genmod/generalized_linear_model.py).

Features
--------

- Gaussian GLM
- Poisson GLM
- QuasiPoisson GLM
- Logistic GLM
- Selection of most common GLM diagnostics
- Supports sparse design matrices

Future Work
-----------

- Add Negative Binomial GLM
- Add Gamma GLM
- Add Zero-inflated/Hurdle extensions of Poisson/Negative Binomial
- Add support for gradient based optimization for maximum likelihood estimation



