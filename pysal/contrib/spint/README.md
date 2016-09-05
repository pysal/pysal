Spatial Interaction Modeling Package
===========================================

The **Sp**atial **Int**eraction Modeling (SpInt) module seeks to provide a collection of tools to study spatial interaction processes and analyze spatial interaction data.

The [initial development](https://github.com/pysal/pysal/wiki/SpInt-Development) of the module was carried out as a Google Summer of Code
project (summer 2016). Documentation of the project progress can be found on the
[project blog](https://tayloroshan.github.io/). 

The module currently supports the calibration of the 'family' of spatial interaction models (Wilson, 1971) which are derived using an entropy maximizing (EM) framework or the equivalent information minimizing (IM) framework. As such, it is able to derive parameters for the following Poisson count models:

Models
------

- unconstrained gravity model
- production-constrained model (origin-constrained)
- attraction-constrained model (destination-constrained)
- doubly-constrained model

Calibration is carried out using iteratively weighted least squares in a generalized linear modleing framework (Cameron & Trivedi, 2013). These model results have been verified against comparable routines laid out in (Fotheringham and O’Kelly, 1989; Willimans and Fotheringham, 1984) and functions avaialble in R such as GL  or Pythons statsmodels. The estimation of the constrained routines are carried out using sparse data strucutres for lower memory overhead and faster computations.

Additional Features
-------------------

- QuasiPoisson model estimation
- Regression-based tests for overdispersion
- Model fit statistics including typical GLM metrics, standardized root mean
  square error, and Sorensen similarit index
- Vector-based Moran's I statistic for testing for spatial autcorrelation in
  spatial interaction data
- Local subset model calibration for mappable sets of parameter estimates and model
  diagnostics
- Three types of spatial interaction spatial weights: origin-destination
  contiguity weights, network-based weights, and vector-based distance weights

In Progress
-----------

- Spatial Autoregressive (Lag) model spatial interaction specification

Future Work
-----------

- Parameter estimation via maximum likelihood and gradient-based optimization
- Zero-inflated Poisson model
- Negative Binomial model/zero-inflated negative binomial model
- Functions to compute competing destinations
- Functions to compute eigenvector spatial filters
- Parameter estimation via neural networks
- Universal (determinsitic) models such as the Radiation model and Inverse
  Population Weighted model

Cameron, C. A. and Trivedi, P. K. (2013). Regression analyis of count data.
Cambridge University Press, 1998. 

Fotheringham, A. S. and O'Kelly, M. E. (1989). Spatial Interaction Models: Formulations and Applications. London: Kluwer Academic Publishers.

Williams, P. A. and A. S. Fotheringham (1984), The Calibration of Spatial Interaction
Models by Maximum Likelihood Estimation with Program SIMODEL, Geographic Monograph
Series, 7, Department of Geography, Indiana University.

Wilson, A. G. (1971). A family of spatial interaction models, and associated developments. Environment and
Planning A, 3, 1–32.
