Spatial Interaction Modeling Package
===========================================

The **Sp**atial **Int**eraction Modeling (SpInt) package seeks to provide a collection of tools to study spatial interaction processes.

It currently supports the calibration of the 'family' of spatial interaction models (Wilson, 1971) which are derived using an entropy maximizing (EM) framework or the equivalent information minimizing (IM) framework. As such, it is able to derive parameters for the following models:

- unconstrained gravity model
- production-constrained model (origin-constrained)
- attraction-constrained model (destination-constrained)
- doubly-constrained model


Calibration is carried out using maximum likelihood estimation routines outlined in (Fotheringham and O’Kelly, 1989; Willimans and Fotheringham, 1984). Optimization is achieved using scipy.optimize.fsolve(). Overall, the package is currently dependent upon numpy, spicy, and pandas.

Fotheringham, A. S. and O'Kelly, M. E. (1989). Spatial Interaction Models: Formulations and Applications. London: Kluwer Academic Publishers.

Williams, P. A. and A. S. Fotheringham (1984), The Calibration of Spatial Interaction
Models by Maximum Likelihood Estimation with Program SIMODEL, Geographic Monograph
Series, 7, Department of Geography, Indiana University.

Wilson, A. G. (1971). A family of spatial interaction models, and associated developments. Environment and
Planning A, 3, 1–32.