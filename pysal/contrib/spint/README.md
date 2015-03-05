{\rtf1\ansi\ansicpg1252\cocoartf1343\cocoasubrtf140
{\fonttbl\f0\froman\fcharset0 TimesNewRomanPSMT;}
{\colortbl;\red255\green255\blue255;\red98\green151\blue85;\red43\green43\blue43;\red0\green0\blue0;
}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720

\f0\fs26 \cf0 \expnd0\expndtw0\kerning0
**Sp**atial **Int**eraction Modeling Module\
===========================================\
\
The SpInt package seeks to provide a collection of tools for\
calibrating and predicting spatial interaction.\
\
It currently support the calibration of the 'family' of spatial interaction model (Wilson, 1971) which are derived using an entropy maximizing (EM) framework or the equivalent information minimizing (IM) framework. As such, it is able to derive parameters for the following models:\
\
- unconstrained gravity model\
- production-constrained model (origin-constrained)\
- attraction-constrained model (destination-constrained)\
- doubly-constrained model\
\
\
Calibration is carried out using maximum likelihood estimation routine outline in (Fotheringham and O\'92Kelly, 1989; Willimans and Fotheringham, 1984). Optimization is achieved using scipy.optimize.fsolve(). Overall, the package is currently dependent upon bumpy, spicy, and pandas.\
\
Fotheringham, A. S. and O'Kelly, M. E. (1989). Spatial Interaction Models: Formulations and Applications. London: Kluwer Academic Publishers.\
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\fs24 \cf2 \cb3 \kerning1\expnd0\expndtw0 Williams, P. A. and A. S. Fotheringham (1984), The Calibration of Spatial Interaction\uc0\u8232  Models by Maximum Likelihood Estimation with Program SIMODEL, Geographic Monograph\u8232  Series, 7, Department of Geography, Indiana University.\
\pard\pardeftab720

\fs26 \cf0 \cb1 \expnd0\expndtw0\kerning0
\
\pard\pardeftab720\sl480

\fs24 \cf4 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec4 Wilson, A. G. (1971). A family of spatial interaction models, and associated developments. Environment and\
Planning A, 3, 1\'9632.\
}