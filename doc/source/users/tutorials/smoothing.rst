.. testsetup:: * 

        import pysal
        import numpy as np

******************
Spatial Smoothing
******************

.. contents::

Introduction
============

In the spatial analysis of attributes measured for areal units, it is often
necessary to transform an extensive variable, such as number of disease cases
per census tract, into an intensive variable that takes into account the
underlying population at risk.  Raw rates, counts divided by population values,
are a common standardization in the literature, yet these tend to have unequal
reliability due to different population sizes across the spatial units.  This problem becomes
severe for areas with small population values, since the raw rates for those
areas tend to have higher variance.

A variety of spatial smoothing methods have been suggested to address this problem by aggregating
the counts and population values for the areas neighboring an observation and
using these new measurements for its rate computation.  PySAL provides a range
of smoothing techniques that exploit different types of moving windows and
non-parametric weighting schemes as well as the Empirical Bayesian principle.
In addition, PySAL offers several methods for calculating age-standardized
rates, since age standardization is critical in estimating rates of some events
where the probability of an event occurrence is different across different age
groups.

In what follows, we overview the methods for age standardization and spatial smoothing 
and describe their implementations in PySAL. [#]_

Age Standardization in PySAL
============================

Raw rates, counts divided by populations values, are based on an implicit assumption 
that the risk of an event is constant over all age/sex categories in the population. 
For many phenomena, however, the risk is not uniform and often highly correlated with age. 
To take this into account explicitly, the risks for individual age categories can be estimated 
separately and averaged to produce a representative value for an area. 

PySAL supports three approaches to this age standardization: crude, direct, and indirect 
standardization.

Crude Age Standardization
-------------------------

In this approach, the rate for an area is simply the sum of age-specific rates weighted by 
the ratios of each age group in the total population. 

To obtain the rates based on this approach, we first need to create two variables
that correspond to event counts and population values, respectively.

.. doctest:: 

    >>> import numpy as np
    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])
    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

Each set of numbers should include n by h elements where n and h are the number of areal units
and the number of age groups. In the above example there are two regions with 4 age groups.
Age groups are identical across regions. The first four elements in b represent the populations of 4 age 
groups in the first region, and the last four elements the populations of the same age groups in the second 
region. 

To apply the crude age standardization, we need to make the following function call:

.. doctest::  

    >>> from pysal.esda import smoothing as sm
    >>> sm.crude_age_standardization(e, b, 2)
    array([ 0.2375    ,  0.26666667])

In the function call above, the last argument indicates the number of area units.
The outcome in the second line shows that the age-standardized rates for two areas 
are about 0.24 and 0.27, respectively.

Direct Age Standardization
--------------------------

Direct age standardization is a variation of the crude age standardization.
While crude age standardization uses the ratios of each age group in the observed population,
direct age standardization weights age-specific rates by the ratios of each age group in a reference 
population. This reference population, the so-called standard million, is another required 
argument in the PySAL implementation of direct age standardization:

.. doctest:: 

    >>> s = np.array([100, 90, 100, 90, 100, 90, 100, 90])
    >>> rate = sm.direct_age_standardization(e, b, s, 2, alpha=0.05)
    >>> np.array(rate).round(6)
    array([[ 0.23744 ,  0.192049,  0.290485],
           [ 0.266507,  0.217714,  0.323051]])

The outcome of direct age standardization includes a set of standardized rates and their confidence 
intervals. The confidence intervals can vary according to the value for the last argument, alpha.

Indirect Age Standardization
----------------------------

While direct age standardization effectively addresses the variety in the risks across 
age groups, its indirect counterpart is better suited to handle the potential
imprecision of age-specific rates due to the small population size. This method
uses age-specific rates from the standard million instead of the observed
population. It then weights the rates by the ratios of each age group in the
observed population. To compute the age-specific rates from the standard
million, the PySAL implementation of indirect age standardization requires
another argument that contains the counts of the events occurred in the
standard million.

.. doctest:: 

    >>> s_e = np.array([10, 15, 12, 10, 5, 3, 20, 8])
    >>> rate = sm.indirect_age_standardization(e, b, s_e, s, 2, alpha=0.05)
    >>> np.array(rate).round(6)
    array([[ 0.208055,  0.170156,  0.254395],
           [ 0.298892,  0.246631,  0.362228]])

The outcome of indirect age standardization is the same as that of its direct counterpart.

Spatial Smoothing in PySAL
==========================

Mean and Median Based Smoothing
-------------------------------

A simple approach to rate smoothing is to find a local average or median from the rates of each 
observation and its neighbors. The first method adopting this approach is the so-called locally 
weighted averages or disk smoother. In this method a rate for each observation is replaced 
by an average of rates for its neighbors. A :ref:`spatial weights object
<weights>` is used to specify the neighborhood relationships among
observations. To obtain locally weighted averages of the homicide rates in the
counties surrounding St. Louis during 1979-84, we first read the corresponding
data table and extract data values for the homicide counts (the 11th column)
and total population (the 13th column):

.. doctest:: 

    >>> import pysal
    >>> stl = pysal.open('../pysal/examples/stl_hom.csv', 'r')
    >>> e, b = np.array(stl[:,10]), np.array(stl[:,13])

We then read the spatial weights file defining neighborhood relationships among the counties 
and ensure that the :ref:`order <id_order>` of observations in the weights object is the same as that in the data table. 

.. doctest:: 

    >>> w = pysal.open('../pysal/examples/stl.gal', 'r').read()
    >>> if not w.id_order_set: w.id_order = range(1,len(stl) + 1)

Now we calculate locally weighted averages of the homicide rates.

.. doctest:: 

    >>> rate = sm.Disk_Smoother(e, b, w)
    >>> rate.r
    array([  4.56502262e-05,   3.44027685e-05,   3.38280487e-05,
             4.78530468e-05,   3.12278573e-05,   2.22596997e-05,
             ...
             5.29577710e-05,   5.51034691e-05,   4.65160450e-05,
             5.32513363e-05,   3.86199097e-05,   1.92952422e-05])

A variation of locally weighted averages is to use median instead of mean.
In other words, the rate for an observation can be replaced by the median of the rates of its neighbors.
This method is called locally weighted median and can be applied in the following way:

.. doctest:: 

    >>> rate = sm.Spatial_Median_Rate(e, b, w)
    >>> rate.r
    array([  3.96047383e-05,   3.55386859e-05,   3.28308921e-05,
             4.30731238e-05,   3.12453969e-05,   1.97300409e-05,
             ...
             6.10668237e-05,   5.86355507e-05,   3.67396656e-05,
             4.82535850e-05,   5.51831429e-05,   2.99877050e-05])

In this method the procedure to find local medians can be iterated until no further change occurs. 
The resulting local medians are called iteratively resmoothed medians.

.. doctest:: 

    >>> rate = sm.Spatial_Median_Rate(e, b, w, iteration=10)
    >>> rate.r
    array([  3.10194715e-05,   2.98419439e-05,   3.10194715e-05,
             3.10159267e-05,   2.99214885e-05,   2.80530524e-05,
             ...
             3.81364519e-05,   4.72176972e-05,   3.75320135e-05,
             3.76863269e-05,   4.72176972e-05,   3.75320135e-05])

The pure local medians can also be replaced by a weighted median. To obtain weighted medians, 
we need to create an array of weights. For example, we can use the total population of the counties 
as auxiliary weights:

.. doctest:: 

    >>> rate = sm.Spatial_Median_Rate(e, b, w, aw=b)
    >>> rate.r
    array([  5.77412020e-05,   4.46449551e-05,   5.77412020e-05,
             5.77412020e-05,   4.46449551e-05,   3.61363528e-05,
             ...
             5.49703305e-05,   5.86355507e-05,   3.67396656e-05,
             3.67396656e-05,   4.72176972e-05,   2.99877050e-05])

When obtaining locally weighted medians, we can consider only a specific subset of neighbors 
rather than all of them. A representative method following this approach is the headbanging smoother. 
In this method all areal units are represented by their geometric centroids. 
Among the neighbors of each observation, only near collinear points are considered for median search. 
Then, triples of points are selected from the near collinear points, and local medians are computed 
from the triples' rates. [#]_
We apply this headbanging smoother to the rates of the deaths from Sudden Infant Death Syndrome (SIDS) 
for North Carolina counties during 1974-78. We first need to read the source data and extract the event 
counts (the 9th column) and population values (the 9th column). 
In this example the population values correspond to the numbers of live births during 1974-78. 

.. doctest:: 

    >>> sids_db = pysal.open('../pysal/examples/sids2.dbf', 'r')
    >>> e, b = np.array(sids_db[:,9]), np.array(sids_db[:,8])

Now we need to find triples for each observation. To support the search of triples, PySAL 
provides a class called Headbanging_Triples. This class requires an array of point observations, 
a spatial weights object, and the number of triples as its arguments:

.. doctest:: 

    >>> from pysal import knnW
    >>> sids = pysal.open('../pysal/examples/sids2.shp', 'r')
    >>> sids_d = np.array([i.centroid for i in sids])
    >>> sids_w = knnW(sids_d,k=5)
    >>> if not sids_w.id_order_set: sids_w.id_order = sids_w.id_order
    >>> triples = sm.Headbanging_Triples(sids_d,sids_w,k=5)

The second line in the above example shows how to extract centroids of polygons. 
In this example we define 5 neighbors for each observation by using nearest neighbors criteria.
In the last line we define the maximum number of triples to be found as 5.

Now we use the triples to compute the headbanging median rates:

.. doctest:: 

    >>> rate = sm.Headbanging_Median_Rate(e,b,triples)
    >>> rate.r
    array([ 0.00075586,  0.        ,  0.0008285 ,  0.0018315 ,  0.00498891,
            0.00482094,  0.00133156,  0.0018315 ,  0.00413223,  0.00142116,
            ...
            0.00221541,  0.00354767,  0.00259903,  0.00392952,  0.00207125,
            0.00392952,  0.00229253,  0.00392952,  0.00229253,  0.00229253])

As in the locally weighted medians, we can use a set of auxiliary weights and resmooth the medians 
iteratively.

Non-parametric Smoothing
------------------------

Non-parametric smoothing methods compute rates without making any assumptions of distributional 
properties of rate estimates. A representative method in this approach is spatial filtering. 
PySAL provides the most simplistic form of spatial filtering where a user-specified grid is imposed 
on the data set and a moving window withi a fixed or adaptive radius visits each vertex of the grid to 
compute the rate at the vertex. Using the previous SIDS example, we can use Spatial_Filtering class:

.. doctest:: 

    >>> bbox = [sids.bbox[:2], sids.bbox[2:]]
    >>> rate = sm.Spatial_Filtering(bbox, sids_d, e, b, 10, 10, r=1.5)
    >>> rate.r
    array([ 0.00152555,  0.00079271,  0.00161253,  0.00161253,  0.00139513,
            0.00139513,  0.00139513,  0.00139513,  0.00139513,  0.00156348,
            ...
            0.00240216,  0.00237389,  0.00240641,  0.00242211,  0.0024854 ,
            0.00255477,  0.00266573,  0.00288918,  0.0028991 ,  0.00293492])

The first and second arguments of the Spatial_Filtering class are a minimum bounding box containing the 
observations and a set of centroids representing the observations.
Be careful that the bounding box is NOT the bounding box of the centroids.
The fifth and sixth arguments are to specify the numbers of grid cells along x and y axes.
The last argument, r, is to define the radius of the moving window. When this parameter is set,
a fixed radius is applied to all grid vertices. To make the size of moving window variable,
we can specify the minimum number of population in the moving window without specifying r:

.. doctest:: 

    >>> rate = sm.Spatial_Filtering(bbox, sids_d, e, b, 10, 10, pop=10000)
    >>> rate.r
    array([ 0.00157398,  0.00157398,  0.00157398,  0.00157398,  0.00166885,
            0.00166885,  0.00166885,  0.00166885,  0.00166885,  0.00166885,
            ...
            0.00202977,  0.00215322,  0.00207378,  0.00207378,  0.00217173,
            0.00232408,  0.00222717,  0.00245399,  0.00267857,  0.00267857])

The spatial rate smoother is another non-parametric smoothing method that PySAL supports.
This smoother is very similar to the locally weighted averages. In this method, however, 
the weighted sum is applied to event counts and population values separately. 
The resulting weighted sum of event counts is then divided by the counterpart of population 
values. To obtain neighbor information, we need to use a spatial weights matrix as before. 

.. doctest:: 

    >>> rate = sm.Spatial_Rate(e, b, sids_w)
    >>> rate.r
    array([ 0.00114976,  0.00104622,  0.00110001,  0.00153257,  0.00399662,
            0.00361428,  0.00146807,  0.00238521,  0.00288871,  0.00145228,
            ...
            0.00240839,  0.00376101,  0.00244941,  0.0028813 ,  0.00240839,
            0.00261705,  0.00226554,  0.0031575 ,  0.00254536,  0.0029003 ])

Another variation of spatial rate smoother is kernel smoother. PySAL supports kernel smoothing 
by using a kernel spatial weights instance in place of a general spatial weights object.

.. doctest:: 

    >>> from pysal import Kernel
    >>> kw = Kernel(sids_d)
    >>> if not kw.id_order_set: kw.id_order = range(0,len(sids_d))
    >>> rate = sm.Kernel_Smoother(e, b, kw)
    >>> rate.r
    array([ 0.0009831 ,  0.00104298,  0.00137113,  0.00166406,  0.00556741,
            0.00442273,  0.00158202,  0.00243354,  0.00282158,  0.00099243,
            ...
            0.00221017,  0.00328485,  0.00257988,  0.00370461,  0.0020566 ,
            0.00378135,  0.00240358,  0.00432019,  0.00227857,  0.00251648])

Age-adjusted rate smoother is another non-parametric smoother that PySAL provides.
This smoother applies direct age standardization while computing spatial rates. 
To illustrate the age-adjusted rate smoother, we create a new set of event counts and population values 
as well as a new kernel weights object.

.. doctest:: 

    >>> e = np.array([10, 8, 1, 4, 3, 5, 4, 3, 2, 1, 5, 3])
    >>> b = np.array([100, 90, 15, 30, 25, 20, 30, 20, 80, 80, 90, 60])
    >>> s = np.array([98, 88, 15, 29, 20, 23, 33, 25, 76, 80, 89, 66])
    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw=Kernel(points)
    >>> if not kw.id_order_set: kw.id_order = range(0,len(points))

In the above example we created 6 observations each of which has two age groups. To apply age-adjusted 
rate smoothing, we use the Age_Adjusted_Smoother class as follows:

.. doctest:: 

    >>> rate = sm.Age_Adjusted_Smoother(e, b, kw, s)
    >>> rate.r
    array([ 0.10519625,  0.08494318,  0.06440072,  0.06898604,  0.06952076,
            0.05020968])

Empirical Bayes Smoothers
-------------------------

The last group of smoothing methods that PySAL supports is based upon the Bayesian principle. These methods adjust 
a raw rate by taking into account information in the other raw rates. 
As a reference PySAL provides a method for a-spatial Empirical Bayes smoothing:

.. doctest:: 

    >>> e, b = sm.sum_by_n(e, np.ones(12), 6), sm.sum_by_n(b, np.ones(12), 6)
    >>> rate = sm.Empirical_Bayes(e, b)
    >>> rate.r
    array([ 0.09080775,  0.09252352,  0.12332267,  0.10753624,  0.03301368,
            0.05934766])

In the first line of the above example we aggregate the event counts and population values by observation.
Next we applied the Empirical_Bayes class to the aggregated counts and population values.

A spatial Empirical Bayes smoother is also implemented in PySAL. This method requires an additional 
argument, i.e., a spatial weights object. We continue to reuse the kernel spatial weights object we built before.

.. doctest:: 

    >>> rate = sm.Spatial_Empirical_Bayes(e, b, kw) 
    >>> rate.r
    array([ 0.10105263,  0.10165261,  0.16104362,  0.11642038,  0.0226908 ,
            0.05270639])

Excess Risk
-----------

Besides a variety of spatial smoothing methods, PySAL provides a class for estimating excess risk from event counts 
and population values. Excess risks are the ratios of observed event counts over expected event counts.
An example for the class usage is as follows:

.. doctest:: 

    >>> risk = sm.Excess_Risk(e, b)
    >>> risk.r
    array([ 1.23737916,  1.45124717,  2.32199546,  1.82857143,  0.24489796,
            0.69659864])

Further Information 
====================

For further details see the :doc:`Smoothing API <../../library/esda/smoothing>`.

.. rubric:: Footnotes

.. [#] Although this tutorial provides an introduction to the PySAL implementations for spatial smoothing, it is not exhaustive. Complete documentation for the implementations can be found by accessing the help from within a Python interpreter. 
.. [#] For the details of triple selection and headbanging smoothing please
       refer to Anselin, L., Lozano, N., and Koschinsky, J. (2006). "`Rate
       Transformations and Smoothing
       <http://geodacenter.asu.edu/pdf/smoothing_06.pdf>`_". GeoDa Center
       Research Report.

