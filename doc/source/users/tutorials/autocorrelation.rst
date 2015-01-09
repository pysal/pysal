.. testsetup:: * 

        import pysal
        import numpy as np
        np.random.seed(12345)

#######################
Spatial Autocorrelation
#######################

.. contents::

Introduction
============

Spatial autocorrelation pertains to the non-random pattern of attribute values
over a set of spatial units. This can take two general forms: positive
autocorrelation which reflects value similarity in space, and negative
autocorrelation or value dissimilarity in space. In either case the
autocorrelation arises when the observed spatial pattern is different from what would
be expected under a random process operating in space.

Spatial autocorrelation can be analyzed from two different perspectives. Global
autocorrelation analysis involves the study of the entire map pattern and
generally asks the question as to whether the pattern displays clustering or
not. Local autocorrelation, on the other hand, shifts the focus to explore
within the global pattern to identify clusters or so called hot spots that may be
either driving the overall clustering pattern, or that reflect heterogeneities
that depart from global pattern.

In what follows, we first highlight the global spatial autocorrelation classes
in PySAL. This is followed by an illustration of the analysis of local spatial
autocorrelation.

Global Autocorrelation
======================

PySAL implements five different tests for global spatial autocorrelation:
the Gamma index of spatial autocorrelation, join count statistics, 
Moran's I, Geary's C, and Getis and Ord's G.

Gamma Index of Spatial Autocorrelation
--------------------------------------

The Gamma Index of spatial autocorrelation consists of the application of the principle
behind a general cross-product statistic to measuring spatial autocorrelation. [#]_
The idea is to assess whether two similarity matrices for n objects, i.e., n by n
matrices A and B measure the same type of similarity. This is reflected in a so-called
Gamma Index :math:`\Gamma = \sum_i \sum_j a_{ij}.b_{ij}`. In other words, the statistic
consists of the sum over all cross-products of matching elements (i,j) in the two 
matrices.

The application of this principle to spatial autocorrelation consists of turning
the first similarity matrix into a measure of attribute similarity and the second
matrix into a measure of locational similarity. Naturally, the second matrix is the
a spatial :doc:`weight <weights>` matrix. The first matrix can be any reasonable measure of attribute
similarity or dissimilarity, such as a cross-product, squared difference or absolute
difference.

Formally, then, the Gamma index is:

.. math::

        \Gamma = \sum_i \sum_j a_{ij}.w_{ij}
        
where the :math:`w_{ij}` are the elements of the weights matrix and 
:math:`a_{ij}` are corresponding measures of attribute similarity.

Inference for this statistic is based on a permutation approach in which the values
are shuffled around among the locations and the statistic is recomputed each
time. This creates a reference distribution for the statistic under the null
hypothesis of spatial randomness. The observed statistic is then compared to this
reference distribution and a pseudo-significance computed as

.. math::

       p = (m + 1) / (n + 1)
       
where m is the number of values from the reference distribution that are equal to
or greater than the observed join count and n is the number of permutations.

The Gamma test is a two-sided test in the sense that both extremely high values (e.g.,
larger than any value in the reference distribution) and extremely low values
(e.g., smaller than any value in the reference distribution) can be considered
to be significant. Depending on how the measure of attribute similarity is defined,
a high value will indicate positive or negative spatial autocorrelation, and vice
versa. For example, for a cross-product measure of attribute similarity, high values
indicate positive spatial autocorrelation and low values negative spatial autocorrelation.
For a squared difference measure, it is the reverse. This is similar to the 
interpretation of the :ref:`moran` statistic and :ref:`geary` statistic respectively.

Many spatial autocorrelation test statistics can be shown to be special cases of the
Gamma index. In most instances, the Gamma index is an unstandardized version of the
commonly used statistics. As such, the Gamma index is scale dependent, since no
normalization is carried out (such as deviations from the mean or rescaling by the
variance). Also, since the sum is over all the elements, the value of a Gamma
statistic will grow with the sample size, everything else being the same.

PySAL implements four forms of the Gamma index. Three of these are pre-specified
and one allows the user to pass any function that computes a measure of attribute
similarity. This function should take three parameters: the vector of observations,
an index i and an index j.

We will illustrate the Gamma index using the same small artificial example
as we use for the  :ref:`moran1`  in order to illustrate the similarities
and differences between them. The data consist of a regular 4 by 4 lattice with
values of 0 in the top half and values of 1 in the bottom half. We start with the usual 
imports, and set the random seed to 12345 in order
to be able to replicate the results of the permutation approach.


        >>> import pysal
        >>> import numpy as np
        >>> np.random.seed(12345)
        
We create the binary weights matrix for the 4 x 4 lattice and generate the
observation vector y:

.. doctest::

        >>> w=pysal.lat2W(4,4)
        >>> y=np.ones(16)
        >>> y[0:8]=0 

The Gamma index function has five arguments, three of which are optional.
The first two arguments are the vector of observations (y) and the spatial
weights object (w). Next are ``operation``, the measure of attribute similarity,
the default of which is ``operation = 'c'`` for cross-product similarity, 
:math:`a_{ij} = y_i.y_j`. The other two built-in options are ``operation = 's'`` for
squared difference, :math:`a_{ij} = (y_i - y_j)^2` and ``operation = 'a'`` for
absolute difference, :math:`a_{ij} = | y_i - y_j |`. The fourth option is to
pass an arbitrary attribute similarity function, as in ``operation = func``, where ``func``
is a function with three arguments, ``def func(y,i,j)`` with y as the vector
of observations, and i and j as indices. This function should return a single
value for attribute similarity.

The fourth argument allows the observed values to be standardized before the
calculation of the Gamma index. To some extent, this addresses the scale dependence
of the index, but not its dependence on the number of observations. The default
is no standardization, ``standardize = 'no'``. To force standardization,
set ``standardize = 'yes'`` or ``'y'``. The final argument is the number of
permutations, ``permutations`` with the default set to 999.

As a first illustration, we invoke the Gamma index using all the default
values, i.e. cross-product similarity, no standardization, and permutations
set to 999. The interesting statistics are the magnitude of the Gamma index ``g``,
the standardized Gamma index using the mean and standard deviation from the
reference distribution, ``g_z`` and the pseudo-p value obtained from the
permutation, ``g_sim_p``. In addition, the minimum (``min_g``), maximum (``max_g``)
and mean (``mean_g``) of the reference distribution are available as well.

.. doctest::

        >>> g = pysal.Gamma(y,w)
        >>> g.g
        20.0
        >>> "%.3f"%g.g_z
        '3.188'
        >>> g.p_sim_g
        0.0030000000000000001
        >>> g.min_g
        0.0
        >>> g.max_g
        20.0
        >>> g.mean_g
        11.093093093093094

Note that the value for Gamma is exactly twice the BB statistic obtained in the
example below, since the attribute similarity criterion is identical, but Gamma is
not divided by 2.0. The observed value is very extreme, with only two replications
from the permutation equalling the value of 20.0. This indicates significant
positive spatial autocorrelation.

As a second illustration, we use the squared difference criterion, which
corresponds to the BW Join Count statistic. We reset the random seed to
keep comparability of the results.

.. doctest::

        >>> np.random.seed(12345)
        >>> g1 = pysal.Gamma(y,w,operation='s')
        >>> g1.g
        8.0
        >>> "%.3f"%g1.g_z
        '-3.706'
        >>> g1.p_sim_g
        0.001
        >>> g1.min_g
        14.0
        >>> g1.max_g
        48.0
        >>> g1.mean_g
        25.623623623623622

The Gamma index value of 8.0 is exactly twice the value of the BW statistic for
this example. However, since the Gamma index is used for a two-sided test, this
value is highly significant, and with a negative z-value, this suggests positive 
spatial autocorrelation (similar
to Geary's C). In other words, this result is consistent with the finding for the
Gamma index that used cross-product similarity.

As a third example, we use the absolute difference for attribute similarity.
The results are identical to those for squared difference since these two
criteria are equivalent for 0-1 values.

.. doctest::

        >>> np.random.seed(12345)
        >>> g2 = pysal.Gamma(y,w,operation='a')
        >>> g2.g
        8.0
        >>> "%.3f"%g2.g_z
        '-3.706'
        >>> g2.p_sim_g
        0.001
        >>> g2.min_g
        14.0
        >>> g2.max_g
        48.0
        >>> g2.mean_g
        25.623623623623622
    
We next illustrate the effect of standardization, using the default operation.
As shown, the value of the statistic is quite different from the unstandardized
form, but the inference is equivalent.

.. doctest::

        >>> np.random.seed(12345)
        >>> g3 = pysal.Gamma(y,w,standardize='y')
        >>> g3.g
        32.0
        >>> "%.3f"%g3.g_z
        '3.706'
        >>> g3.p_sim_g
        0.001
        >>> g3.min_g
        -48.0
        >>> g3.max_g
        20.0
        >>> "%.3f"%g3.mean_g
        '-3.247'

Note that all the tests shown here have used the weights matrix in binary form.
However, since the Gamma index is perfectly general,
any standardization can be applied to the weights.

Finally, we illustrate the use of an arbitrary attribute similarity function.
In order to compare to the results above, we will define a function that 
produces a cross product similarity measure. We will then pass this function
to the ``operation`` argument of the Gamma index.

.. doctest::

        >>> np.random.seed(12345)
        >>> def func(z,i,j):
        ...     q = z[i]*z[j]
        ...     return q
        ... 
        >>> g4 = pysal.Gamma(y,w,operation=func)
        >>> g4.g
        20.0
        >>> "%.3f"%g4.g_z
        '3.188'
        >>> g4.p_sim_g
        0.0030000000000000001

As expected, the results are identical to those obtained with the default
operation. 


.. _moran1:

Join Count Statistics
---------------------

The join count statistics measure global spatial autocorrelation for binary data, i.e.,
with observations coded as 1 or B (for Black) and 0 or W (for White). They follow the
very simple principle of counting joins, i.e., the arrangement of values between
pairs of observations where the pairs correspond to neighbors. The three resulting
join count statistics are BB, WW and BW. Both BB and WW are measures of positive
spatial autocorrelation, whereas BW is an indicator of negative spatial autocorrelation.

To implement the join count statistics, we need the spatial weights matrix in 
binary (not row-standardized) form. With :math:`y` as the vector of observations
and the spatial :doc:`weight <weights>` as :math:`w_{i,j}`, the three statistics can be expressed as:

.. math::

       BB = (1/2) \sum_{i}\sum_{j} y_i y_j w_{ij}
     
.. math::      
  
       WW = (1/2) \sum_{i}\sum_{j} (1 - y_i)(1 - y_j) w_{ij}

.. math::

       BW = (1/2) \sum_{i}\sum_{j} (y_i - y_j)^2 w_{ij}
     
By convention, the join counts are divided by 2 to avoid double counting. Also, since
the three joins exhaust all the possibilities, they sum to one half (because of the
division by 2) of the total sum of weights :math:`J = (1/2)S_0 = (1/2)\sum_{i}\sum_{j} w_{ij}`.

Inference for the join count statistics can be based on either an analytical approach
or a computational approach. The analytical approach starts from the binomial distribution
and derives the moments of the statistics under the assumption of free sampling
and non-free sampling. The resulting mean and variance are used to construct a
standardized z-variable which can be approximated as a standard normal variate. [#]_
However, the approximation is often poor in practice. We therefore only implement the
computational approach.

Computational inference is based on a permutation approach in which the values of y
are randomly reshuffled many times to obtain a reference distribution of the statistics
under the null hypothesis of spatial randomness. The observed join count is then
compared to this reference distribution and a pseudo-significance computed as

.. math::

       p = (m + 1) / (n + 1)
       
where m is the number of values from the reference distribution that are equal to
or greater than the observed join count and n is the number of permutations. Note
that the join counts are a one sided-test. If the counts are extremely smaller
than the reference distribution, this is not an indication of significance. For
example, if the BW counts are extremely small, this is not an indication of
*negative* BW autocorrelation, but instead points to the presence of BB or WW
autocorrelation.

We will illustrate the join count statistics with a simple artificial example
of a 4 by 4 square lattice with values of 0 in the top half and values of 1 in
the bottom half.

We start with the usual imports, and set the random seed to 12345 in order
to be able to replicate the results of the permutation approach.

.. doctest::

        >>> import pysal
        >>> import numpy as np
        >>> np.random.seed(12345)
        
We create the binary weights matrix for the 4 x 4 lattice and generate the
observation vector y:

.. doctest::

        >>> w=pysal.lat2W(4,4)
        >>> y=np.ones(16)
        >>> y[0:8]=0 

We obtain an instance of the joint count statistics BB, BW and WW as (J is
half the sum of all the weights and should equal the sum of BB, WW and BW):

.. doctest::

        >>> jc=pysal.Join_Counts(y,w)
        >>> jc.bb
        10.0
        >>> jc.bw
        4.0
        >>> jc.ww
        10.0
        >>> jc.J
        24.0

The number of permutations is set to 999 by default. For other values, this parameter
needs to be passed explicitly, as in:


        >>> jc=pysal.Join_Counts(y,w,permutations=99)
        
The results in our simple example show that the BB counts are 10. There are
in fact 3 horizontal joins in each of the bottom rows of the lattice as well as
4 vertical joins, which makes for bb = 3 + 3 + 4 = 10. The BW joins are 4, matching the
separation between the bottom and top part.

The permutation results give a pseudo-p value for BB of 0.003, suggesting highly
significant positive spatial autocorrelation. The average BB count
for the sample of 999 replications is 5.5, quite a bit lower than the count of 10
we obtain. Only two instances of the replicated samples yield a value equal to 10,
none is greater (the randomly permuted samples yield bb values between 0 and 10).

.. doctest::

        >>> len(jc.sim_bb)
        999
        >>> jc.p_sim_bb
        0.0030000000000000001
        >>> np.mean(jc.sim_bb)
        5.5465465465465469
        >>> np.max(jc.sim_bb)
        10.0
        >>> np.min(jc.sim_bb)
        0.0

The results for BW (negative spatial autocorrelation) show a probability of 1.0
under the null hypothesis. This means that all the values of BW from the randomly
permuted data sets were larger than the observed value of 4. In fact the range
of these values is between 7 and 24. In other words, this again strongly points
towards the presence of positive spatial autocorrelation. The observed number of
BB and WW joins (10 each) is so high that there are hardly any BW joins (4).

.. doctest::

        >>> len(jc.sim_bw)
        999
        >>> jc.p_sim_bw
        1.0
        >>> np.mean(jc.sim_bw)
        12.811811811811811
        >>> np.max(jc.sim_bw)
        24.0
        >>> np.min(jc.sim_bw)
        7.0

.. _moran:

Moran's I
---------

Moran's I measures the global spatial autocorrelation in an attribute :math:`y` measured over :math:`n` spatial units and is given as:

.. math::

        I = n/S_0  \sum_{i}\sum_j z_i w_{i,j} z_j / \sum_i z_i z_i

where :math:`w_{i,j}` is a spatial :doc:`weight <weights>`, :math:`z_i = y_i - \bar{y}`, and :math:`S_0=\sum_i\sum_j w_{i,j}`.  We illustrate the use of Moran's I with a case study of homicide rates for a group of 78 counties surrounding St. Louis over the period 1988-93. [#]_
We start with the usual imports:


        >>> import pysal
        >>> import numpy as np

Next, we read in the homicide rates:

.. doctest::

        >>> f = pysal.open(pysal.examples.get_path("stl_hom.txt"))
        >>> y = np.array(f.by_col['HR8893'])

To calculate Moran's I we first need to read in a GAL file for a rook weights
matrix and create an instance of W:

.. doctest::

        >>> w = pysal.open(pysal.examples.get_path("stl.gal")).read()
        
The instance of Moran's I can then be obtained with:

.. doctest::

        >>> mi = pysal.Moran(y, w, two_tailed=False)
        >>> "%.3f"%mi.I
        '0.244'
        >>> mi.EI
        -0.012987012987012988
        >>> "%.5f"%mi.p_norm
        '0.00014'

From these results, we see that the observed value for I is significantly above its expected value, under the assumption of normality for the homicide rates. 

If we peek inside the mi object to learn more:

        >>> help(mi)

which generates::

        Help on instance of Moran in module pysal.esda.moran:

        class Moran
         |  Moran's I Global Autocorrelation Statistic
         |  
         |  Parameters
         |  ----------
         |  
         |  y               : array
         |                    variable measured across n spatial units
         |  w               : W
         |                    spatial weights instance
         |  permutations    : int
         |                    number of random permutations for calculation of pseudo-p_values
         |  
         |  
         |  Attributes
         |  ----------
         |  y            : array
         |                 original variable
         |  w            : W
         |                 original w object
         |  permutations : int
         |                 number of permutations
         |  I            : float
         |                 value of Moran's I
         |  EI           : float
         |                 expected value under normality assumption
         |  VI_norm      : float
         |                 variance of I under normality assumption
         |  seI_norm     : float
         |                 standard deviation of I under normality assumption
         |  z_norm       : float
         |                 z-value of I under normality assumption
         |  p_norm       : float
         |                 p-value of I under normality assumption (one-sided)
         |                 for two-sided tests, this value should be multiplied by 2
         |  VI_rand      : float
         |                 variance of I under randomization assumption
         |  seI_rand     : float
         |                 standard deviation of I under randomization assumption
         |  z_rand       : float
         |                 z-value of I under randomization assumption
         |  p_rand       : float
         |                 p-value of I under randomization assumption (1-tailed)
         |  sim          : array (if permutations>0)
        
we see that we can base the inference not only on the normality assumption, but also on random permutations of the values on the spatial units to generate a reference distribution for I under the null:

.. doctest::

        >>> np.random.seed(10)
        >>> mir = pysal.Moran(y, w, permutations = 9999)

The pseudo p value based on these permutations is: 

.. doctest::

        >>> print mir.p_sim
        0.0022

in other words there were 14 permutations that generated values for I that
were as extreme as the original value, so the p value becomes (14+1)/(9999+1). [#]_
Alternatively, we could use the realized values for I from the permutations and
compare the original I using a z-transformation to get:

.. doctest::

        >>> print mir.EI_sim
        -0.0118217511619
        >>> print mir.z_sim
        4.55451777821
        >>> print mir.p_z_sim
	2.62529422013e-06

When the variable of interest (:math:`y`) is rates based on populations with different sizes, 
the Moran's I value for :math:`y` needs to be adjusted to account for the differences among populations. [#]_
To apply this adjustment, we can create an instance of the Moran_Rate class rather than the Moran class.
For example, let's assume that we want to estimate the Moran's I for the rates of newborn infants who died of 
Sudden Infant Death Syndrome (SIDS). We start this estimation by reading in the total number of newborn infants (BIR79)
and the total number of newborn infants who died of SIDS (SID79):

.. doctest::

        >>> f = pysal.open(pysal.examples.get_path("sids2.dbf"))
        >>> b = np.array(f.by_col('BIR79'))
        >>> e = np.array(f.by_col('SID79'))

Next, we create an instance of W:

.. doctest::

        >>> w = pysal.open(pysal.examples.get_path("sids2.gal")).read()

Now, we create an instance of Moran_Rate:

.. doctest::

        >>> mi = pysal.esda.moran.Moran_Rate(e, b, w, two_tailed=False)
        >>> "%6.4f" % mi.I
        '0.1662'
        >>> "%6.4f" % mi.EI
        '-0.0101'
        >>> "%6.4f" % mi.p_norm
        '0.0042'

From these results, we see that the observed value for I is significantly higher than its expected value,
after the adjustment for the differences in population.

.. _geary:

Geary's C
---------
The fourth statistic for global spatial autocorrelation implemented in PySAL is Geary's C:

.. math::

        C=\frac{(n-1)}{2S_0} \sum_i\sum_j w_{i,j} (y_i-y_j)^2 / \sum_i z_i^2

with all the terms defined as above. Applying this to the St. Louis data:

.. doctest::

    >>> np.random.seed(12345)
    >>> f = pysal.open(pysal.examples.get_path("stl_hom.txt"))
    >>> y = np.array(f.by_col['HR8893'])
    >>> w = pysal.open(pysal.examples.get_path("stl.gal")).read()
    >>> gc = pysal.Geary(y, w)
    >>> "%.3f"%gc.C
    '0.597'
    >>> gc.EC
    1.0
    >>> "%.3f"%gc.z_norm
    '-5.449'

we see that the statistic :math:`C` is significantly lower than its expected
value :math:`EC`. Although the sign of the standardized statistic is negative (in contrast to what held for :math:`I`, the interpretation is the same, namely evidence of strong positive spatial autocorrelation in the homicide rates.

Similar to what we saw for Moran's I, we can base inference on Geary's :math:`C` using
random spatial permutations, which are actually run as a default with the
number of permutations=999 (this is why we set the seed of the random number
generator to 12345 to replicate the result):

.. doctest::

    >>> gc.p_sim
    0.001

which indicates that none of the C values from the permuted samples was as extreme as our observed value.

Getis and Ord's G
-----------------
The last statistic for global spatial autcorrelation implemented in PySAL is Getis and Ord's G:

.. math::

        G(d)=\frac{\sum_i\sum_j w_{i,j}(d) y_i y_j}{\sum_i\sum_j y_i y_j}

where :math:`d` is a threshold distance used to define a spatial :doc:`weight <weights>`.
Only :class:`pysal.weights.Distance.DistanceBand` weights objects are applicable to Getis and Ord's G.
Applying this to the St. Louis data:

.. doctest::

        >>> dist_w = pysal.threshold_binaryW_from_shapefile('../pysal/examples/stl_hom.shp',0.6)
        >>> dist_w.transform = "B"
        >>> from pysal.esda.getisord import G
        >>> g = G(y, dist_w)
        >>> print g.G
        0.103483215873
        >>> print g.EG
        0.0752580752581
        >>> print g.z_norm
        3.28090342959
        >>> print g.p_norm
        0.000517375830488

Although we switched the contiguity-based weights object into another distance-based one,
we see that the statistic :math:`G` is significantly higher than its expected
value :math:`EG` under the assumption of normality for the homicide rates.

Similar to what we saw for Moran's I and Geary's C, we can base inference on Getis and Ord's G using random spatial permutations:

.. doctest::

    >>> np.random.seed(12345)
    >>> g = G(y, dist_w, permutations=9999)
    >>> print g.p_z_sim
    0.000564384586974
    >>> print g.p_sim
    0.0065

with the first p-value based on a z-transform of the observed G relative to the
distribution of values obtained in the permutations, and the second based on
the cumulative probability of the observed value in the empirical distribution.

Local Autocorrelation
=====================

.. _lisa:

To measure local autocorrelation quantitatively, 
PySAL implements Local Indicators of Spatial Association (LISAs) for Moran's I and Getis and Ord's G.

Local Moran's I
----------------

PySAL implements local Moran's I as follows:

.. math::

        I_i =  \sum_j z_i w_{i,j} z_j / \sum_i z_i z_i

which results in :math:`n` values of local spatial autocorrelation, 1 for each spatial unit. Continuing on with the St. Louis example, the LISA statistics are obtained with:

.. doctest::

       >>> f = pysal.open(pysal.examples.get_path("stl_hom.txt"))
       >>> y = np.array(f.by_col['HR8893'])
       >>> w = pysal.open(pysal.examples.get_path("stl.gal")).read()
       >>> np.random.seed(12345)
       >>> lm = pysal.Moran_Local(y,w)
       >>> lm.n
       78
       >>> len(lm.Is)
       78
        
thus we see 78 LISAs are stored in the vector lm.Is. Inference about these values is obtained through conditional randomization [#]_ which leads to pseudo p-values for each LISA:

.. doctest::

    >>> lm.p_sim
    array([ 0.176,  0.073,  0.405,  0.267,  0.332,  0.057,  0.296,  0.242,
            0.055,  0.062,  0.273,  0.488,  0.44 ,  0.354,  0.415,  0.478,
            0.473,  0.374,  0.415,  0.21 ,  0.161,  0.025,  0.338,  0.375,
            0.285,  0.374,  0.208,  0.3  ,  0.373,  0.411,  0.478,  0.414,
            0.009,  0.429,  0.269,  0.015,  0.005,  0.002,  0.077,  0.001,
            0.088,  0.459,  0.435,  0.365,  0.231,  0.017,  0.033,  0.04 ,
            0.068,  0.101,  0.284,  0.309,  0.113,  0.457,  0.045,  0.269,
            0.118,  0.346,  0.328,  0.379,  0.342,  0.39 ,  0.376,  0.467,
            0.357,  0.241,  0.26 ,  0.401,  0.185,  0.172,  0.248,  0.4  ,
            0.482,  0.159,  0.373,  0.455,  0.083,  0.128])

To identify the significant [#]_ LISA values we can use numpy indexing:

.. doctest::

        >>> sig = lm.p_sim<0.05
        >>> lm.p_sim[sig]
        array([ 0.025,  0.009,  0.015,  0.005,  0.002,  0.001,  0.017,  0.033,
                0.04 ,  0.045])

and then use this indexing on the q attribute to find out which quadrant of the Moran scatter plot each of the significant values is contained in:

.. doctest::

        >>> lm.q[sig]
        array([4, 1, 3, 1, 3, 1, 1, 3, 3, 3])

As in the case of global Moran's I, when the variable of interest is rates based on populations with different sizes,
we need to account for the differences among population to estimate local Moran's Is. 
Continuing on with the SIDS example above, the adjusted local Moran's Is are obtained with:

.. doctest::

	>>> f = pysal.open(pysal.examples.get_path("sids2.dbf"))
	>>> b = np.array(f.by_col('BIR79'))
	>>> e = np.array(f.by_col('SID79'))
	>>> w = pysal.open(pysal.examples.get_path("sids2.gal")).read()
    >>> np.random.seed(12345)
    >>> lm = pysal.esda.moran.Moran_Local_Rate(e, b, w)
    >>> lm.Is[:10]
    array([-0.13452366, -1.21133985,  0.05019761,  0.06127125, -0.12627466,
            0.23497679,  0.26345855, -0.00951288, -0.01517879, -0.34513514])

As demonstrated above, significant Moran's Is can be identified by using numpy indexing:

.. doctest::

        >>> sig = lm.p_sim<0.05
        >>> lm.p_sim[sig]
        array([ 0.021,  0.04 ,  0.047,  0.015,  0.001,  0.017,  0.032,  0.031,
                0.019,  0.014,  0.004,  0.048,  0.003])


Local G and G*
--------------

Getis and Ord's G can be localized in two forms: :math:`G_i` and :math:`G^*_i`.

.. math::

        G_i(d) = \frac{\sum_j w_{i,j}(d) y_j - W_i\bar{y}(i)}{s(i)\{[(n-1)S_{1i} - W^2_i]/(n-2)\}^(1/2)}, j \neq i

.. math::

        G^*_i(d) = \frac{\sum_j w_{i,j}(d) y_j - W^*_i\bar{y}}{s\{[(nS^*_{1i}) - (W^*_i)^2]/(n-1)\}^(1/2)}, j = i

where we have :math:`W_i = \sum_{j \neq i} w_{i,j}(d)`, :math:`\bar{y}(i) = \frac{\sum_j y_j}{(n-1)}`, :math:`s^2(i) = \frac{\sum_j y^2_j}{(n-1)} - [\bar{y}(i)]^2`, :math:`W^*_i = W_i + w{i,i}`, :math:`S_{1i} = \sum_j w^2_{i,j} (j \neq i)`, and :math:`S^*_{1i} = \sum_j w^2_{i,j} (\forall j)`, :math:`\bar{y}` and :math:`s^2` denote the usual sample mean and variance of :math:`y`.

Continuing on with the St. Louis example, the :math:`G_i` and :math:`G^*_i` statistics are obtained with:

.. doctest::

        >>> from pysal.esda.getisord import G_Local
        >>> np.random.seed(12345)
        >>> lg = G_Local(y, dist_w)
        >>> lg.n
        78
        >>> len(lg.Gs)
        78
        >>> lgstar = G_Local(y, dist_w, star=True)
        >>> lgstar.n
        78
        >>> len(lgstar.Gs)
        78
        
thus we see 78 :math:`G_i` and :math:`G^*_i` are stored in the vector lg.Gs and lgstar.Gs, respectively. Inference about these values is obtained through conditional randomization as in the case of local Moran's I:

.. doctest::

    >>> lg.p_sim
    array([ 0.301,  0.037,  0.457,  0.011,  0.062,  0.006,  0.094,  0.163,
            0.075,  0.078,  0.419,  0.286,  0.138,  0.443,  0.36 ,  0.484,
            0.434,  0.251,  0.415,  0.21 ,  0.177,  0.001,  0.304,  0.042,
            0.285,  0.394,  0.208,  0.089,  0.244,  0.493,  0.478,  0.433,
            0.006,  0.429,  0.037,  0.105,  0.005,  0.216,  0.23 ,  0.023,
            0.105,  0.343,  0.395,  0.305,  0.264,  0.017,  0.033,  0.01 ,
            0.001,  0.115,  0.034,  0.225,  0.043,  0.312,  0.045,  0.092,
            0.118,  0.428,  0.258,  0.379,  0.408,  0.39 ,  0.475,  0.493,
            0.357,  0.298,  0.232,  0.454,  0.149,  0.161,  0.226,  0.4  ,
            0.482,  0.159,  0.27 ,  0.423,  0.083,  0.128])


To identify the significant :math:`G_i` values we can use numpy indexing:

.. doctest::


    >>> sig = lg.p_sim<0.05
    >>> lg.p_sim[sig]
    array([ 0.037,  0.011,  0.006,  0.001,  0.042,  0.006,  0.037,  0.005,
            0.023,  0.017,  0.033,  0.01 ,  0.001,  0.034,  0.043,  0.045])

Further Information 
====================

For further details see the :doc:`ESDA  API <../../library/esda/index>`.




.. rubric:: Footnotes


.. [#] Hubert, L., R. Golledge and C.M. Costanzo (1981). Generalized procedures for evaluating spatial autocorrelation. Geographical Analysis 13, 224-233.
.. [#] Technical details and derivations can be found in A.D. Cliff and J.K. Ord (1981). Spatial Processes, Models and Applications. London, Pion, pp. 34-41.
.. [#] Messner, S.,  L. Anselin, D. Hawkins, G. Deane, S. Tolnay, R. Baller (2000). An Atlas of the Spatial Patterning of County-Level Homicide, 1960-1990. Pittsburgh, PA, National Consortium on Violence Research (NCOVR)
.. [#] Because the permutations are random, results from those presented here may vary if you replicate this example.
.. [#] Assuncao, R. E. and Reis, E. A. 1999. A new proposal to adjust Moran's I for population density. Statistics in Medicine. 18, 2147-2162.
.. [#] The n-1 spatial units other than i are used to generate the empirical distribution of the LISA statistics for each i.
.. [#] Caution is required in interpreting the significance of the LISA statistics due to difficulties with multiple comparisons and a lack of independence across the individual tests. For further discussion see Anselin, L. (1995). "Local indicators of spatial association – LISA". Geographical Analysis, 27, 93-115.

