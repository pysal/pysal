..

.. testsetup:: *

    import pysal
    import numpy as np

***************
Regionalization
***************

Introduction
============

PySAL offers a number of tools for the construction of regions.  For the
purposes of this section, a "region" is a group of "areas," and there are
generally multiple regions in a particular dataset.  At this time, PySAL
offers the max-p regionalization algorithm and tools for constructing random
regions.

max-p
=====

Most regionalization algorithms require the user to define a priori the number
of regions to be built (e.g. k-means clustering). The max-p algorithm [#]_
determines the number of regions (p) endogenously based on a set of areas, a
matrix of attributes on each area and a floor constraint.  The floor
constraint defines the minimum bound that a variable must reach for each
region; for example, a constraint might be the minimum population each region
must have.  max-p further enforces a contiguity constraint on the areas within
regions.

To illustrate this we will use data on per capita income from the lower 48 US
states over the period 1929-2010. The goal is to form contiguous regions of
states displaying similar levels of income throughout this period:

.. doctest:: 

    >>> import pysal
    >>> import numpy as np
    >>> import random
    >>> f = pysal.open("../pysal/examples/usjoin.csv")
    >>> pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
    >>> pci = pci.transpose()
    >>> pci.shape
    (48, 81)

We also require set of binary contiguity :ref:`weights<weights>` for the Maxp class:

.. doctest:: 

    >>> w = pysal.open("../pysal/examples/states48.gal").read()

Once we have the attribute data and our weights object we can create an instance of Maxp:

.. doctest:: 

    >>> np.random.seed(100)
    >>> random.seed(10)
    >>> r = pysal.Maxp(w, pci, floor = 5, floor_variable = np.ones((48, 1)), initial = 99)

Here we are forming regions with a minimum of 5 states in each region, so we set the floor_variable to a simple unit vector to ensure this floor constraint is satisfied. We also specify the initial number of feasible solutions to 99 - which are then searched over to pick the optimal feasible solution to then commence with the more expensive swapping component of the algorithm. [#]_

The Maxp instance s has a number of attributes regarding the solution. First is the definition of the regions:

.. doctest:: 

    >>> r.regions
    [['44', '34', '3', '25', '1', '4', '47'], ['12', '46', '20', '24', '13'], ['14', '45', '35', '30', '39'], ['6', '27', '17', '29', '5', '43'], ['33', '40', '28', '15', '41', '9', '23', '31', '38'], ['37', '8', '0', '7', '21', '2'], ['32', '19', '11', '10', '22'], ['16', '26', '42', '18', '36']]

which is a list of eight lists of region ids. For example, the first nested list indicates there are seven states in the first region, while the last region has five states.  To determine which states these are we can read in the names from the original csv file:

.. doctest:: 

    >>> f.header
    ['Name', 'STATE_FIPS', '1929', '1930', '1931', '1932', '1933', '1934', '1935', '1936', '1937', '1938', '1939', '1940', '1941', '1942', '1943', '1944', '1945', '1946', '1947', '1948', '1949', '1950', '1951', '1952', '1953', '1954', '1955', '1956', '1957', '1958', '1959', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
    >>> names = f.by_col('Name')
    >>> names = np.array(names)
    >>> print names
    ['Alabama' 'Arizona' 'Arkansas' 'California' 'Colorado' 'Connecticut'
     'Delaware' 'Florida' 'Georgia' 'Idaho' 'Illinois' 'Indiana' 'Iowa'
     'Kansas' 'Kentucky' 'Louisiana' 'Maine' 'Maryland' 'Massachusetts'
     'Michigan' 'Minnesota' 'Mississippi' 'Missouri' 'Montana' 'Nebraska'
     'Nevada' 'New Hampshire' 'New Jersey' 'New Mexico' 'New York'
     'North Carolina' 'North Dakota' 'Ohio' 'Oklahoma' 'Oregon' 'Pennsylvania'
     'Rhode Island' 'South Carolina' 'South Dakota' 'Tennessee' 'Texas' 'Utah'
     'Vermont' 'Virginia' 'Washington' 'West Virginia' 'Wisconsin' 'Wyoming']
    

and then loop over the region definitions to identify the specific states comprising each of the regions:

.. doctest:: 

    >>> for region in r.regions:
    ...     ids = map(int,region)
    ...     print names[ids]
    ...     
    ['Washington' 'Oregon' 'California' 'Nevada' 'Arizona' 'Colorado' 'Wyoming']
    ['Iowa' 'Wisconsin' 'Minnesota' 'Nebraska' 'Kansas']
    ['Kentucky' 'West Virginia' 'Pennsylvania' 'North Carolina' 'Tennessee']
    ['Delaware' 'New Jersey' 'Maryland' 'New York' 'Connecticut' 'Virginia']
    ['Oklahoma' 'Texas' 'New Mexico' 'Louisiana' 'Utah' 'Idaho' 'Montana'
     'North Dakota' 'South Dakota']
    ['South Carolina' 'Georgia' 'Alabama' 'Florida' 'Mississippi' 'Arkansas']
    ['Ohio' 'Michigan' 'Indiana' 'Illinois' 'Missouri']
    ['Maine' 'New Hampshire' 'Vermont' 'Massachusetts' 'Rhode Island']


We can evaluate our solution by developing a pseudo pvalue for the regionalization.
This is done by comparing the within region sum of squares for the solution against
simulated solutions where areas are randomly assigned to regions that maintain
the cardinality of the original solution. This method must be explicitly called once the 
Maxp instance has been created:

.. doctest:: 

    >>> r.inference()
    >>> r.pvalue
    0.01

so we see we have a regionalization that is significantly different than a chance partitioning.


Random Regions
==============

PySAL offers functionality to generate random regions based on user-defined
constraints.  There are three optional parameters to constrain the
regionalization: number of regions, cardinality and contiguity.  The default
case simply takes a list of area IDs and randomly selects the number of
regions and then allocates areas to each region.  The user can also pass a
vector of integers to the cardinality parameter to designate the number of
areas to randomly assign to each region.  The contiguity parameter takes a
:ref:`spatial weights object <weights>` and uses that to ensure that each
region is made up of spatially contiguous areas.  When the contiguity
constraint is enforced, it is possible to arrive at infeasible solutions; the
maxiter parameter can be set to make multiple attempts to find a feasible
solution.  The following examples show some of the possible combinations of
constraints.

.. doctest:: 

    >>> import random
    >>> import numpy as np
    >>> import pysal
    >>> from pysal.region import Random_Region
    >>> nregs = 13
    >>> cards = list(range(2,14)) + [10]
    >>> w = pysal.lat2W(10,10,rook = False)
    >>> ids = w.id_order
    >>>
    >>> # unconstrained
    >>> random.seed(10)
    >>> np.random.seed(10)
    >>> t0 = Random_Region(ids)
    >>> t0.regions[0]
    [19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 1, 60]
    >>> # cardinality and contiguity constrained (num_regions implied)
    >>> random.seed(60)
    >>> np.random.seed(60)
    >>> t1 = pysal.region.Random_Region(ids, num_regions = nregs, cardinality = cards, contiguity = w)
    >>> t1.regions[0]
    [88, 97, 98, 89, 99, 86, 78, 59, 49, 69, 68, 79, 77]
    >>> # cardinality constrained (num_regions implied)
    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t2 = Random_Region(ids, num_regions = nregs, cardinality = cards)
    >>> t2.regions[0]
    [37, 62]
    >>> # number of regions and contiguity constrained
    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t3 = Random_Region(ids, num_regions = nregs, contiguity = w)
    >>> t3.regions[1]
    [71, 72, 70, 93, 51, 91, 85, 74, 63, 73, 61, 62, 82]
    >>> # cardinality and contiguity constrained
    >>> random.seed(60)
    >>> np.random.seed(60)
    >>> t4 = Random_Region(ids, cardinality = cards, contiguity = w)
    >>> t4.regions[0]
    [88, 97, 98, 89, 99, 86, 78, 59, 49, 69, 68, 79, 77]
    >>> # number of regions constrained
    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t5 = Random_Region(ids, num_regions = nregs)
    >>> t5.regions[0]
    [37, 62, 26, 41, 35, 25, 36]
    >>> # cardinality constrained
    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t6 = Random_Region(ids, cardinality = cards)
    >>> t6.regions[0]
    [37, 62]
    >>> # contiguity constrained
    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t7 = Random_Region(ids, contiguity = w)
    >>> t7.regions[0]
    [37, 27, 36, 17]
    >>>

Further Information 
====================

For further details see the :doc:`Regionalization  API <../../library/region/index>`.


.. rubric:: Footnotes

.. [#] Duque, J. C., L. Anselin and S. J. Rey. 2011. "The max-p-regions problem."  *Journal of Regional Science* `DOI: 10.1111/j.1467-9787.2011.00743.x <http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9787.2011.00743.x/abstract>`_
.. [#] Because this is a randomized algorithm, results may vary when replicating this example. To reproduce a regionalization solution, you should first set the random seed generator. See http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html for more information.
