"""
Methods for identifying space-time interaction in spatio-temporal event
data.
"""
__author__  = "Nicholas Malizia <nmalizia@asu.edu> "

import pysal
from pysal.common import *
import pysal.weights.Distance as Distance
from pysal import cg
import util

__all__=['SpaceTimeEvents','knox','mantel','jacquez','modified_knox']


class SpaceTimeEvents:
    """
    Method for reformatting event data stored in a shapefile for use in
    calculating metrics of spatio-temporal interaction.

    Parameters
    ----------
    path            : string
                      the path to the appropriate shapefile, including the
                      file name, but excluding the extension              
    time            : string
                      column header in the DBF file indicating the column
                      containing the time stamp

    Attributes
    ----------
    n               : int
                      number of events
    x               : array
                      n x 1 array of the x coordinates for the events
    y               : array
                      n x 1 array of the y coordinates for the events
    t               : array
                      n x 1 array of the temporal coordinates for the events
    space           : array
                      n x 2 array of the spatial coordinates (x,y) for the
                      events
    time            : array
                      n x 2 array of the temporal coordinates (t,1) for the
                      events, the second column is a vector of ones

    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example shapefile data, ensuring to omit the file
    extension. In order to successfully create the event data the .dbf file
    associated with the shapefile should have a column of values that are a
    timestamp for the events. There should be a numerical value (not a
    date) in every field.       

    >>> path = pysal.examples.get_path("burkitt")

    Create an instance of SpaceTimeEvents from a shapefile, where the
    temporal information is stored in a column named "T". 

    >>> events = SpaceTimeEvents(path,'T')

    See how many events are in the instance. 

    >>> events.n
    188

    Check the spatial coordinates of the first event. 

    >>> events.space[0]
    array([ 300.,  302.])

    Check the time of the first event.

    >>> events.t[0]
    array([413])
    

    """
    def __init__(self, path, time_col):
        shp = pysal.open(path + '.shp')
        dbf = pysal.open(path + '.dbf')

        # extract the spatial coordinates from the shapefile
        x = []
        y = []
        n = 0
        for i in shp:
            count = 0
            for j in i:
                if count==0:
                    x.append(j)
                elif count==1:
                    y.append(j)
                count += 1
            n += 1

        self.n = n
        x = np.array(x)
        y = np.array(y)
        self.x = np.reshape(x,(n,1))
        self.y = np.reshape(y,(n,1))
        self.space = np.hstack((self.x,self.y))

        # extract the temporal information from the database
        t = np.array(dbf.by_col(time_col))
        line = np.ones((n,1))
        self.t = np.reshape(t,(n,1)) 
        self.time = np.hstack((self.t,line))
        
        # close open objects
        dbf.close()
        shp.close()

        

def knox(events, delta, tau, permutations=99):
    """
    Knox test for spatio-temporal interaction. [1]_

    Parameters
    ----------
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    delta           : float
                      threshold for proximity in space
    tau             : float
                      threshold for proximity in time
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99)

    Returns
    -------
    knox_result     : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue)
    stat            : float
                      value of the knox test for the dataset
    pvalue          : float
                      pseudo p-value associated with the statistic

    References
    ----------
    .. [1] E. Knox. 1964. The detection of space-time
       interactions. Journal of the Royal Statistical Society. Series C
       (Applied Statistics), 13(1):25-30.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = pysal.examples.get_path("burkitt")
    >>> events = SpaceTimeEvents(path,'T')

    Set the random seed generator. This is used by the permutation based
    inference to replicate the pseudo-significance of our example results -
    the end-user will normally omit this step.

    >>> np.random.seed(100)

    Run the Knox test with distance and time thresholds of 20 and 5,
    respectively. This counts the events that are closer than 20 units in
    space, and 5 units in time.  

    >>> result = knox(events,delta=20,tau=5,permutations=99)

    Next, we examine the results. First, we call the statistic from the
    results results dictionary. This reports that there are 13 events close
    in both space and time, according to our threshold definitions. 

    >>> print(result['stat'])
    13.0

    Next, we look at the pseudo-significance of this value, calculated by
    permuting the timestamps and rerunning the statistics. In this case,
    the results indicate there is likely no space-time interaction between
    the events.

    >>> print("%2.2f"%result['pvalue'])
    0.18
    

    """
    n = events.n
    s = events.space
    t = events.t

    # calculate the spatial and temporal distance matrices for the events
    sdistmat = cg.distance_matrix(s)   
    tdistmat = cg.distance_matrix(t)

    # identify events within thresholds
    spacmat = np.ones((n,n))
    test = sdistmat <= delta
    spacmat = spacmat * test

    timemat = np.ones((n,n))
    test = tdistmat <= tau
    timemat = timemat * test

    # calculate the statistic
    knoxmat = timemat * spacmat
    stat = (knoxmat.sum()-n)/2

    # return results (if no inference)
    if permutations==0: return stat
    distribution=[]

    # loop for generating a random distribution to assess significance
    for p in range(permutations):
        rtdistmat = util.shuffle_matrix(tdistmat,range(n))
        timemat = np.ones((n,n))
        test = rtdistmat <= tau
        timemat = timemat * test
        knoxmat = timemat*spacmat
        k = (knoxmat.sum()-n)/2
        distribution.append(k)

    # establish the pseudo significance of the observed statistic
    distribution=np.array(distribution)
    greater = np.ma.masked_greater_equal(distribution,stat)
    count = np.ma.count_masked(greater)
    pvalue = (count+1.0)/(permutations+1.0)


    # return results
    knox_result ={'stat':stat, 'pvalue':pvalue}
    return knox_result



def mantel(events, permutations=99, scon=1.0, spow=-1.0, tcon=1.0, tpow=-1.0):
    """
    Standardized Mantel test for spatio-temporal interaction. [2]_

    Parameters
    ----------
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99) 
    scon            : float
                      constant added to spatial distances
    spow            : float
                      value for power transformation for spatial distances
    tcon            : float
                      constant added to temporal distances
    tpow            : float
                      value for power transformation for temporal distances


    Returns
    -------
    mantel_result   : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue)
    stat            : float
                      value of the knox test for the dataset
    pvalue          : float
                      pseudo p-value associated with the statistic

    Reference
    ---------
    .. [2] N. Mantel. 1967. The detection of disease clustering and a
    generalized regression approach. Cancer Research, 27(2):209-220.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = pysal.examples.get_path("burkitt")
    >>> events = SpaceTimeEvents(path,'T')

    Set the random seed generator. This is used by the permutation based
    inference to replicate the pseudo-significance of our example results -
    the end-user will normally omit this step.

    >>> np.random.seed(100)

    The standardized Mantel test is a measure of matrix correlation between
    the spatial and temporal distance matrices of the event dataset. The
    following example runs the standardized Mantel test without a constant
    or transformation; however, as recommended by Mantel (1967) [2]_, these
    should be added by the user. This can be done by adjusting the constant
    and power parameters. 

    >>> result = mantel(events, 99, scon=1.0, spow=-1.0, tcon=1.0, tpow=-1.0)

    Next, we examine the result of the test. 

    >>> print("%6.6f"%result['stat'])
    0.048368

    Finally, we look at the pseudo-significance of this value, calculated by
    permuting the timestamps and rerunning the statistic for each of the 99
    permutations. According to these parameters, the results indicate 
    space-time interaction between the events.

    >>> print("%2.2f"%result['pvalue'])
    0.01


    """
    n = events.n
    s = events.space
    t = events.t

    # calculate the spatial and temporal distance matrices for the events
    distmat = cg.distance_matrix(s)   
    timemat = cg.distance_matrix(t)

    # calculate the transformed standardized statistic
    timevec = (util.get_lower(timemat)+tcon)**tpow
    distvec = (util.get_lower(distmat)+scon)**spow
    stat = stats.pearsonr(timevec,distvec)[0].sum()

    # return the results (if no inference)
    if permutations==0: return stat

    # loop for generating a random distribution to assess significance
    dist=[]
    for i in range(permutations):
        trand = util.shuffle_matrix(timemat,range(n))
        timevec = (util.get_lower(trand)+tcon)**tpow
        m = stats.pearsonr(timevec,distvec)[0].sum()
        dist.append(m)
 
    ## establish the pseudo significance of the observed statistic
    distribution=np.array(dist)
    greater = np.ma.masked_greater_equal(distribution,stat)
    count = np.ma.count_masked(greater)
    pvalue = (count+1.0)/(permutations+1.0)


    # report the results
    mantel_result = {'stat':stat, 'pvalue':pvalue}
    return mantel_result



def jacquez(events, k, permutations=99):
    """
    Jacquez k nearest neighbors test for spatio-temporal interaction. [3]_

    Parameters
    ----------
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    k               : int
                      the number of nearest neighbors to be searched
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99)

    Returns
    -------
    jacquez_result  : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue)
    stat            : float
                      value of the Jacquez k nearest neighbors test for the
                      dataset
    pvalue          : float
                      p-value associated with the statistic (normally
                      distributed with k-1 df)

    References
    ----------
    .. [3] G. Jacquez. 1996. A k nearest neighbour test for space-time
       interaction. Statistics in Medicine, 15(18):1935-1949.


    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example data and create an instance of SpaceTimeEvents.
    
    >>> path = pysal.examples.get_path("burkitt")
    >>> events = SpaceTimeEvents(path,'T')

    The Jacquez test counts the number of events that are k nearest
    neighbors in both time and space. The following runs the Jacquez test
    on the example data and reports the resulting statistic. In this case,
    there are 13 instances where events are nearest neighbors in both space
    and time. 

    >>> np.random.seed(100)
    >>> result = jacquez(events,k=3,permutations=99)
    >>> print result['stat']
    13

    The significance of this can be assessed by calling the p-
    value from the results dictionary, as shown below. Again, no 
    space-time interaction is observed.

    >>> print("%2.2f"%result['pvalue'])
    0.21

    """    
    n = events.n
    time = events.time
    space = events.space

    # calculate the nearest neighbors in space and time separately
    knnt = Distance.knnW(time,k)
    knns = Distance.knnW(space,k)

    nnt = knnt.neighbors
    nns = knns.neighbors
    knn_sum = 0

    # determine which events are nearest neighbors in both space and time 
    for i in range(n):
        t_neighbors = nnt[i]
        s_neighbors = nns[i]
        check = set(t_neighbors)
        inter = check.intersection(s_neighbors)
        count = len(inter)
        knn_sum += count

    stat = knn_sum

    # return the results (if no inference)
    if permutations==0: return stat

    # loop for generating a random distribution to assess significance
    dist=[]
    for p in range(permutations):
        j = 0
        trand = np.random.permutation(time)
        knnt = Distance.knnW(trand,k)
        nnt = knnt.neighbors
        for i in range(n):
            t_neighbors = nnt[i]
            s_neighbors = nns[i]
            check = set(t_neighbors)
            inter = check.intersection(s_neighbors)
            count = len(inter)
            j += count

        dist.append(j)

    # establish the pseudo significance of the observed statistic
    distribution=np.array(dist)
    greater = np.ma.masked_greater_equal(distribution,stat)
    count = np.ma.count_masked(greater)
    pvalue = (count+1.0)/(permutations+1.0)

    # report the results
    jacquez_result ={'stat':stat, 'pvalue':pvalue}
    return jacquez_result



def modified_knox(events, delta, tau, permutations=99):
    """
    Baker's modified Knox test for spatio-temporal interaction. [1]_

    Parameters
    ----------
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    delta           : float
                      threshold for proximity in space
    tau             : float
                      threshold for proximity in time
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99)

    Returns
    -------
    modknox_result  : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue)
    stat            : float
                      value of the modified knox test for the dataset
    pvalue          : float
                      pseudo p-value associated with the statistic

    References
    ----------
    .. [1] R.D. Baker. Identifying space-time disease clusters. Acta Tropica, 
       91(3):291-299, 2004
       

    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = pysal.examples.get_path("burkitt")
    >>> events = SpaceTimeEvents(path,'T')

    Set the random seed generator. This is used by the permutation based
    inference to replicate the pseudo-significance of our example results -
    the end-user will normally omit this step.

    >>> np.random.seed(100)

    Run the modified Knox test with distance and time thresholds of 20 and 5,
    respectively. This counts the events that are closer than 20 units in
    space, and 5 units in time.  

    >>> result = modified_knox(events,delta=20,tau=5,permutations=99)

    Next, we examine the results. First, we call the statistic from the
    results dictionary. This reports the difference between the observed
    and expected Knox statistic.  

    >>> print("%2.8f"%result['stat'])
    2.81016043

    Next, we look at the pseudo-significance of this value, calculated by
    permuting the timestamps and rerunning the statistics. In this case,
    the results indicate there is likely no space-time interaction.

    >>> print("%2.2f"%result['pvalue'])
    0.11

    """
    n = events.n
    s = events.space
    t = events.t

    # calculate the spatial and temporal distance matrices for the events
    sdistmat = cg.distance_matrix(s)   
    tdistmat = cg.distance_matrix(t)

    # identify events within thresholds
    spacmat = np.ones((n,n))
    spacbin = sdistmat <= delta
    spacmat = spacmat * spacbin
    timemat = np.ones((n,n))
    timebin = tdistmat <= tau
    timemat = timemat * timebin

    # calculate the observed (original) statistic
    knoxmat = timemat * spacmat
    obsstat = (knoxmat.sum()-n)

    # calculate the expectated value
    ssumvec = np.reshape((spacbin.sum(axis=0) - 1),(n,1))
    tsumvec = np.reshape((timebin.sum(axis=0) - 1),(n,1))
    expstat = (ssumvec*tsumvec).sum()

    # calculate the modified stat
    stat = (obsstat-(expstat/(n-1.0)))/2.0

    # return results (if no inference)
    if permutations==0: return stat
    distribution=[]

    # loop for generating a random distribution to assess significance
    for p in range(permutations):
        rtdistmat = util.shuffle_matrix(tdistmat,range(n))
        timemat = np.ones((n,n))
        timebin = rtdistmat <= tau
        timemat = timemat * timebin

        # calculate the observed knox again
        knoxmat = timemat * spacmat
        obsstat = (knoxmat.sum()-n)

        # calculate the expectated value again
        ssumvec = np.reshape((spacbin.sum(axis=0) - 1),(n,1))
        tsumvec = np.reshape((timebin.sum(axis=0) - 1),(n,1))
        expstat = (ssumvec*tsumvec).sum()

        # calculate the modified stat
        tempstat = (obsstat-(expstat/(n-1.0)))/2.0
        distribution.append(tempstat)


    # establish the pseudo significance of the observed statistic
    distribution=np.array(distribution)
    greater = np.ma.masked_greater_equal(distribution,stat)
    count = np.ma.count_masked(greater)
    pvalue = (count+1.0)/(permutations+1.0)

    # return results
    modknox_result ={'stat':stat, 'pvalue':pvalue}
    return modknox_result



def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == '__main__':
    _test()



