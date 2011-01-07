"""
Methods for identifying space-time interaction in spatio-temporal event
data.
"""
__author__  = "Nicholas Malizia <nmalizia@asu.edu> "

import pysal
from pysal.common import *
import pysal.weights.Distance as Distance
import pysal.spatial_dynamics.util as util

__all__=['SpaceTimeEvents','knox','mantel','jacquez']


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

    Read in the example data.    

    >>> path = "../examples/burkitt"

    Create an instance of SpaceTimeEvents from a shapefile, where temporal
    information is stored in a column named "T". 

    >>> events = SpaceTimeEvents(path,'T')

    See how many events are in the instance. 

    >>> events.n
    188

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
    t               : array
                      n x 1 array (optional) of the temporal coordinates
                      for the events as produced by SpaceTimeEvents

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
    .. [1] E. Knox and M. Bartlett. 1964. The detection of space-time
       interactions. Journal of the Royal Statistical Society. Series C
       (Applied Statistics), 13(1):25-30.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = "../examples/burkitt"
    >>> events = SpaceTimeEvents(path,'T')

    Run the Knox test with a distance and time thresholds of 20 and 5.

    >>> result = knox(events,delta=20,tau=5,permutations=9)

    Print the results.

    >>> print result['stat']
    13.0

    """
    n = events.n
    s = events.space
    t = events.t

    # calculate the spatial and temporal distance matrices for the events
    sdistmat = Distance.distance_matrix(s)   
    tdistmat = Distance.distance_matrix(t)

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
        timemat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if rtdistmat[i,j] < tau:
                    timemat[i,j] = 1
        knoxmat = timemat*spacmat
        k = (knoxmat.sum()-n)/2
        distribution.append(k)

    # establish the pseudo significance of the observed statistic
    distribution=np.array(distribution)
    t = (stat-distribution.mean())/distribution.std()
    pvalue = stats.t.sf(t,permutations+1)

    # return results
    knox_result ={'stat':stat, 'pvalue':pvalue}
    return knox_result



def mantel(events, permutations=99, scon=0.0, spow=1.0, tcon=0.0, tpow=1.0):
    """
    Standardized Mantel test for spatio-temporal interaction. [2]_

    Parameters
    ----------
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99)                 
    t               : array
                      n x 1 array (optional) of the temporal coordinates
                      for the events as produced by SpaceTimeEvents

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

    >>> path = "../examples/burkitt"
    >>> events = SpaceTimeEvents(path,'T')

    Run the standardized Mantel test without a constant or transformation.

    >>> result = mantel(events, 9, scon=0.0, spow=1.0, tcon=0.0, tpow=1.0)

    Print the results.

    >>> print("%6.6f"%result['stat'])
    0.014154

    """
    n = events.n
    s = events.space
    t = events.t

    # calculate the spatial and temporal distance matrices for the events
    distmat = Distance.distance_matrix(s)   
    timemat = Distance.distance_matrix(t)

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

    # establish the pseudo significance of the observed statistic
    distribution=np.array(dist)
    t = (stat-distribution.mean())/distribution.std()
    pvalue = stats.t.sf(t,permutations+1)

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
    time            : array
                      n x 2 array (optional) of the temporal coordinates
                      for the events as produced by SpaceTimeEvents
    space           : array
                      n x 2 array (optional) of the spatial coordinates for
                      the events as produced by SpaceTimeEvents

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
    
    >>> path = "../examples/burkitt"
    >>> events = SpaceTimeEvents(path,'T')

    Run the Jacquez test and report the resulting statistic. 

    >>> result = jacquez(events,k=3,permutations=9)
    >>> print result['stat']
    13

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
    j = 0
    for p in range(permutations):
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
    distribution = np.array(dist)
    t = (stat-distribution.mean())/distribution.std()
    pvalue = stats.t.sf(t,permutations+1)

    # report the results
    jacquez_result ={'stat':stat, 'pvalue':pvalue}
    return jacquez_result



if __name__ == '__main__':

    import doctest
    doctest.testmod()


