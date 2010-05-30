"""
Markov based methods for PySAL


To do:

    add mobility measures 
    add homogeneity and Markov property tests
"""

import numpy as np
import numpy.linalg as la
import pysal
import ergodic

__all__=["Markov","LISA_Markov","Spatial_Markov"]


class Markov:
    """
    Classic Markov transition matrices

    Parameters
    ----------
    class_ids    : array (n,t) 
                   One row per observation, one column per state of each
                   observation, with as many columns as time periods
    classes      : array (k) 
                   All different classes (bins) of the matrix

    Attributes
    ----------
    p            : matrix (k,k)
                   transition probility matrix

    steady_state : matrix (k,1)
                   ergodic distribution

    transitions  : matrix (k,k)
                   count of transitions between each state i and j

    Examples:
    ---------
    >>> c=np.array([['b','a','c'],['c','c','a'],['c','b','c'],['a','a','b'],['a','b','c']])
    >>> m=Markov(c)
    >>> m.classes
    array(['a', 'b', 'c'], 
          dtype='|S1')
    >>> m.p
    matrix([[ 0.25      ,  0.5       ,  0.25      ],
            [ 0.33333333,  0.        ,  0.66666667],
            [ 0.33333333,  0.33333333,  0.33333333]])
    >>> m.steady_state
    matrix([[ 0.30769231],
            [ 0.28846154],
            [ 0.40384615]])

    # US nominal per capita income 48 states 81 years 1929-2009
    >>> f=pysal.open("../../examples/usjoin.csv")
    >>> pci=np.array([f.by_col[str(y)] for y in range(1929,2010)])

    # set classes to quintiles for each year
    >>> q5=np.array([pysal.Quantiles(y).yb for y in pci]).transpose()
    >>> m=Markov(q5)
    >>> m.transitions
    array([[ 729.,   71.,    1.,    0.,    0.],
           [  72.,  567.,   80.,    3.,    0.],
           [   0.,   81.,  631.,   86.,    2.],
           [   0.,    3.,   86.,  573.,   56.],
           [   0.,    0.,    1.,   57.,  741.]])
    >>> m.p
    matrix([[ 0.91011236,  0.0886392 ,  0.00124844,  0.        ,  0.        ],
            [ 0.09972299,  0.78531856,  0.11080332,  0.00415512,  0.        ],
            [ 0.        ,  0.10125   ,  0.78875   ,  0.1075    ,  0.0025    ],
            [ 0.        ,  0.00417827,  0.11977716,  0.79805014,  0.07799443],
            [ 0.        ,  0.        ,  0.00125156,  0.07133917,  0.92740926]])
    >>> m.steady_state
    matrix([[ 0.20774716],
            [ 0.18725774],
            [ 0.20740537],
            [ 0.18821787],
            [ 0.20937187]])

    # Relative incomes
    >>> pci=pci.transpose()
    >>> rpci=pci/(pci.mean(axis=0))
    >>> rq=pysal.Quantiles(rpci.flatten()).yb
    >>> rq.shape=(48,81)
    >>> mq=Markov(rq)
    >>> mq.transitions
    array([[ 707.,   58.,    7.,    1.,    0.],
           [  50.,  629.,   80.,    1.,    1.],
           [   4.,   79.,  610.,   73.,    2.],
           [   0.,    7.,   72.,  650.,   37.],
           [   0.,    0.,    0.,   48.,  724.]])
    >>> mq.steady_state
    matrix([[ 0.17957376],
            [ 0.21631443],
            [ 0.21499942],
            [ 0.21134662],
            [ 0.17776576]])
    
    """
    def __init__(self,class_ids,classes=[]):
        if len(classes):
            self.classes=classes
        else:
            self.classes=np.unique(class_ids)

        n,t=class_ids.shape
        k=len(self.classes)
        js=range(t-1)

        classIds=self.classes.tolist()
        transitions=np.zeros((k,k))
        for state_0 in js:
            state_1=state_0+1
            state_0=class_ids[:,state_0]
            state_1=class_ids[:,state_1]
            initial=np.unique(state_0)
            for i in initial:
                ending=state_1[state_0==i]
                uending=np.unique(ending)
                row=classIds.index(i)
                for j in uending:
                    col=classIds.index(j)
                    transitions[row,col]+=sum(ending==j)
        self.transitions=transitions
        row_sum=transitions.sum(axis=1)
        p=np.dot(np.diag(1/(row_sum+(row_sum==0))),transitions)
        self.p=np.matrix(p)

        # steady_state vector 
        v,d=la.eig(np.transpose(self.p))
        # for a regular P maximum eigenvalue will be 1
        mv=max(v)
        # find its position
        i=v.tolist().index(mv)
        # normalize eigenvector corresponding to the eigenvalue 1
        self.steady_state= d[:,i]/sum(d[:,i])

class Spatial_Markov:
    """
    Markov transitions conditioned on the value of the spatial lag
    
    Parameters
    ----------

    y            : array (n,t) 
                   One row per observation, one column per state of each
                   observation, with as many columns as time periods

    w            : spatial weights object

    k            : integer
                   number of classes (quantiles)

    Attributes
    ----------
    p            : matrix (k,k)
                   transition probability matrix for a-spatial Markov

    s            : matrix (k,1)
                   ergodic distribution for a-spatial Markov

    transitions  : matrix (k,k)
                   counts of transitions between each state i and j
                   for a-spatial Markov

    T            : matrix (k,k,k)
                   counts of transitions for each conditional Markov
                   T[0] is the matrix of transitions for observations with
                   lags in the 0th quantile, T[k-1] is the transitions for
                   the observations with lags in the k-1th

    P            : matrix(k,k,k)
                   transition probability matrix for spatial Markov
                   first dimension is the conditioned on the lag


    S            : matrix(k,k)
                   steady state distributions for spatial Markov
                   each row is a conditional steady_state

    F            : matrix(k,k,k)
                   first mean passage times
                   first dimension is conditioned on the lag


    Notes
    -----
    Based on  Rey (2001) [1]_

    
    Examples
    --------
    >>> f=pysal.open("../../examples/usjoin.csv")
    >>> pci=np.array([f.by_col[str(y)] for y in range(1929,2010)])
    >>> pci=pci.transpose()

    relative incomes with simple contiguity 

    >>> rpci=pci/(pci.mean(axis=0))
    >>> w=pysal.open("../../examples/states48.gal").read()
    >>> sm=Spatial_Markov(rpci,w)
    >>> for p in sm.P:
    ...     print p
    ...     
    [[ 0.86470588  0.11764706  0.01323529  0.00441176]
     [ 0.29901961  0.57843137  0.10294118  0.01960784]
     [ 0.0483871   0.24193548  0.48387097  0.22580645]
     [ 0.06382979  0.06382979  0.23404255  0.63829787]]
    [[ 0.69339623  0.27830189  0.02358491  0.00471698]
     [ 0.11872146  0.74429224  0.12785388  0.00913242]
     [ 0.02803738  0.21962617  0.62149533  0.13084112]
     [ 0.          0.03296703  0.21978022  0.74725275]]
    [[ 0.68831169  0.23376623  0.06493506  0.01298701]
     [ 0.17619048  0.54285714  0.26190476  0.01904762]
     [ 0.0044843   0.12780269  0.71524664  0.15246637]
     [ 0.          0.01860465  0.25116279  0.73023256]]
    [[ 0.41666667  0.5         0.08333333  0.        ]
     [ 0.11764706  0.55882353  0.26470588  0.05882353]
     [ 0.03097345  0.13274336  0.60619469  0.2300885 ]
     [ 0.00844595  0.02533784  0.11655405  0.84966216]]

    The probability of a poor state remaining poor is 0.865 if their
    neighbors are poor, 0.693 if their neighbors are in the second
    quartile, 0.688 if they are in the third quartile and 0.417 if their
    neighbors are in the fourth quartile.

    >>> sm.ss
    array([[ 0.58782316,  0.23273848,  0.09829031,  0.08114805],
           [ 0.18806096,  0.4290554 ,  0.23975543,  0.14312821],
           [ 0.11354596,  0.18984682,  0.43300937,  0.26359785],
           [ 0.05725753,  0.17605819,  0.27575718,  0.4909271 ]])

    The long run distribution for states with poor (rich) neighbors has
    0.588 (0.057) of the values in the first quartile, 0.233 (0.176) in
    the second quartile, 0.0.98 (0.276) in the third and 0.081 (0.491)
    in the fourth quartile.

    >>> for f in sm.F:
    ...     print f
    ...     
    [[  1.70119192   8.28618421  26.83809207  44.98041833]
     [  4.73193953   4.29666806  21.93735075  40.40440826]
     [  7.99329754   6.36567982  10.17394282  25.09398059]
     [  8.77188773   8.34594298  11.37213697  12.3231546 ]]
    [[  5.31742481   3.87717709  12.20466959  32.58128415]
     [ 13.63327366   2.33070133   9.76395664  30.37914291]
     [ 17.9223053    6.14355367   4.17091708  22.68280767]
     [ 21.31938813   9.29874233   5.2300813    6.98674266]]
    [[  8.80700651   6.11889551   7.23710729  15.86674881]
     [ 22.35790647   5.26740443   5.14604327  14.07786188]
     [ 32.61629382  11.15262368   2.30941887  10.08009153]
     [ 35.61571538  14.09037377   4.06179609   3.79365763]]
    [[ 17.46495301   3.49409128   5.89710333  10.0826972 ]
     [ 27.3435784    5.67994031   4.87995388   8.71798134]
     [ 33.51796573  12.45863896   3.6263788    6.27099237]
     [ 37.24599226  16.50692747   7.80544747   2.0369623 ]]

    States with incomes in the first quartile with neighbors in the
    first quartile return to the first quartile after 1.701 years, after
    leaving the first quartile. They enter the fourth quartile after
    44.98 years after leaving the first quartile, on average
    Poor states within neighbors in the fourth quartile return to the
    first quartile, on average, after 17.47 years, and would enter the
    fourth quartile after 10.08 years.

    References
    ----------
    .. [1] Rey, S.J. 2001. "Spatial empirics for economic growth
       and convergence", 34 Geographical Analysis, 33, 195-214.
    
    """
    def __init__(self,y,w,k=4):

        classes=np.array([pysal.Quantiles(yi,k=k).yb for yi in y])
        classic=Markov(classes)
        self.aspatial=classic
        self.classes=classes

        # lag markov
        l_y=pysal.lag_spatial(w,y)
        l_classes=np.array([pysal.Quantiles(yi,k=k).yb for yi in l_y])
        l_classic=Markov(l_classes)
        self.lag_markov=l_classic
        self.l_classes=l_classes

        T=np.zeros((k,k,k))
        n,t=y.shape
        for t1 in range(t-1):
            t2=t1+1
            for i in range(n):
                T[l_classes[i,t1],classes[i,t1],classes[i,t2]]+=1
        self.T=T

        P=np.zeros_like(T)
        F=np.zeros_like(T) # fmpt
        ss=np.zeros_like(T[0])
        for i,mat in enumerate(T):
            p_i=np.matrix(np.diag(1./mat.sum(axis=1))*np.matrix(mat))
            ss[i]=ergodic.steady_state(p_i).transpose()
            F[i]=ergodic.fmpt(p_i)
            P[i]=p_i
        self.P=P
        self.ss=ss
        self.F=F
        self.T=T

        # add tests based on multinomial differences, classic against each of
        # the conditional transition matrices
            

class LISA_Markov(Markov):
    """
    Markov for Local Indicators of Spatial Association
    
    Parameters
    ----------

    y  : array (n,t)
         n cross-sectional units observed over t time periods

    w  : weights instance

    Attributes
    ----------
    classes      : array (4,1)
                   1=HH,2=LH,3=LL,4=HL (own,lag)
    p            : matrix (k,k)
                   transition probility matrix

    steady_state : matrix (k,1)
                   ergodic distribution

    transitions  : matrix (k,k)
                   count of transitions between each state i and j

    Examples:
    ---------
 
    >>> import numpy as np
    >>> f=pysal.open("../../examples/usjoin.csv")
    >>> pci=np.array([f.by_col[str(y)] for y in range(1929,2010)]).transpose()
    >>> w=pysal.open("../../examples/states48.gal").read()
    >>> lm=LISA_Markov(pci,w)
    >>> lm.classes
    array([1, 2, 3, 4])
    >>> lm.steady_state
    matrix([[ 0.31122083+0.j],
            [ 0.14496933+0.j],
            [ 0.37728447+0.j],
            [ 0.16652538+0.j]])
    >>> lm.transitions
    array([[ 495.,   47.,  384.,  244.],
           [ 180.,   83.,  178.,  115.],
           [ 334.,  321.,  661.,  190.],
           [ 169.,  105.,  218.,   83.]])
    >>> lm.p
    matrix([[ 0.42307692,  0.04017094,  0.32820513,  0.20854701],
            [ 0.32374101,  0.14928058,  0.32014388,  0.20683453],
            [ 0.22177955,  0.21314741,  0.43891102,  0.12616202],
            [ 0.29391304,  0.1826087 ,  0.37913043,  0.14434783]])
    
    """
    def __init__(self,y,w):
        y=y.transpose()
        q=np.array([pysal.Moran_Local(yi,w,permutations=0).q for yi in y])
        classes=np.arange(1,5) # no guarantee all 4 quadrants are visited
        Markov.__init__(self,q,classes)

def mobility(pmat):
    """
    Mobility indices for a Markov probability transition matrix.
    """
    pass

def prais(pmat):
    """
    Prais conditional mobility
    """
    return (pmat.sum(axis=1)-np.diag(pmat))[0]

def shorrock(pmat):
    """
    Shorrocks mobility measure
    """
    t=np.trace(pmat)
    k=pmat.shape[1]
    return (k-t)/(k-1)

def directional(pmat):
    """
    Directional mobility measures

    Returns
    -------


    direction  :  array (kx2)
                  conditional upward mobility measure (col 1)
                  conditional downward mobility measure (col 2)


    Notes
    -----

    2*shorrock(p) = direction(p).sum()

    """
    k=pmat.shape[1]
    p=pmat-np.eye(k)*np.diag(pmat)
    up=np.triu(p).sum(axis=1)
    down=np.tril(p).sum(axis=1)
    direction=np.hstack((up,down))
    return direction


def homogeneity(classids,colIds=None):
    """
    Test of Markov homogeneity property
    """
    m=Markov(classids)
    l=np.zeros_like(classids)
    for i,cl in enumerate(m.classes.tolist()):
        l+=(classids==cl)*i

    n,t=classids.shape
    
    k=m.classes.shape[0]
    rt=np.arange(t-1)
    ttype=l[:,rt]*k+l[:,rt+1]
    if colIds is None:
        first=int((t-1)/2.)
        print first



    return ttype 


def path_probabilities(class_ids,classes=[]):
    """
    Determines the probability of the observed paths conditional upon a
    probability transition matrix.


    """
    m=Markov(class_ids,classes)
    p=m.p
    s=np.zeros(class_ids.shape,int)
    for i,c in enumerate(m.classes):
        s+=(class_ids==c)*(i+1)

    s-=1
    n,T=class_ids.shape
    pp=np.zeros((n,T-1),float)
    for t in range(T-1):
        pp[:,t]=p[s[:,t],s[:,t+1]]

    
    p=pp.prod(axis=1)

    return (p,pp)

                
def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()




