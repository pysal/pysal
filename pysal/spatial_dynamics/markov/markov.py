"""
Markov based methods for PySAL



To do:

    add mobility measures 
    add homogeneity and Markov property tests
"""

import numpy as np
import numpy.linalg as la
import pysal

__all__=["Markov","LISA_Markov"]


class Markov:
    """Classic Markov transition matrices

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




