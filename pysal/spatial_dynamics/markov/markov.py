"""
Markov based methods for PySAL



To do:

    add mobility measures 
    add homogeneity and Markov property tests
"""

import numpy as np
import numpy.linalg as la
import pysal


class Markov:
    """Classic Markov transition matrices

    Parameters
    ----------
    class_ids   : array (n,t) 
                  One row per observation, one column per state of each
                  observation, with as many columns as time periods
    classes     : array (k) 
                  All different classes (bins) of the matrix

    Attributes
    ----------
    p           : transition matrix
    steady_state: steady state

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
    """Markov for Local Indicators of Spatial Association"""
    def __init__(self,y,w):
        q=pysal.Local_Moran(y,w).q
        classes=range(1,5) # no guarantee all 4 quadrants are visited
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




