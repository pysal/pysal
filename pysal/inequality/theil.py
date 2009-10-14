
"""
Thei Inequality metrics for PySAL
----------------------------------------------------------------------
AUTHOR(S):  Sergio J. Rey sjrey@users.sourceforge.net
----------------------------------------------------------------------
Copyright (c) 2009 Sergio J. Rey
======================================================================


OVERVIEW

Classes:
    Theil            Theils T Inequality Measure
    TheilD           Spatial (or other) Decomposition of T

    to add
    TheilDSim        Inference on TheilD based on permutations


Notes:

    Currently this reproduces results from Rey and Sastre (2009).

    Porting over STARS code to move away from Numeric to numpy

    Work in progress


"""

import numpy as num
SMALL = 0.0000001

class Theil:
    """Computes aggregate Theil measure of inequality


    """

    def __init__(self,y):
        n = len(y)
        y = y + SMALL * (y==0)
        yt = y.sum(axis=0)
        s = y/(yt*1.0)
        lns = num.log(n*s)
        slns = s*lns
        t=sum(slns)
        self.T = t


class TheilD:
    """Computes a decomposition of Theil's T based on partitioning of
    observations into exhaustive and mutually exclusive groups """
    def __init__(self,y, partition):

        groups=num.unique(partition)
        T=Theil(y).T
        ytot=y.sum(axis=0)

        #group totals
        gtot=[y[partition==gid].sum(axis=0) for gid in groups]
        mm=num.dot

        #group shares
        sg=mm(gtot,num.diag(1./ytot))
        ng=num.array([sum(partition==gid) for gid in groups])
        n=y.shape[0]
        # between group inequality
        bg=num.multiply(sg,num.log(mm(num.diag(n*1./ng),sg))).sum(axis=0)
        
        self.T=T
        self.bg=bg
        self.wg=T-bg



if __name__ == '__main__':


    import numpy as num
    def read_array(filename, dtype, separator=',',skiprows=0):
        """ Read a file with an arbitrary number of columns.
            The type of data in each column is arbitrary
            It will be cast to the given dtype at runtime
        """
        cast = num.cast
        data = [[] for dummy in xrange(len(dtype))]
        for li,line in enumerate(open(filename, 'r')):
            if li>=skiprows:
                fields = line.strip().split(separator)
                for i, number in enumerate(fields):
                    data[i].append(number)
        for i in xrange(len(dtype)):
            data[i] = cast[dtype[i]](data[i])
        return num.rec.array(data, dtype=dtype)




    descr= num.dtype([('State','S25'),
                    ('pcgdp1940','float'),
                    ('pcgdp1950','float'),
                    ('pcgdp1960','float'),
                    ('pcgdp1970','float'),
                    ('pcgdp1980','float'),
                    ('pcgdp1990','float'),
                    ('pcgdp2000','float'),
                    ('hanson03','float'),
                    ('hanson98','float'),
                    ('esquivel99','float'),
                    ('inegi','float'),
                    ('inegi2','float'),
                    ])
    data=read_array('sea_mexico.csv',descr,skiprows=1)

    x=num.arange(100)
    xt=Theil(x)

    pref="pcgdp"
    vnames=[ pref+str(year) for year in range(1940,2010,10)]
    globalT=[Theil(data[v]) for v in vnames]

    ymat=num.array([data[v] for v in vnames])
    ymat=num.transpose(ymat)

    
    partition=data['hanson98']
    y=ymat
    ytot=y.sum(axis=0)
    gtot=[y[partition==gid].sum(axis=0) for gid in num.unique(partition)]
    mm=num.dot
    sg=mm(gtot,num.diag(1./ytot))
    ng=num.array([sum(partition==gid) for gid in num.unique(partition)])
    n=y.shape[0]
    bg=num.multiply(sg,num.log(mm(num.diag(n*1./ng),sg))).sum(axis=0)
    tall=TheilD(ymat,data['hanson98'])


