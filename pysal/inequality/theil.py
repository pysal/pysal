
"""
Theil Inequality metrics for PySAL
----------------------------------------------------------------------
AUTHOR(S):  Sergio J. Rey sjrey@users.sourceforge.net
----------------------------------------------------------------------
Copyright (c) 2009 Sergio J. Rey
======================================================================


OVERVIEW

Classes:
    Theil            Theils T Inequality Measure
    TheilD           Spatial (or other) Decomposition of T
    TheilDSim        Inference on TheilD based on permutations

Notes:

    Currently this reproduces results from Rey and Sastre (2009).

    Porting over STARS code to move away from Numeric to numpy
"""

import numpy as num
SMALL = num.finfo('float').tiny

class Theil:
    """Computes aggregate Theil measure of inequality


    """

    def __init__(self,y):

        """
        Parameters:

            y - (n,t) or (n, ) array with n taken as the observations across which
            inequality is calculated

                If y is (n,) then a scalar inequality value is determined. If
                y is (n,t) then an array of inequality values are determined,
                one value for each column in y.

        
        """

        n = len(y)
        y = y + SMALL * (y==0) # can't have 0 values
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

        """
        Parameters:

            y - (n,t) or (n, ) array with n taken as the observations across which
            inequality is calculated

                If y is (n,) then a scalar inequality value is determined. If
                y is (n,t) then an array of inequality values are determined,
                one value for each column in y.

            partition - (n, ) array with elements indicating which partition
            each observation belongs to. These are assumed to be exhaustive.


        Attributes:

            T - global inequality
            bg - between group inequality
            wg - within group inequality

                Depending on the shape of y these attributes are either
                scalars or arrays.
        """

        groups=num.unique(partition)
        T=Theil(y).T
        ytot=y.sum(axis=0)

        #group totals
        gtot=num.array([y[partition==gid].sum(axis=0) for gid in groups])
        mm=num.dot

        """
        #group shares
        try:
            sg=mm(gtot,num.diag(1./ytot))
        except:
            sg=gtot/ytot
            sg.shape=(sg.size,1)
        """
        if ytot.size==1: # y is 1-d
            sg=gtot/ytot
            sg.shape=(sg.size,1)
        else:
            sg=mm(gtot,num.diag(1./ytot))
        ng=num.array([sum(partition==gid) for gid in groups])
        ng.shape=(ng.size,) # ensure ng is 1-d
        n=y.shape[0]
        # between group inequality
        bg=num.multiply(sg,num.log(mm(num.diag(n*1./ng),sg))).sum(axis=0)
        self.T=T
        self.bg=bg
        self.wg=T-bg



class TheilDSim:
    """Random permutation based inference on Theil Decomposition.
    
           """
    def __init__(self,y, partition, n_perm=99):
        """
        Parameters:
        ===========

            y - (n,t) or (n, ) array with n taken as the observations across which
            inequality is calculated

                If y is (n,) then a scalar inequality value is determined. If
                y is (n,t) then an array of inequality values are determined,
                one value for each column in y.

            partition - (n, ) array with elements indicating which partition
            each observation belongs to. These are assumed to be exhaustive.

            n_perm - number of permutations to use.


        Attributes:
        ===========

            observed - TheilD instance for the observed data. 

            bg - array (n_perm+1,t) between group inequality

            bg_pvalue - array (t,1) p-value for the between group measure.
            Measures the percentage of the realized values that were greater
            than or equal to the observed bg value. Includes the observed
            value.

            wg - array (size=n_perm+1) within group inequality
                Depending on the shape of y, 1 or 2-dimensional
        """

        observed=TheilD(y, partition)
        bg_ct=observed.bg==observed.bg # already have one extreme value
        bg_ct=bg_ct*1.0
        results=[observed]
        for perm in range(n_perm):
            yp=num.random.permutation(y)
            t=TheilD(yp,partition)
            bg_ct+=(1.0*t.bg>=observed.bg)
            results.append(t)
        self.results=results
        self.T=observed.T
        self.bg_pvalue=bg_ct/(n_perm*1.0 + 1)
        self.bg=num.array([r.bg for r in results])
        self.wg=num.array([r.wg for r in results])



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
    tall=TheilD(ymat,data['hanson98'])
    bgs=TheilDSim(ymat,partition)

    bgs1=TheilDSim(ymat[:,0],partition)


