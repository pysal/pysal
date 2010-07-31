"""
Rank and spatial rank mobility measures
"""
__author__  = "Sergio J. Rey <srey@asu.edu> "
from pysal.common import *
from scipy.stats.mstats import rankdata
from scipy.special import erfc

__all__=['Theta','SpatialTau']

class Theta:
    """
    Regime mobility measure

    For sequence of time periods Theta measures the extent to which rank
    changes for a variable measured over n locations are in the same direction
    within mutually exclusive and exhaustive partitions (regimes) of the n locations.

    Theta is defined as the sum of the absolute sum of rank changes within
    the regimes over the sum of all absolute rank changes. [1]_


    Parameters
    ----------
    y            : array (n,k) with k>=2
                   successive columns of y are later moments in time (years,
                   months,etc)
    regime       : array (n,)
                   values corresponding to which regime each observation belongs to
    permutations : int
                   number of random spatial permutations to generate for
                   computationally based inference

    Attributes
    ----------
    ranks        : array
                   ranks of the original y array (by columns)
    regimes      : array
                   the original regimes array 
    total        : array (k-1,)
                   the total number of rank changes for each of the k periods
    max_total    : int
                   the theoretical maximum number of rank changes for n
                   observations
    theta        : array (k-1,)
                   the theta statistic for each of the k-1 intervals
    permutations : int
                   the number of permutations
    pvalue_left  : float
                   p-value for test that observed theta is significantly lower
                   than its expectation under complete spatial randomness
    pvalue_right : float
                   p-value for test that observed theta is significantly
                   greater than its expectation under complete spatial randomness


    References
    ----------
    .. [1] Rey, S.J. (2004) "Spatial dependence in the evolution of regional
       income distributions," in A. Getis, J. Mur and H.Zoeller (eds). Spatial Econometrics and Spatial
       Statistics. Palgrave, London, pp. 194-213.


    Examples
    --------
    >>> import pysal
    >>> f=pysal.open("../examples/mexico.csv")
    >>> vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
    >>> y=np.transpose(np.array([f.by_col[v] for v in vnames]))
    >>> regime=np.array(f.by_col['esquivel99'])
    >>> np.random.seed(10)
    >>> t=Theta(y,regime,999)
    >>> t.theta
    array([[ 0.41538462,  0.28070175,  0.61363636,  0.62222222,  0.33333333,
             0.47222222]])
    >>> t.pvalue_left
    array([ 0.307,  0.077,  0.823,  0.552,  0.045,  0.735])
    >>> t.total
    array([ 130.,  114.,   88.,   90.,   90.,   72.])
    >>> t.max_total
    512
    >>> 
    """
    def __init__(self,y,regime,permutations=999):
        ranks=rankdata(y,axis=0)
        self.ranks=ranks
        n,k=y.shape
        ranks_d=ranks[:,range(1,k)] - ranks[:,range(k-1)]
        self.ranks_d=ranks_d
        regimes=sp.unique(regime)
        self.regimes=regimes
        self.total=sum(abs(ranks_d))
        self.max_total=sum([abs(i-n+i-1) for i in range(1,n+1)])
        self._calc(regime)
        self.theta=self._calc(regime)
        self.permutations=permutations
        if permutations:
            np.perm = np.random.permutation
            sim=np.array([self._calc(np.perm(regime)) for i in xrange(permutations)])
            self.theta.shape=(1,len(self.theta))
            sim=np.concatenate((self.theta,sim))
            self.sim=sim
            den=permutations+1.
            self.pvalue_left=(sim<=sim[0]).sum(axis=0)/den
            self.pvalue_right=(sim>sim[0]).sum(axis=0)/den
            self.z=(sim[0]-sim.mean(axis=0))/sim.std(axis=0)

    def _calc(self,regime):
        within=[abs(sum(self.ranks_d[regime==reg])) for reg in self.regimes]
        return np.array(sum(within)/self.total)

class SpatialTau:
    """
    Spatial version of Kendall's rank correlation statistic

    Kendall's Tau is based on a comparison of the number of pairs of n
    observations that have concordant ranks between two variables. The spatial
    Tau decomposes these pairs into those that are spatial neighbors and those
    that are not, and examines whether the rank correlation is different
    between the two sets. [1]_



    Parameters
    ----------
    x            : array (n,)
                   first variable
    y            : array (n,)
                   second variable
    w            : W
                   spatial weights object
    permutations : int
                   number of random spatial permutations for computationally
                   based inference

    Attributes
    ----------
    tau          : float
                   The classic Tau statistic
    wn           : int
                   The number of neighboring pairs
    tau_w        : float
                   Spatial Tau statistic
    tau_nw       : float
                   Tau for non-neighboring pairs
    p_tau_diff   : float
                   p-value for test of difference between tau_w and tau_nw
                   based on asymptotic distribution (Use with caution in small
                   samples).
    wnc          : int
                   number of concordant neighbor pairs
    wdc          : int
                   number of discordant neighbor pairs
    ev_wnc       : int
                   average value of concordant neighbor pairs under random
                   spatial permuations
    s_wnc        : float
                   standard deviation of the number of concordant neighbor
                   pairs under random spatial permutations.
    p_rand_wnc   : float
                   p-value for test of difference between wnc and its expected
                   value under spatial random permutations
    z_rand_wnc   : z-value for test of difference between wnc and its expected
                   value under spatial random permutations

    References
    ----------
    .. [1] Rey, S.J. (2004) "Spatial dependence in the evolution of regional
       income distributions," in A. Getis, J. Mur and H.Zoeller (eds). Spatial Econometrics and Spatial
       Statistics. Palgrave, London, pp. 194-213.


    Examples
    --------
    >>> import pysal
    >>> f=pysal.open("../examples/mexico.csv")
    >>> vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
    >>> y=np.transpose(np.array([f.by_col[v] for v in vnames]))
    >>> regime=np.array(f.by_col['esquivel99'])
    >>> w=pysal.weights.regime_weights(regime)
    >>> np.random.seed(10)
    >>> res=[SpatialTau(y[:,i],y[:,i+1],w,99) for i in range(6)]
    >>> for r in res:
    ...     "%8.3f %8.3f %8.3f"%(r.wnc,r.ev_wnc,r.p_rand_wnc)
    ...     
    '  44.000   52.354    0.000'
    '  47.000   53.576    0.006'
    '  52.000   55.747    0.031'
    '  54.000   55.556    0.212'
    '  53.000   53.384    0.436'
    '  57.000   57.566    0.390'
    >>> 
    """
    def __init__(self, x, y, w, permutations=0):
        self.x=x
        self.y=y
        self.w=w
        self.wn=w.s0/2. # number of neighboring pairs
        self.permutations = permutations
        res=self._calc(x,y,w)
        self.tau=res['tau']
        self.tau_w=res['tau_w']
        self.tau_nw=res['tau_nw']
        self.p_tau_diff=res['p_tau_diff']
        self.wnc = res['wnc'] # number of concordant neighbor pairs
        self.wdc = res['wdc'] # number of concordant nonneighbor pairs

        n=len(y)
        diff=self.tau_w-self.tau_nw
        adiff=abs(diff)
        ids=range(n)
        p=0
        pa=0
        pwnc=0
        counts=[]
        if permutations:
            for it in range(permutations):
                id=np.random.permutation(ids)
                r=self._calc(x[id],y[id],w)
                d=r['tau_w']-r['tau_nw']
                if d >= diff:
                    p+=1
                if abs(d) >= adiff:
                    pa+=1
                counts.append(r['wnc'])
            self.p_rand = p*1./(permutations+1)
            self.pa_rand =pa*1./(permutations+1)
            counts=np.array(counts)
            self.ev_wnc=ev_wnc=counts.mean()
            self.s_wnc=s_wnc=counts.std(ddof=1)
            z=(self.wnc-ev_wnc)/s_wnc
            self.p_rand_wnc= 1.0-sp.stats.norm.cdf(abs(z))
            self.z_rand_wnc=z


    def _calc(self,x,y,w):
        tx=0
        ty=0
        wtx=0
        wty=0
        nc = 0
        nd = 0
        wnc = 0
        wdc = 0
        n1 = 0
        n2 = 0
        n=len(x)
        self.wn = w.s0/2. # number of neighboring pairs
        ids=w.id_order
        for j in range(n-1):
            for k in range(j+1,n):
                wj=ids[j]
                wk=ids[k]
                dx = x[j] - x[k]
                dy = y[j] - y[k]
                p = dx * dy
                if (p):  
                    if p > 0:
                        nc+=1
                        if wk in w[wj]:
                            wnc+=1
                    else:
                        nd+=1
                        if wk in w[wj]:
                            wdc+=1
                else:
                    if dx:
                        ty+=1
                        if wk in w[wj]:
                            wty+=1
                    if dy:
                        tx+=1
                        if wk in w[wj]:
                            wtx+=1
        den=n*(n-1)/2.
        results={}
        tau = (nc-nd) / den
        results['tau']=tau
        tau_b = (nc-nd) / np.sqrt( (den-tx) * (den-ty))
        results['tau_b']=tau_b
        ties_y=ty
        results['ties_y']=ties_y
        ties_x=tx
        results['ties_x']=ties_x
        nc=nc
        results['nc']=nc
        nd=nd
        results['nd']=nd
        vtau=(2*(2*n+5.))/(9*n*(n+1.))
        results['vtau']=vtau
        ztau=tau/np.sqrt(vtau)
        results['ztau']=ztau
        p_norm=erfc(abs(ztau)/1.4142136)
        results['p_norm']=p_norm
        wnc=wnc
        results['wnc']=wnc
        wdc=wdc
        results['wdc']=wdc

        tau_w = (wnc-wdc) / self.wn
        results['tau_w']=tau_w
        tau_nw = (nc-wnc - (nd-wdc)) / (den - self.wn)
        results['tau_nw']=tau_nw
        wn=self.wn
        vtau_w=(4*wn+10.)/(9*wn*(wn+1.))
        results['vtau_w']=vtau_w
        vtau_nw=(4*(den-wn)+10)/(9*(den-wn)*(den-wn+1.))
        results['vtau_nw']=vtau_nw
        z_tau_w=tau_w/np.sqrt(vtau_w)
        results['z_tau_w']=z_tau_w
        p_tau_w_norm=erfc(abs(z_tau_w)/1.4142136) # from scipy.stats.kedalltau
        results['p_tau_w_norm']=p_tau_w_norm
        z_tau_nw=tau_nw/np.sqrt(vtau_nw)
        results['z_tau_nw']=z_tau_nw
        p_tau_nw_norm=erfc(abs(z_tau_nw)/1.4142136)
        results['p_tau_nw_norm']=p_tau_nw_norm
        # difference of taus for contiguous versus non contiguous
        d=z_tau_w-z_tau_nw
        p_tau_diff=1.0-sp.stats.norm.cdf(abs(d),0,2)
        results['p_tau_diff']=p_tau_diff
        return results


if __name__ == '__main__':
    import doctest
    doctest.testmod()

