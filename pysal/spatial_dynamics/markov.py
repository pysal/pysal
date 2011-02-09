"""
Markov based methods for spatial dynamics
"""
__author__ =  "Sergio J. Rey <srey@asu.edu"

import numpy as np
import numpy.linalg as la
from ergodic import fmpt, steady_state
from scipy import stats
import pysal

__all__ = ["Markov", "LISA_Markov", "Spatial_Markov" ]

# TT predefine LISA transitions
# TT[i,j] is the transition type from i to j
# i = quadrant in period 0
# j = quadrant in period 1
# uses one offset so first row and col of TT are ignored 
TT = np.zeros((5, 5), int)
c = 1
for i in range(1, 5):
    for j in range(1, 5):
        TT[i, j] = c
        c += 1

# MOVE_TYPES is a dictionary that returns the move type of a LISA transition
# filtered on the significance of the LISA end points
# True indicates significant LISA in a particular period
# e.g. a key of (1, 3, True, False) indicates a significant LISA located in
# quadrant 1 in period 0 moved to quadrant 3 in period 1 but was not
# significant in quadrant 3.

MOVE_TYPES = {}
c = 1
for i in range(1, 5):
    for j in range(1, 5):
        key = (i, j, True, True)
        MOVE_TYPES[key] = c
        c += 1
    for j in range(1, 5):
        key = (i, j, False, True)
        MOVE_TYPES[key] = c
    c += 1
for i in range(1, 5):
    for j in range(1, 5):
        key = (i, j, True, False)
        MOVE_TYPES[key] = c
        key = (i, j, False, False)
        MOVE_TYPES[key] = 25
    c += 1

class Markov:
    """
    Classic Markov transition matrices

    Parameters
    ----------
    class_ids    : array (n, t) 
                   One row per observation, one column recording the state of each
                   observation, with as many columns as time periods
    classes      : array (k) 
                   All different classes (bins) of the matrix

    Attributes
    ----------
    p            : matrix (k, k)
                   transition probability matrix

    steady_state : matrix (k, 1)
                   ergodic distribution

    transitions  : matrix (k, k)
                   count of transitions between each state i and j

    Examples
    --------
    >>> c = np.array([['b','a','c'],['c','c','a'],['c','b','c'],['a','a','b'],['a','b','c']])
    >>> m = Markov(c)
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

    US nominal per capita income 48 states 81 years 1929-2009

    >>> import pysal
    >>> f = pysal.open("../examples/usjoin.csv")
    >>> pci = np.array([f.by_col[str(y)] for y in range(1929,2010)])

    set classes to quintiles for each year

    >>> q5 = np.array([pysal.Quantiles(y).yb for y in pci]).transpose()
    >>> m = Markov(q5)
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

    Relative incomes

    >>> pci = pci.transpose()
    >>> rpci = pci/(pci.mean(axis=0))
    >>> rq = pysal.Quantiles(rpci.flatten()).yb
    >>> rq.shape = (48,81)
    >>> mq = Markov(rq)
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
    def __init__(self, class_ids, classes=[]):
        if len(classes):
            self.classes = classes
        else:
            self.classes = np.unique(class_ids)

        n, t = class_ids.shape
        k = len(self.classes)
        js = range(t-1)

        classIds = self.classes.tolist()
        transitions=np.zeros((k,k))
        for state_0 in js:
            state_1 = state_0+1
            state_0 = class_ids[:,state_0]
            state_1 = class_ids[:,state_1]
            initial = np.unique(state_0)
            for i in initial:
                ending = state_1[state_0 == i]
                uending = np.unique(ending)
                row = classIds.index(i)
                for j in uending:
                    col = classIds.index(j)
                    transitions[row, col] += sum(ending == j)
        self.transitions = transitions
        row_sum = transitions.sum(axis=1)
        p = np.dot(np.diag(1/(row_sum+(row_sum == 0))), transitions)
        self.p = np.matrix(p)

        # steady_state vector 
        v, d = la.eig(np.transpose(self.p))
        # for a regular P maximum eigenvalue will be 1
        mv = max(v)
        # find its position
        i = v.tolist().index(mv)
        # normalize eigenvector corresponding to the eigenvalue 1
        self.steady_state =  d[:,i]/sum(d[:,i])

class Spatial_Markov:
    """
    Markov transitions conditioned on the value of the spatial lag
    
    Parameters
    ----------

    y               : array (n,t) 
                      One row per observation, one column per state of each
                      observation, with as many columns as time periods

    w               : spatial weights object

    k               : integer
                      number of classes (quantiles)

    permutations    : int
                      number of permutations (default=0) for use in randomization
                      based inference

    fixed           : boolean
                      If true, quantiles are taken over the entire n*t
                      pooled series. If false, quantiles are taken each
                      time period over n.

    Attributes
    ----------
    p               : matrix (k, k)
                      transition probability matrix for a-spatial Markov
    s               : matrix (k, 1)
                      ergodic distribution for a-spatial Markov
    transitions     : matrix (k, k)
                      counts of transitions between each state i and j
                      for a-spatial Markov
    T               : matrix (k, k, k)
                      counts of transitions for each conditional Markov
                      T[0] is the matrix of transitions for observations with
                      lags in the 0th quantile, T[k-1] is the transitions for
                      the observations with lags in the k-1th
    P               : matrix(k, k, k)
                      transition probability matrix for spatial Markov
                      first dimension is the conditioned on the lag
    S               : matrix(k, k)
                      steady state distributions for spatial Markov
                      each row is a conditional steady_state
    F               : matrix(k, k, k)
                      first mean passage times
                      first dimension is conditioned on the lag
    shtest          : list (k elements)
                      each element of the list is a tuple for a multinomial
                      difference test between the steady state distribution from
                      a conditional distribution versus the overall steady state
                      distribution, first element of the tuple is the chi2 value,
                      second its p-value and the third the degrees of freedom
    chi2            : list (k elements)
                      each element of the list is a tuple for a chi-squared test
                      of the difference between the conditional transition
                      matrix against the overall transition matrix, first
                      element of the tuple is the chi2 value, second its
                      p-value and the third the degrees of freedom 
    x2              : float
                      sum of the chi2 values for each of the conditional tests
                      (see chi2 above)
    x2_pvalue       : float (if permutations>0)
                      pseudo p-value for x2 based on random spatial permutations
                      of the rows of the original transitions 
    x2_realizations : array (permutations,1)
                      the values of x2 for the random permutations 

    Notes
    -----
    Based on  Rey (2001) [1]_

    The shtest and chi2 tests should be used with caution as they are based on
    classic theory assuming random transitions. The x2 based test is
    preferable since it simulates the randomness under the null. It is an
    experimental test requiring further analysis.

    
    Examples
    --------
    >>> import pysal
    >>> f = pysal.open("../examples/usjoin.csv")
    >>> pci = np.array([f.by_col[str(y)] for y in range(1929,2010)])
    >>> pci = pci.transpose()
    >>> rpci = pci/(pci.mean(axis=0))
    >>> w = pysal.open("../examples/states48.gal").read()
    >>> w.transform = 'r'
    >>> sm = Spatial_Markov(rpci, w, fixed=True, k=5)
    >>> for p in sm.P:
    ...     print p
    ...     
    [[ 0.96341463  0.0304878   0.00609756  0.          0.        ]
     [ 0.06040268  0.83221477  0.10738255  0.          0.        ]
     [ 0.          0.14        0.74        0.12        0.        ]
     [ 0.          0.03571429  0.32142857  0.57142857  0.07142857]
     [ 0.          0.          0.          0.16666667  0.83333333]]
    [[ 0.79831933  0.16806723  0.03361345  0.          0.        ]
     [ 0.0754717   0.88207547  0.04245283  0.          0.        ]
     [ 0.00537634  0.06989247  0.8655914   0.05913978  0.        ]
     [ 0.          0.          0.06372549  0.90196078  0.03431373]
     [ 0.          0.          0.          0.19444444  0.80555556]]
    [[ 0.84693878  0.15306122  0.          0.          0.        ]
     [ 0.08133971  0.78947368  0.1291866   0.          0.        ]
     [ 0.00518135  0.0984456   0.79274611  0.0984456   0.00518135]
     [ 0.          0.          0.09411765  0.87058824  0.03529412]
     [ 0.          0.          0.          0.10204082  0.89795918]]
    [[ 0.8852459   0.09836066  0.          0.01639344  0.        ]
     [ 0.03875969  0.81395349  0.13953488  0.          0.00775194]
     [ 0.0049505   0.09405941  0.77722772  0.11881188  0.0049505 ]
     [ 0.          0.02339181  0.12865497  0.75438596  0.09356725]
     [ 0.          0.          0.          0.09661836  0.90338164]]
    [[ 0.33333333  0.66666667  0.          0.          0.        ]
     [ 0.0483871   0.77419355  0.16129032  0.01612903  0.        ]
     [ 0.01149425  0.16091954  0.74712644  0.08045977  0.        ]
     [ 0.          0.01036269  0.06217617  0.89637306  0.03108808]
     [ 0.          0.          0.          0.02352941  0.97647059]]
    

    The probability of a poor state remaining poor is 0.963 if their
    neighbors are in the 1st quintile and 0.798 if their neighbors are
    in the 2nd quintile. The probability of a rich economy remaining
    rich is 0.977 if their neighbors are in the 5th quintile, but if their
    neights are in the 4th quintile this drops to 0.903.

    >>> sm.S
    array([[ 0.43509425,  0.2635327 ,  0.20363044,  0.06841983,  0.02932278],
           [ 0.13391287,  0.33993305,  0.25153036,  0.23343016,  0.04119356],
           [ 0.12124869,  0.21137444,  0.2635101 ,  0.29013417,  0.1137326 ],
           [ 0.0776413 ,  0.19748806,  0.25352636,  0.22480415,  0.24654013],
           [ 0.01776781,  0.19964349,  0.19009833,  0.25524697,  0.3372434 ]])

    The long run distribution for states with poor (rich) neighbors has
    0.435 (0.018) of the values in the first quintile, 0.263 (0.200) in
    the second quintile, 0.204 (0.190) in the third, 0.0684 (0.255) in the
    fourth and 0.029 (0.337) in the fifth quintile.

    >>> for f in sm.F:
    ...     print f
    ...     
    [[   2.29835259   28.95614035   46.14285714   80.80952381  279.42857143]
     [  33.86549708    3.79459555   22.57142857   57.23809524  255.85714286]
     [  43.60233918    9.73684211    4.91085714   34.66666667  233.28571429]
     [  46.62865497   12.76315789    6.25714286   14.61564626  198.61904762]
     [  52.62865497   18.76315789   12.25714286    6.           34.1031746 ]]
    [[   7.46754205    9.70574606   25.76785714   74.53116883  194.23446197]
     [  27.76691978    2.94175577   24.97142857   73.73474026  193.4380334 ]
     [  53.57477715   28.48447637    3.97566318   48.76331169  168.46660482]
     [  72.03631562   46.94601483   18.46153846    4.28393653  119.70329314]
     [  77.17917276   52.08887197   23.6043956     5.14285714   24.27564033]]
    [[   8.24751154    6.53333333   18.38765432   40.70864198  112.76732026]
     [  47.35040872    4.73094099   11.85432099   34.17530864  106.23398693]
     [  69.42288828   24.76666667    3.794921     22.32098765   94.37966594]
     [  83.72288828   39.06666667   14.3           3.44668119   76.36702977]
     [  93.52288828   48.86666667   24.1           9.8           8.79255406]]
    [[  12.87974382   13.34847151   19.83446328   28.47257282   55.82395142]
     [  99.46114206    5.06359731   10.54545198   23.05133495   49.68944423]
     [ 117.76777159   23.03735526    3.94436301   15.0843986    43.57927247]
     [ 127.89752089   32.4393006    14.56853107    4.44831643   31.63099455]
     [ 138.24752089   42.7893006    24.91853107   10.35          4.05613474]]
    [[  56.2815534     1.5          10.57236842   27.02173913  110.54347826]
     [  82.9223301     5.00892857    9.07236842   25.52173913  109.04347826]
     [  97.17718447   19.53125       5.26043557   21.42391304  104.94565217]
     [ 127.1407767    48.74107143   33.29605263    3.91777427   83.52173913]
     [ 169.6407767    91.24107143   75.79605263   42.5           2.96521739]]

    States with incomes in the first quintile with neighbors in the
    first quintile return to the first quartile after 2.298 years, after
    leaving the first quintile. They enter the fourth quintile after
    80.810 years after leaving the first quintile, on average.
    Poor states within neighbors in the fourth quintile return to the
    first quintile, on average, after 12.88 years, and would enter the
    fourth quintile after 28.473 years.

    >>> np.matrix(sm.chi2)
    matrix([[  4.06139105e+01,   6.32961385e-04,   1.60000000e+01],
            [  5.55485793e+01,   2.88879565e-06,   1.60000000e+01],
            [  1.77772638e+01,   3.37100315e-01,   1.60000000e+01],
            [  4.00925436e+01,   7.54729084e-04,   1.60000000e+01],
            [  4.68588786e+01,   7.16364084e-05,   1.60000000e+01]])
    >>> np.matrix(sm.shtest)
    matrix([[  4.61209613e+02,   0.00000000e+00,   4.00000000e+00],
            [  1.48140694e+02,   0.00000000e+00,   4.00000000e+00],
            [  6.33129261e+01,   5.83089133e-13,   4.00000000e+00],
            [  7.22778509e+01,   7.54951657e-15,   4.00000000e+00],
            [  2.32659201e+02,   0.00000000e+00,   4.00000000e+00]])
    

    References
    ----------

    .. [1] Rey, S.J. 2001. "Spatial empirics for economic growth
       and convergence", 34 Geographical Analysis, 33, 195-214.
    
    """
    def __init__(self, y, w, k=4, permutations=0, fixed=False):

        self.y = y
        rows,cols = y.shape
        self.cols = cols
        npm = np.matrix
        npa = np.array
        self.fixed = fixed
        if fixed:
            yf = y.flatten()
            yb = pysal.Quantiles(yf, k=k).yb
            yb.shape = (rows, cols)
            classes = yb
        else:
            classes = npa([pysal.Quantiles(y[:,i],k=k).yb for i in np.arange(cols)]).transpose()
        classic = Markov(classes)
        self.classes = classes
        self.p = classic.p
        self.s = classic.steady_state
        self.transitions = classic.transitions
        T, P, ss, F = self._calc(y, w, classes, k=k)
        self.T = T
        self.P = P
        self.S = ss
        self.F = F
        self.shtest = self._mn_test()
        self.chi2 = self._chi2_test()
        self.x2 = sum([c[0] for c in self.chi2])

        if permutations:
            nrp = np.random.permutation
            rp = range(permutations)
            counter = 0
            x2_realizations = np.zeros((permutations, 1))
            x2ss = []
            for perm in range(permutations):
                T, P, ss, F = self._calc(nrp(y), w, classes, k=k)
                x2 = [chi2(T[i],self.transitions)[0] for i in range(k)]
                x2s = sum(x2)
                x2_realizations[perm] = x2s
                if x2s >= self.x2:
                    counter += 1
            self.x2_pvalue = (counter+1.0)/(permutations+1.)
            self.x2_realizations = x2_realizations


    def _calc(self, y, w, classes, k):
        # lag markov
        ly = pysal.lag_spatial(w, y)
        npm = np.matrix
        npa = np.array
        if self.fixed:
            l_classes = pysal.Quantiles(ly.flatten(), k=k).yb
            l_classes.shape = ly.shape
        else:
            l_classes = npa([pysal.Quantiles(ly[:,i],k=k).yb for i in np.arange(self.cols)])
            l_classes = l_classes.transpose()
        l_classic = Markov(l_classes)
        T = np.zeros((k, k, k))
        n, t = y.shape
        for t1 in range(t-1):
            t2 = t1 + 1
            for i in range(n):
                T[l_classes[i, t1], classes[i, t1], classes[i, t2]] += 1

        P = np.zeros_like(T)
        F = np.zeros_like(T) # fmpt
        ss = np.zeros_like(T[0])
        for i, mat in enumerate(T):
            p_i = np.matrix(np.diag(1./mat.sum(axis=1))*np.matrix(mat))
            ss[i] = steady_state(p_i).transpose()
            try:
                F[i] = fmpt(p_i)
            except:
                print "Singlular fmpt matrix for class ",i
            P[i] = p_i
        return T, P, ss, F

    def _mn_test(self):
        """
        helper to calculate tests of differences between steady state
        distributions from the conditional and overall distributions.
        """
        n, t = self.y.shape
        nt =  n * (t - 1)
        n0, n1, n2 = self.T.shape
        rn = range(n0)
        mat = [self._ssmnp_test(self.s, self.S[i], self.T[i].sum()) for i in rn]
        return mat


    def _ssmnp_test(self, p1, p2, nt):
        """
        Steady state multinomial probability difference test

        Arguments
        ---------

        p1       :  array (k, 1)
                    first steady state probability distribution

        p1       :  array (k, 1)
                    second steady state probability distribution

        nt       :  int
                    number of transitions to base the test on


        Returns
        -------

        implicit : tuple (3 elements)
                   (chi2 value, pvalue, degrees of freedom)

        """
        p1 = np.array(p1)
        k, c = p1.shape
        p1.shape = (k, )
        o = nt * p2
        e = nt * p1
        d = np.multiply((o - e), (o - e))
        d = d/e
        chi2 = d.sum()
        pvalue = 1-stats.chi2.cdf(chi2, k-1)
        return (chi2,pvalue,k-1)

    def _chi2_test(self):
        """
        helper to calculate tests of differences between the conditional
        transition matrices and the overall transitions matrix.
        """
        n, t = self.y.shape
        n0, n1, n2 = self.T.shape
        rn = range(n0)
        mat = [chi2(self.T[i], self.transitions) for i in rn]
        return mat

def chi2(T1, T2):
    """
    chi-squared test of difference between two transition matrices.

    Parameters
    ----------

    T1   : matrix (k, k)
           matrix of transitions (counts)

    T2   : matrix (k, k)
           matrix of transitions (counts) to use to form the probabilities
           under the null


    Returns
    -------

    implicit : tuple (3 elements)
               (chi2 value, pvalue, degrees of freedom)

    Examples
    --------

    >>> import pysal
    >>> f = pysal.open("../examples/usjoin.csv")
    >>> pci = np.array([f.by_col[str(y)] for y in range(1929,2010)]).transpose()
    >>> rpci = pci/(pci.mean(axis=0))
    >>> w = pysal.open("../examples/states48.gal").read()
    >>> w.transform='r'
    >>> sm = Spatial_Markov(rpci, w, fixed=True)
    >>> T1 = sm.T[0]
    >>> T1
    array([[ 562.,   22.,    1.,    0.],
           [  12.,  201.,   22.,    0.],
           [   0.,   17.,   97.,    4.],
           [   0.,    0.,    3.,   19.]])
    >>> T2 = sm.transitions
    >>> T2
    array([[ 884.,   77.,    4.,    0.],
           [  68.,  794.,   87.,    3.],
           [   1.,   92.,  815.,   51.],
           [   1.,    0.,   60.,  903.]])
    >>> chi2(T1,T2)
    (23.422628044813656, 0.0053137895983268457, 9)
    
    Notes
    -----

    Second matrix is used to form the probabilities under the null.
    Marginal sums from first matrix are distributed across these probabilities
    under the null. In other words the observed transitions are taken from T1
    while the expected transitions are formed as:
        .. math::

            E_{i,j} = \sum_j T1_{i,j} * T2_{i,j}/\sum_j T2_{i,j}

    Degrees of freedom corrected for any rows in either T1 or T2 that have
    zero total transitions.
    """
    rs2 = T2.sum(axis=1)
    rs1 = T1.sum(axis=1)
    rs2nz = rs2>0
    rs1nz = rs1>0
    dof1 = sum(rs1nz)
    dof2 = sum(rs2nz)
    rs2 = rs2 + rs2nz
    dof = (dof1-1) * (dof2-1)
    p = np.diag(1/rs2) * np.matrix(T2)
    E = np.diag(rs1) * np.matrix(p)
    num = T1 - E
    num = np.multiply(num, num)
    E = E + (E==0)
    chi2 = num/E
    chi2 = chi2.sum()
    pvalue = 1-stats.chi2.cdf(chi2, dof)
    return chi2, pvalue, dof

class LISA_Markov(Markov):
    """
    Markov for Local Indicators of Spatial Association
    
    Parameters
    ----------

    y  : array (n,t)
         n cross-sectional units observed over t time periods

    w  : weights instance

    permutations : int
                   number of permutations used to determine LISA significance
                   default = 0

    significance_level : float
                         significance level (two-sided) for filtering significant LISA end
                         points in a transition
                         default = 0.05
    Attributes
    ----------
    classes      : array (4, 1)
                   1=HH, 2=LH, 3=LL, 4=HL (own, lag)
    p            : matrix (k, k)
                   transition probility matrix
    steady_state : matrix (k, 1)
                   ergodic distribution
    transitions  : matrix (k, k)
                   count of transitions between each state i and j
    move_types   : matrix (n, t-1)
                   integer values indicating which type of LISA transition
                   occurred (q1 is quadrant in period 1, q2 is quadrant in
                   period 2)
                   q1   q2   move_type
                   1    1    1
                   2    1    2
                   3    1    3
                   4    1    4
                   1    2    5
                   2    2    6
                   3    2    7
                   4    2    8
                   1    3    9
                   2    3    10
                   3    3    11
                   4    3    12
                   1    4    13
                   2    4    14
                   3    4    15
                   4    4    16
    significant_moves    : (if permutations > 0)
                           matrix (n, t-1)
                           integer values indicating the type and significance of a LISA
                           transition. Nonsignificant LISAs are represented as n
                           q1   q2   move_type 
                           1    1    1       
                           2    1    2
                           3    1    3
                           4    1    4
                           n    1    5
                           1    2    6
                           2    2    7
                           3    2    8
                           4    2    9
                           n    2    10
                           1    3    11
                           2    3    12
                           3    3    13
                           4    3    14
                           n    3    15
                           1    4    16
                           2    4    17
                           3    4    18
                           4    4    19
                           n    5    20
                           1    n    21
                           2    n    22
                           3    n    23
                           4    n    24
                           n    n    25

    Examples
    --------
 
    >>> import numpy as np
    >>> f = pysal.open("../examples/usjoin.csv")
    >>> pci = np.array([f.by_col[str(y)] for y in range(1929,2010)]).transpose()
    >>> w = pysal.open("../examples/states48.gal").read()
    >>> lm = LISA_Markov(pci,w)
    >>> lm.classes
    array([1, 2, 3, 4])
    >>> lm.steady_state
    matrix([[ 0.28561505],
            [ 0.14190226],
            [ 0.40493672],
            [ 0.16754598]])
    >>> lm.transitions
    array([[  1.08700000e+03,   4.40000000e+01,   4.00000000e+00,
              3.40000000e+01],
           [  4.10000000e+01,   4.70000000e+02,   3.60000000e+01,
              1.00000000e+00],
           [  5.00000000e+00,   3.40000000e+01,   1.42200000e+03,
              3.90000000e+01],
           [  3.00000000e+01,   1.00000000e+00,   4.00000000e+01,
              5.52000000e+02]])
    >>> lm.p
    matrix([[ 0.92985458,  0.03763901,  0.00342173,  0.02908469],
            [ 0.07481752,  0.85766423,  0.06569343,  0.00182482],
            [ 0.00333333,  0.02266667,  0.948     ,  0.026     ],
            [ 0.04815409,  0.00160514,  0.06420546,  0.88603531]])
    >>> lm.move_types
    array([[11, 11, 11, ..., 11, 11, 11],
           [ 6,  6,  6, ...,  6,  7, 11],
           [11, 11, 11, ..., 11, 11, 11],
           ..., 
           [ 6,  6,  6, ...,  6,  6,  6],
           [ 1,  1,  1, ...,  6,  6,  6],
           [16, 16, 16, ..., 16, 16, 16]])

    Now consider only moves with one, or both, of the LISA end points being
    significant

    >>> np.random.seed(10)
    >>> lm_random = pysal.LISA_Markov(pci, w, permutations=99)
    >>> lm_random.significant_moves[:,0]
    array([13, 25, 13, 25, 25,  1,  1, 25, 13, 25, 25, 25, 25, 25, 25, 13, 25,
           25,  1, 25, 25, 13, 25, 25, 25, 25, 25,  1, 25,  1, 13, 25, 25, 25,
           25, 25,  1, 13, 25, 13, 25, 25, 25, 25, 25, 25, 25, 25])

    """
    def __init__(self, y, w, permutations=0, 
            significance_level=0.05):
        y = y.transpose()
        pml = pysal.Moran_Local

        #################################################################
        # have to optimize conditional spatial permutations over a
        # time series - this is a place holder for the foreclosure paper
        ml = [ pml(yi, w, permutations=permutations) for yi in y]
        #################################################################

        q = np.array([ mli.q for mli in ml]).transpose()
        classes = np.arange(1,5) # no guarantee all 4 quadrants are visited
        Markov.__init__(self, q, classes)
        self.q = q
        n, k = q.shape
        k -= 1
        move_types = np.zeros((n, k), int)
        sm = np.zeros((n, k), int)
        self.significance_level = significance_level
        if permutations > 0:
            p = np.array([ mli.p_z_sim for mli in ml]).transpose()
            self.p = p
            pb = p <= significance_level
        else:
            pb = np.zeros_like(y.T)
        for t in range(k):
            origin = q[:, t]
            dest = q[:, t+1]
            p_origin = pb[:, t]
            p_dest = pb[:, t]
            for r in range(n):
                move_types[r, t] = TT[origin[r], dest[r]]
                key = (origin[r], dest[r], p_origin[r], p_dest[r])
                sm[r, t] = MOVE_TYPES[key]
        if permutations > 0:
            self.significant_moves = sm       
        self.move_types=move_types

def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == '__main__':
    _test()
