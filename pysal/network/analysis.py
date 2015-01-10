import numpy as np


class NetworkBase(object):
    def __init__(self, ntw, pointpattern,nsteps=10,
                 permutations=99, threshold=0.5,
                 distirbution='poisson',
                 lowerbound=None,upperbound=None):

        self.ntw = ntw
        self.pointpattern = pointpattern
        self.nsteps = nsteps
        self.permutations = permutations
        self.threshold = threshold

        self.distirbution = distirbution
        self.validatedistribution()

        self.sim = np.empty((permutations, nsteps))
        self.npts = self.pointpattern.npoints

        self.lowerbound = lowerbound
        self.upperbound = upperbound

        #Compute Statistic
        self.computeobserved()
        self.computepermutations()

        #Compute the envelope vectors
        self.computeenvelope()

    def validatedistribution(self):
        valid_distributions = ['uniform', 'poisson']
        assert(self.distirbution in valid_distributions),"Disstribution not in {}".format(valid_distributions)

    def computeenvelope(self):
        upper = 1.0 - self.threshold / 2.0
        lower = self.threshold / 2.0

        self.upperenvelope = np.nanmax(self.sim, axis=0) * upper
        self.lowerenvelope = np.nanmin(self.sim, axis=0) * lower

    def setbounds(self, nearest):
        if self.lowerbound == None:
            self.lowerbound = np.nanmin(nearest)
        if self.upperbound == None:
            self.upperbound = np.nanmax(nearest)

class NetworkG(NetworkBase):
    """
    Compute a network constrained G statistic

    Attributes
    ==========

    """

    def computeobserved(self):
        nearest = np.nanmin(self.ntw.allneighbordistances(self.pointpattern), axis=1)
        self.setbounds(nearest)
        observedx, observedy = gfunction(nearest,self.lowerbound,self.upperbound,
                                     nsteps=self.nsteps)
        self.observed = observedy
        self.xaxis = observedx

    def computepermutations(self):
        for p in xrange(self.permutations):
            sim = self.ntw.simulate_observations(self.npts,
                                                 distribution=self.distirbution)
            nearest = np.nanmin(self.ntw.allneighbordistances(sim), axis=1)

            simx, simy = gfunction(nearest,
                                   self.lowerbound,
                                   self.upperbound,
                                   nsteps=self.nsteps)
            self.sim[p] = simy


class NetworkK(NetworkBase):
    """
    Network constrained K Function
    """

    def computeobserved(self):
        nearest = self.ntw.allneighbordistances(self.pointpattern)
        self.setbounds(nearest)

        self.lam = self.npts / np.sum(np.array(self.ntw.edge_lengths.values()))
        observedx, observedy = kfunction(nearest,
                                         self.upperbound,
                                         self.lam,
                                         nsteps = self.nsteps)
        self.observed = observedy
        self.xaxis = observedx

    def computepermutations(self):
        for p in xrange(self.permutations):
            sim = self.ntw.simulate_observations(self.npts,
                                                 distribution=self.distirbution)
            nearest = self.ntw.allneighbordistances(sim)

            simx, simy = kfunction(nearest,
                                   self.upperbound,
                                   self.lam,
                                   nsteps=self.nsteps)
            self.sim[p] = simy


def kfunction(nearest, upperbound, intensity, nsteps=10):
    nobs = len(nearest)
    x = np.linspace(0, upperbound, nsteps)
    y = np.empty(len(x))

    for i, s in enumerate(x):
        y[i] = len(nearest[nearest <= s])
    y *= (intensity ** -1)
    return x, y


def gfunction(nearest, lowerbound, upperbound, nsteps = 10):
    """
    Compute a G-Function

    Parameters
    ----------
    nearest         ndarray A vector of nearest neighbor distances
    nsteps          int The number of distance bands
    permutations    int The number of permutations to perform
    threshold       int Upper and lower significance threshold
    envelope        bool Return results of all permutations
    poisson         bool Use a poisson distribution to
                         determine the number of points
    """
    nobs = len(nearest)
    x = np.linspace(lowerbound, upperbound, nsteps)
    nearest = np.sort(nearest)

    y = np.empty(len(x))
    for i,r in enumerate(x):
        cnt = len(nearest[nearest <= r])
        if cnt > 0:
            g = cnt / float(nobs)
        else:
            g = 0
        y[i] = g
    return x, y
