import numpy as np


class NetworkBase(object):
    """Base object for performing network analysis on a spaghetti.Network
    object.
    
    Parameters
    ----------
    
    ntw : spaghetti.Network
        spaghetti Network object.
    
    pointpattern : spaghetti.network.PointPattern
        A spaghetti point pattern object.
    
    nsteps : int
        The number of steps at which the count of the nearest neighbors
        is computed.
    
    permutations : int
        The number of permutations to perform (default 99).
    
    threshold : float
        The level at which significance is computed.
        -- 0.5 would be 97.5% and 2.5%
    
    distribution : str
        The distribution from which random points are sampled
        -- uniform or poisson
    
    lowerbound : float
        The lower bound at which the function is computed. (Default 0).
    
    upperbound : float
        The upper bound at which the function is computed. Defaults to
        the maximum observed nearest neighbor distance.
    
    Attributes
    ----------
    
    sim : numpy.ndarray
        simulated distance matrix
    
    npts : int
        pointpattern.npoints
    
    xaxis : numpy.ndarray
        observed x-axis of values
    
    observed : numpy.ndarray
        observed y-axis of values
    
    """
    def __init__(self, ntw, pointpattern, nsteps=10, permutations=99,
                 threshold=0.5, distribution='poisson',
                 lowerbound=None, upperbound=None):
        self.ntw = ntw
        self.pointpattern = pointpattern
        self.nsteps = nsteps
        self.permutations = permutations
        self.threshold = threshold

        self.distribution = distribution
        self.validatedistribution()

        self.sim = np.empty((permutations, nsteps))
        self.npts = self.pointpattern.npoints

        self.lowerbound = lowerbound
        self.upperbound = upperbound

        # Compute Statistic.
        self.computeobserved()
        self.computepermutations()

        # Compute the envelope vectors.
        self.computeenvelope()


    def validatedistribution(self):
        """enusure statistical distribution is supported
        """
        valid_distributions = ['uniform', 'poisson']
        assert(self.distribution in valid_distributions),\
               "Distribution not in {}".format(valid_distributions)


    def computeenvelope(self):
        """compute upper and lower bounds of envelope
        """
        upper = 1.0 - self.threshold / 2.0
        lower = self.threshold / 2.0

        self.upperenvelope = np.nanmax(self.sim, axis=0) * upper
        self.lowerenvelope = np.nanmin(self.sim, axis=0) * lower


    def setbounds(self, nearest):
        """set upper and lower bounds
        """
        if self.lowerbound is None:
            self.lowerbound = 0
        if self.upperbound is None:
            self.upperbound = np.nanmax(nearest)


class NetworkG(NetworkBase):
    """Compute a network constrained G statistic. This requires the capability
    to compute a distance matrix between two point patterns. In this case one
    will be observed and one will be simulated.
    """
    def computeobserved(self):
        """compute the observed nearest
        """
        nearest = np.nanmin(self.ntw.allneighbordistances(self.pointpattern),
                            axis=1)
        self.setbounds(nearest)
        observedx, observedy = gfunction(nearest, self.lowerbound,
                                         self.upperbound, nsteps=self.nsteps)
        self.observed = observedy
        self.xaxis = observedx


    def computepermutations(self):
        """compute permutations of the nearest
        """
        for p in range(self.permutations):
            sim = self.ntw.simulate_observations(self.npts,
                                                distribution=self.distribution)
            nearest = np.nanmin(self.ntw.allneighbordistances(sim), axis=1)

            simx, simy = gfunction(nearest,
                                   self.lowerbound,
                                   self.upperbound,
                                   nsteps=self.nsteps)
            self.sim[p] = simy


class NetworkK(NetworkBase):
    """Compute a network constrained K statistic. This requires the capability
    to compute a distance matrix between two point patterns. In this case one
    will be observed and one will be simulated.
    
    Attributes
    ----------
    
    lam : float
        lambda value
    
    Notes
    -----
    
    Based on :cite:`Okabe2001`.
    
    """
    def computeobserved(self):
        """compute the observed nearest
        """
        nearest = self.ntw.allneighbordistances(self.pointpattern)
        self.setbounds(nearest)
        
        self.lam = self.npts / sum(self.ntw.edge_lengths.values())
        observedx, observedy = kfunction(nearest,
                                         self.upperbound,
                                         self.lam,
                                         nsteps=self.nsteps)
        self.observed = observedy
        self.xaxis = observedx


    def computepermutations(self):
        """compute permutations of the nearest
        """
        for p in range(self.permutations):
            sim = self.ntw.simulate_observations(self.npts,
                                                distribution=self.distribution)
            nearest = self.ntw.allneighbordistances(sim)
            
            simx, simy = kfunction(nearest,
                                   self.upperbound,
                                   self.lam,
                                   nsteps=self.nsteps)
            self.sim[p] = simy


class NetworkF(NetworkBase):
    """Compute a network constrained F statistic. This requires the capability
    to compute a distance matrix between two point patterns. In this case one
    will be observed and one will be simulated.
    
    Attributes
    ----------
    
    fsim : spaghetti.network.SimulatedPointPattern
        simulated point pattern of `self.npts` points
    
    """
    
    
    def computeobserved(self):
        """compute the observed nearest and simulated nearest
        """
        self.fsim = self.ntw.simulate_observations(self.npts)
        # Compute nearest neighbor distances from the simulated to the
        # observed.
        nearest = np.nanmin(self.ntw.allneighbordistances(self.fsim,
                                                          self.pointpattern),
                                                          axis=1)
        self.setbounds(nearest)
        # Generate a random distribution of points.
        observedx, observedy = ffunction(nearest, self.lowerbound,
                                         self.upperbound, nsteps=self.nsteps,
                                         npts=self.npts)
        self.observed = observedy
        self.xaxis = observedx


    def computepermutations(self):
        """compute permutations of the nearest
        """
        for p in range(self.permutations):
            sim = self.ntw.simulate_observations(self.npts,
                                                distribution=self.distribution)
            nearest = np.nanmin(self.ntw.allneighbordistances(sim, self.fsim),
                                axis=1)
            simx, simy = ffunction(nearest, self.lowerbound, self.upperbound,
                                   self.npts, nsteps=self.nsteps)
            self.sim[p] = simy


def gfunction(nearest, lowerbound, upperbound, nsteps=10):
    """Compute a G-Function

    Parameters
    ----------
    
    nearest : numpy.ndarray
        A vector of nearest neighbor distances.
    
    lowerbound : int or float
        The starting value of the sequence.
    
    upperbound : int or float
        The end value of the sequence.
    
    nsteps : int
        The number of distance bands. Default is 10. Must be non-negative.
    
    Returns
    -------
    
    x : numpy.ndarray
        x-axis of values
    
    y : numpy.ndarray
        y-axis of values
    
    """
    nobs = len(nearest)
    x = np.linspace(lowerbound, upperbound, nsteps)
    nearest = np.sort(nearest)
    
    y = np.empty(len(x))
    for i, r in enumerate(x):
        cnt = len(nearest[nearest <= r])
        if cnt > 0:
            g = cnt / float(nobs)
        else:
            g = 0
        y[i] = g
    return x, y


def kfunction(nearest, upperbound, intensity, nsteps=10):
    """Compute a K-Function

    Parameters
    ----------
    
    nearest : numpy.ndarray
        A vector of nearest neighbor distances.
    
    upperbound : int or float
        The end value of the sequence.
    
    intensity : float
        lambda value
    
    nsteps : int
        The number of distance bands. Default is 10. Must be non-negative.
    
    Returns
    -------
    
    x : numpy.ndarray
        x-axis of values
    
    y : numpy.ndarray
        y-axis of values
    
    """
    nobs = len(nearest)
    x = np.linspace(0, upperbound, nsteps)
    y = np.empty(len(x))
    
    for i, s in enumerate(x):
        y[i] = len(nearest[nearest <= s])
    y *= (intensity ** -1)
    return x, y


def ffunction(nearest, lowerbound, upperbound, npts, nsteps=10):
    """Compute an F-Function

    Parameters
    ----------
    
    nearest : numpy.ndarray
        A vector of nearest neighbor distances.
    
    lowerbound : int or float
        The starting value of the sequence.
    
    upperbound : int or float
        The end value of the sequence.
    
    npts : int
        pointpattern.npoints
    
    nsteps : int
        The number of distance bands. Default is 10. Must be non-negative.
    
    Returns
    -------
    
    x : numpy.ndarray
        x-axis of values
    
    y : numpy.ndarray
        y-axis of values
    
    """
    nobs = len(nearest)
    x = np.linspace(lowerbound, upperbound, nsteps)
    nearest = np.sort(nearest)
    y = np.empty(len(x))
    for i, r in enumerate(x):
        cnt = len(nearest[nearest <= r])
        if cnt > 0:
            g = cnt / float(npts)
        else:
            g = 0
        y[i] = g
    return x, y
