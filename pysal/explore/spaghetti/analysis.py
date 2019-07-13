import numpy as np


class NetworkBase(object):
    """Base object for performing network analysis on a
    ``spaghetti.Network`` object.
    
    Parameters
    ----------
    
    ntw : spaghetti.Network
        spaghetti Network object.
    
    pointpattern : spaghetti.network.PointPattern
        A spaghetti point pattern object.
    
    nsteps : int
            The number of steps at which the count of the nearest
            neighbors is computed.
        
    permutations : int
        The number of permutations to perform. Default 99.
    
    threshold : float
        The level at which significance is computed.
        (0.5 would be 97.5% and 2.5%).
    
    distribution : str
        The distribution from which random points are sampled
        Either ``"uniform"`` or ``"poisson"``.
    
    lowerbound : float
        The lower bound at which the G-function is computed.
        Default 0.
    
    upperbound : float
        The upper bound at which the G-function is computed.
        Defaults to the maximum observed nearest neighbor distance.
    
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
        
        # set initial class attributes
        self.ntw = ntw
        self.pointpattern = pointpattern
        self.nsteps = nsteps
        self.permutations = permutations
        self.threshold = threshold
        
        # set and validate the distribution
        self.distribution = distribution
        self.validatedistribution()
        
        # create an empty array to store the simulated points
        self.sim = np.empty((permutations, nsteps))
        self.npts = self.pointpattern.npoints
        
        # set the lower and upper bounds (lower only for G)
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        
        # compute the statistic (F, G, or K)
        self.computeobserved()
        self.computepermutations()

        # compute the envelope vectors
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
    """Compute a network constrained G statistic. This requires the
    capability to compute a distance matrix between two point patterns.
    In this case one will be observed and one will be simulated.
    """
    
    def computeobserved(self):
        """compute the observed nearest
        """
        
        # find nearest point that is not NaN
        nearest = np.nanmin(self.ntw.allneighbordistances(self.pointpattern),
                            axis=1)
        self.setbounds(nearest)
        
        # compute a G-Function
        observedx, observedy = gfunction(nearest,
                                         self.lowerbound,
                                         self.upperbound,
                                         nsteps=self.nsteps)
        
        # set observed values
        self.observed = observedy
        self.xaxis = observedx


    def computepermutations(self):
        """compute permutations of the nearest
        """
        
        # for each round of permutations
        for p in range(self.permutations):
            
            # simulate a point pattern
            sim = self.ntw.simulate_observations(\
                                                self.npts,
                                                distribution=self.distribution)
            
            # find nearest observation
            nearest = np.nanmin(self.ntw.allneighbordistances(sim),
                                axis=1)
            
            # compute a G-Function
            simx, simy = gfunction(nearest,
                                   self.lowerbound,
                                   self.upperbound,
                                   nsteps=self.nsteps)
            
            # label the permutation
            self.sim[p] = simy


class NetworkK(NetworkBase):
    """Compute a network constrained K statistic. This requires the
    capability to compute a distance matrix between two point patterns.
    In this case one will be observed and one will be simulated.
    
    Attributes
    ----------
    
    lam : float
        ``lambda`` value
    
    Notes
    -----
    
    Based on :cite:`Okabe2001`.
    
    """
    
    def computeobserved(self):
        """compute the observed nearest
        """
        
        # find nearest point that is not NaN
        nearest = self.ntw.allneighbordistances(self.pointpattern)
        self.setbounds(nearest)
        
        # set the intensity (lambda)
        self.lam = self.npts / sum(self.ntw.arc_lengths.values())
        
        # compute a K-Function
        observedx, observedy = kfunction(nearest,
                                         self.upperbound,
                                         self.lam,
                                         nsteps=self.nsteps)
        
        # set observed values
        self.observed = observedy
        self.xaxis = observedx


    def computepermutations(self):
        """compute permutations of the nearest
        """
        
        # for each round of permutations
        for p in range(self.permutations):
            
            # simulate a point pattern
            sim = self.ntw.simulate_observations(\
                                                self.npts,
                                                distribution=self.distribution)
            
            # find nearest observation
            nearest = self.ntw.allneighbordistances(sim)
            
            # compute a K-Function
            simx, simy = kfunction(nearest,
                                   self.upperbound,
                                   self.lam,
                                   nsteps=self.nsteps)
            
            # label the permutation
            self.sim[p] = simy


class NetworkF(NetworkBase):
    """Compute a network constrained F statistic. This requires the
    capability to compute a distance matrix between two point patterns.
    In this case one will be observed and one will be simulated.
    
    Attributes
    ----------
    
    fsim : spaghetti.network.SimulatedPointPattern
        simulated point pattern of ``self.nptsv points
    
    """
    
    def computeobserved(self):
        """compute the observed nearest and simulated nearest
        """
        
        # create an initial simulated point pattern
        self.fsim = self.ntw.simulate_observations(self.npts)
        
        # find nearest neighbor distances from
        # the simulated to the observed
        nearest = np.nanmin(self.ntw.allneighbordistances(self.fsim,
                                                          self.pointpattern),
                                                          axis=1)
        self.setbounds(nearest)
        
        # compute an F-function
        observedx, observedy = ffunction(nearest,
                                         self.lowerbound,
                                         self.upperbound,
                                         nsteps=self.nsteps,
                                         npts=self.npts)
        
        # set observed values
        self.observed = observedy
        self.xaxis = observedx


    def computepermutations(self):
        """compute permutations of the nearest
        """
        
        # for each round of permutations
        for p in range(self.permutations):
            
            # simulate a point pattern
            sim = self.ntw.simulate_observations(\
                                                self.npts,
                                                distribution=self.distribution)
            
            # find nearest observation
            nearest = np.nanmin(self.ntw.allneighbordistances(sim, self.fsim),
                                axis=1)
            
            # compute an F-function
            simx, simy = ffunction(nearest,
                                   self.lowerbound,
                                   self.upperbound,
                                   self.npts,
                                   nsteps=self.nsteps)
            
            # label the permutation
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
        The number of distance bands. Default is 10. Must be
        non-negative.
    
    Returns
    -------
    
    x : numpy.ndarray
        x-axis of values
    
    y : numpy.ndarray
        y-axis of values
    
    """
    
    # set observation count
    nobs = len(nearest)
    
    # create interval for x-axis
    x = np.linspace(lowerbound,
                    upperbound,
                    nsteps)
    
    # sort nearest neighbor distances
    nearest = np.sort(nearest)
    
    # create empty y-axis vector
    y = np.empty(len(x))
    
    # iterate over x-axis interval
    for i, r in enumerate(x):
        
        # slice out and count neighbors within radius
        cnt = len(nearest[nearest <= r])
        
        # if there is one or more neighbors compute `g`
        if cnt > 0:
            g = cnt / float(nobs)
        # otherwise set `g` to zero
        else:
            g = 0
        
        # label `g` on the y-axis
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
        The number of distance bands. Default is 10. Must be
        non-negative.
    
    Returns
    -------
    
    x : numpy.ndarray
        x-axis of values
    
    y : numpy.ndarray
        y-axis of values
    
    """
    
    # set observation count
    nobs = len(nearest)
    
    # create interval for x-axis
    x = np.linspace(0,
                    upperbound,
                    nsteps)
    
    # create empty y-axis vector
    y = np.empty(len(x))
    
    # iterate over x-axis interval
    for i, r in enumerate(x):
        
        # slice out and count neighbors within radius
        y[i] = len(nearest[nearest <= r])
    
    # compute k for y-axis vector
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
        The number of distance bands. Default is 10. Must be
        non-negative.
    
    Returns
    -------
    
    x : numpy.ndarray
        x-axis of values
    
    y : numpy.ndarray
        y-axis of values
    
    """
    
    # set observation count
    nobs = len(nearest)
    
    # create interval for x-axis
    x = np.linspace(lowerbound,
                    upperbound,
                    nsteps)
    
    # sort nearest neighbor distances
    nearest = np.sort(nearest)
    
    # create empty y-axis vector
    y = np.empty(len(x))
    
    # iterate over x-axis interval
    for i, r in enumerate(x):
        
        # slice out and count neighbors within radius
        cnt = len(nearest[nearest <= r])
        
        # if there is one or more neighbors compute `f`
        if cnt > 0:
            f = cnt / float(npts)
        # otherwise set `f` to zero
        else:
            f = 0
       
        # label `f` on the y-axis
        y[i] = f
    
    
    return x, y
