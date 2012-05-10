#!/usr/env python

"""
A library for computing local K function for network-constrained data

Author:
Andrew Winslow Andrew.Winslow@asu.edu
Myunghwa Hwang mhwang4@gmail.com

"""
import unittest
import numpy as np
import geodanet.network as pynet
import geodanet.kfuncs as pykfuncs
import geodanet.simulator as pysim
import time
import random
import platform                                                                                                                 
try:
    if platform.system() == 'Darwin':
        import multiprocessing
    else:
        multiprocessing = None
except ImportError:
    multiprocessing = None 

class WeightedRandomSampleGenerator(object):
    """
    A generator for randomly sampling n elements from 
    a population group with consideration to a given set of weights
    """

    def __init__(self, weights, population, n):
	"""
	weights: an iterable with m numeric elements
	population: a numpy array with m elements
	n: an integer representing sample size
	"""
        self.totals = np.cumsum(weights)
        self.population = population
        self.n = n
        self.norm = self.totals[-1]

    def next(self):
        sample = []
        for i in xrange(self.n):
            throw = np.random.rand()*self.norm
            sample.append(self.population[np.searchsorted(self.totals, throw)])
        return sample

    def __call__(self):
        return self.next()

class RandomSampleGenerator(object):
    """
    A generator for randomly sampling n elements 
    from a population group
    """
    def __init__(self, population, n):
	"""
	population: a numpy array with m elements
	n: an integer representing sample size
	"""
        self.population = population
        self.n = n

    def next(self):
        return random.sample(self.population, self.n)

    def __call__(self):
        return self.next()

def local_k(network, events, refs, scale_set, cache=None):
    """
    Computes local K function

    network: an undirected network data to which reference points are injected
    refs: a set of reference points on the given network
          points unprojected into the network
    events: a set of event points on the given network
            points projected into the network
    scale_set: a tuple defining spatial scales to be examined
               (min, max, interval)
    """

    node2localK = {}
    net_distances = {}
    if cache: net_distances = cache
    for node in refs:
        node = node[1][0]
        a_dest = network[node].keys()[0]
        node_proj = (node, a_dest, 0, network[node][a_dest])
        if node not in net_distances:
            net_distances[node] = pynet.dijkstras(network, node, scale_set[1])
        if a_dest not in net_distances:
            net_distances[a_dest] = pynet.dijkstras(network, node, scale_set[1])
        distances = pynet.proj_distances_undirected(network, node_proj, events, scale_set[1], cache=net_distances).values()
        node2localK[node] = pykfuncs.kt_values(scale_set, distances, 1)
    return node2localK, net_distances

def cluster_type(obs, lower, upper):
    if obs < lower: return -1
    if obs > upper: return 1
    return 0

def simulate_local_k_01(args):
    sims = args[0]
    n = args[1]
    net_file = args[2]
    network = args[3]
    events = args[4]
    refs = args[5]
    scale_set = args[6]
    cache = args[7]

    #print 'simulated_local_k_01'
    simulator = pysim.Simulation(net_file)
    sims_outcomes = []
    for sim in xrange(sims):
        points = simulator.getRandomPoints(n, projected=True)
        sim_events = []
        for edge in points:
            for point in points[edge]:
                sim_events.append(point)
        res, dists = local_k(network, sim_events, refs, scale_set, cache=cache)
        sims_outcomes.append(res)

    return sims_outcomes

def simulate_local_k_02(args):
    sims = args[0]
    n = args[1]
    refs = args[2]
    scale_set = args[3]
    cache = args[4]

    #print 'simulated_local_k_02'
    sims_outcomes = []
    sampler = RandomSampleGenerator(refs, n).next
    for sim in xrange(sims):
        sim_events = sampler()
        sim_localk = {}
        for node in refs:
            all_distances = cache[node[1][0]]
            distances = []
            for event in sim_events:
                event = event[1][0]
                if event in all_distances:
                    distances.append(all_distances[event]) 
            sim_localk[node[1][0]] = pykfuncs.kt_values(scale_set, distances, 1)
        sims_outcomes.append(sim_localk)
    
    return sims_outcomes

def k_cluster(network, events, refs, scale_set, sims, sig=0.1, sim_network=None, cpus=1):

    """
    Parameters:
    network: a network to which reference points are injected
    events: a set of event points projected into the network
    refs: a set of reference points unprojected into the network
    scale_set: tuple same as (min, max, resolution)
    sims: integer; the number of simulations
    sig: float; siginificance level
    sim_network: the source shape file containing the network data
                 this is used to simualte point patterns for inference 
    cpus: integer: the number of cpus
          multiprocessing can be used for inference
    """

    """
    1. For an observed set of n events on the network, calculate local K function 
    values for all m reference points
    """
    node2localK, net_dists = local_k(network, events, refs, scale_set)
    """
    When n < m (simulator == None):
    2. Select n out of m reference points randomly and 
    calculate local K function values for these randomly sampled points
    When n >= m (simulator != None):
    2. Randomly simulate n points on network edges and
    calculate local K function values for these randomly simulated points
    3. Repeat 2 as many as the number of simulations
    Note: on Darwin systems, simulation will be parallelized
    """
    n = len(events)
    sims_outcomes = []
    if not multiprocessing or cpus == 1:
        if sim_network:
            sims_outcomes = simulate_local_k_01((sims, n, sim_network, network, events, refs, scale_set, net_dists))
        else:
            sims_outcomes = simulate_local_k_02((sims, n, refs, scale_set, net_dists))
    elif multiprocessing and cpus >= 2:
        pool = multiprocessing.Pool(cpus)
        sims_list = range(sims)
        sims_list = map(len, [sims_list[i::cpus] for i in xrange(cpus)])
        partial_outcomes = None
        if sim_network:
             partial_outcomes = pool.map(simulate_local_k_01, 
                         [(sim, n, sim_network, network, events, refs, scale_set, net_dists) for sim in sims_list])
        else:
             partial_outcomes = pool.map(simulate_local_k_02, 
                         [(sim, n, refs, scale_set, net_dists) for sim in sims_list])
        sims_outcomes = partial_outcomes[0]
        for partial in partial_outcomes[1:]:
             sims_outcomes.extend(partial)

    """
    4. Determine lower and upper envelopes for the observed K function values 
       as well as the type of cluster (dispersion or clustering)
    """
    # 4. P-value evaluation
    lower_envelope = {}
    upper_envelope = {}
    lower_p = int(sims*sig/2)
    upper_p = int(sims*(1-sig/2))
    localKs = {}
    for node in refs:
        node = node[1][0]
        lower_envelope[node] = {}
        upper_envelope[node] = {}
        localKs[node] = {}
        for scale in node2localK[node].keys():
            local_outcomes = [sim[node][scale] for sim in sims_outcomes]
            local_outcomes.sort()
            obs = node2localK[node][scale]
            lower = local_outcomes[lower_p]
            upper = local_outcomes[upper_p]
            cluster = cluster_type(obs, lower, upper)
            localKs[node][scale] = [obs, lower, upper, cluster]

    return localKs

