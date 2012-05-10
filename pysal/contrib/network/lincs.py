#!/usr/bin/env python

"""
A library for computing local indicators of network-constrained clusters

Author:
Myunghwa Hwang mhwang4@gmail.com

"""
import unittest
import numpy as np
import scipy.stats as stats
import geodanet.network as pynet
import pysal, copy
import time

def unconditional_sim(event, base, s): 
    """ 
    Parameters:
        event: n*1 numpy array with integer values
              observed values for an event variable
        base: n*1 numpy array with integer values
              observed values for a population variable
        s: integer
              the number of simulations

    Returns:
            : n*s numpy array
    """
    mean_risk = event.sum()*1.0/base.sum()
    if base.dtype != int:
        base = np.array([int(v) for v in base])
    base_zeros = (base == 0.0)
    base[base_zeros] += 1.0
    sims = np.random.binomial(base, mean_risk, (s, len(event))).transpose()
    sims[base_zeros, :] = 0.0
    return sims

def unconditional_sim_poisson(event, base, s): 
    """ 
    Parameters:
        event: n*1 numpy array with integer values
              observed values for an event variable
        base: n*1 numpy array with integer values
              observed values for a population variable
        s: integer
              the number of simulations

    Returns:
            : n*s numpy array
    """
    mean_risk = event.sum()*1.0/base.sum()
    E = base*mean_risk
    return np.random.poisson(E, (s, len(event))).transpose()

def conditional_multinomial(event, base, s): 
    """ 
    Parameters:
        event: n*1 numpy array with integer values
              observed values for an event variable
        base: n*1 numpy array with integer values
              observed values for a population variable
        s: integer
              the number of simulations

    Returns:
            : n*s numpy array
    """
    m = int(event.sum())
    props = base*1.0/base.sum()
    return np.random.multinomial(m, props, s).transpose()

def pseudo_pvalues(obs, sims):
    """
    Get pseudo p-values from a set of observed indices and their simulated ones.

    Parameters:
        obs: n*1 numpy array for observed values
        sims: n*sims numpy array; sims is the number of simulations

    Returns:
        p_sim : n*1 numpy array for pseudo p-values
        E_sim : mean of p_sim
        SE_sim: standard deviation of p_sim
        V_sim: variance of p_sim
        z_sim: standardarized observed values
        p_z_sim: p-value of z_sim based on normal distribution   
    """

    sims = np.transpose(sims)
    permutations = sims.shape[0]
    above = sims >= obs
    larger = sum(above)
    low_extreme = (permutations - larger) < larger
    larger[low_extreme] = permutations - larger[low_extreme]
    p_sim = (larger + 1.0)/(permutations + 1.0)
    E_sim = sims.mean()
    SE_sim = sims.std()
    V_sim = SE_sim*SE_sim
    z_sim = (obs - E_sim)/SE_sim
    p_z_sim = 1 - stats.norm.cdf(np.abs(z_sim))
    return p_sim, E_sim, SE_sim, V_sim, z_sim, p_z_sim 

def node_weights(network, attribute=False):
    """
    Obtains a spatial weights matrix of edges in a network
    if two edges share a node, they are neighbors

    Parameters:
        network: a network with/without attributes
        attribute: boolean
                   if true, attributes of edges are added to a dictionary of edges,
                   which is a return value

    Returns:
        w: a spatial weights instance
        id2link: an associative dictionary that connects a sequential id to a unique 
                 edge on the network
                 if attribute is true, each item in the dictionary includes the attributes

    """
    link2id, id2link = {}, {}
    counter = 0 
    neighbors, weights = {},{}
    for n1 in network:
        for n2 in network[n1]:
            if (n1,n2) not in link2id or link2id[(n1,n2)] not in neighbors:
                if (n1,n2) not in link2id:
                    link2id[(n1,n2)] = counter
                    link2id[(n2,n1)] = counter
                    if not attribute:
                        id2link[counter] = (n1, n2) 
                    else:
                        id2link[counter] = tuple([(n1,n2)] + list(network[n1][n2][1:]))
                    counter += 1
                neighbors_from_n1 = [(n1, n) for n in network[n1] if n != n2] 
                neighbors_from_n2 = [(n2, n) for n in network[n2] if n != n1] 
                neighbors_all = neighbors_from_n1 + neighbors_from_n2
                neighbor_ids = []
                for edge in neighbors_all:
                    if edge not in link2id:
                        link2id[edge] = counter
                        link2id[(edge[-1], edge[0])] = counter
                        if not attribute:
                            id2link[counter] = edge
                        else:
                            id2link[counter] = tuple([edge] + list(network[edge[0]][edge[1]][1:]))
                        neighbor_ids.append(counter)    
                        counter += 1
                    else:
                        neighbor_ids.append(link2id[edge])
                neighbors[link2id[(n1,n2)]] = neighbor_ids
                weights[link2id[(n1,n2)]] = [1.0]*(len(neighbors_from_n1) + len(neighbors_from_n2))
    return pysal.weights.W(neighbors, weights), id2link 

def edgepoints_from_network(network, attribute=False):
    """
    Obtains a list of projected points which are midpoints of edges
    
    Parameters:
        network: a network with/without attributes
        attribute: boolean
                   if true, one of return values includes attributes for each edge

    Returns:
        id2linkpoints: a dictionary that associates a sequential id to a projected, midpoint of each edge
        id2attr: a dictionary that associates a sequential id to the attributes of each edge
        link2id: a dictionary that associates each edge to its id
    """
    link2id, id2linkpoints, id2attr = {}, {}, {}
    counter = 0
    for n1 in network:
        for n2 in network[n1]:
            if (n1,n2) not in link2id or (n2,n1) not in link2id:
                link2id[(n1,n2)] = counter
                link2id[(n2,n1)] = counter
                if type(network[n1][n2]) != list:
                    half_dist = network[n1][n2]/2 
                else:
                    half_dist = network[n1][n2][0]/2 
                if n1[0] < n2[0] or (n1[0] == n2[0] and n1[1] < n2[1]):
                    id2linkpoints[counter] = (n1,n2,half_dist,half_dist)
                else:
                    id2linkpoints[counter] = (n2,n1,half_dist,half_dist)
                if attribute:
                    id2attr[counter] = network[n1][n2][1:]
                counter += 1
    return id2linkpoints, id2attr, link2id

def dist_weights(network, id2linkpoints, link2id, bandwidth):
    """
    Obtains a distance-based spatial weights matrix using network distance

    Parameters:
        network: an undirected network without additional attributes 
        id2linkpoints: a dictionary that includes a list of network-projected, midpoints of edges in the network
        link2id: a dictionary that associates each edge to a unique id
        bandwidth: a threshold distance for creating a spatial weights matrix

    Returns:
        w : a distance-based, binary spatial weights matrix
        id2link: a dictionary that associates a unique id to each edge of the network
    """
    linkpoints = id2linkpoints.values()
    neighbors, id2link = {}, {}
    net_distances = {}
    for linkpoint in id2linkpoints:
        if linkpoints[linkpoint] not in net_distances:
            net_distances[linkpoints[linkpoint][0]] = pynet.dijkstras(network, linkpoints[linkpoint][0], r=bandwidth)
            net_distances[linkpoints[linkpoint][1]] = pynet.dijkstras(network, linkpoints[linkpoint][1], r=bandwidth)
        ngh = pynet.proj_distances_undirected(network, linkpoints[linkpoint], linkpoints, r=bandwidth, cache=net_distances)
        #ngh = pynet.proj_distances_undirected(network, linkpoints[linkpoint], linkpoints, r=bandwidth)
        if linkpoints[linkpoint] in ngh:
            del ngh[linkpoints[linkpoint]]
        if linkpoint not in neighbors:
            neighbors[linkpoint] = []
        for k in ngh.keys():
            neighbor = link2id[k[:2]]
            if neighbor not in neighbors[linkpoint]:
                neighbors[linkpoint].append(neighbor)
            if neighbor not in neighbors:
                neighbors[neighbor] = []
            if linkpoint not in neighbors[neighbor]:
                neighbors[neighbor].append(linkpoint)
        id2link[linkpoint] = id2linkpoints[linkpoint][:2]
    weights = copy.copy(neighbors)
    for ngh in weights:
        weights[ngh] = [1.0]*len(weights[ngh])
    return pysal.weights.W(neighbors, weights), id2link


def lincs(network, event, base, weight, dist=None, lisa_func='moran', sim_method="permutations", sim_num=99):
    """
    Compute local Moran's I for edges in the network

    Parameters:
        network: a clean network where each edge has up to three attributes:
                 Its length, an event variable, and a base variable
        event: integer
               an index for the event variable 
        base: integer 
              an index for the base variable
        weight: string
                type of binary spatial weights
                two options are allowed: Node-based, Distance-based
        dist: float
              threshold distance value for the distance-based weight
        lisa_func: string
                   type of LISA functions
                   three options allowed: moran, g, and g_star
        sim_method: string
                    type of simulation methods
                    four options allowed: permutations, binomial (unconditional),
                    poisson (unconditional), multinomial (conditional)
        sim_num: integer
                 the number of simulations

    Returns:
               : a dictionary of edges
                 an edge and its moran's I are the key and item
               : a Weights object
                 PySAL spatial weights object

    """
    if lisa_func in ['g', 'g_star'] and weight == 'Node-based':
        print 'Local G statistics can work only with distance-based weights matrix'
        raise 

    if lisa_func == 'moran':
        lisa_func = pysal.esda.moran.Moran_Local
    else:
        lisa_func = pysal.esda.getisord.G_Local

    star = False
    if lisa_func == 'g_star':
        star = True    

    if base:
        def getBase(edges, edge, base):
            return edges[edge][base]
    else:
        def getBase(edges, edge, base):
            return 1.0
    w, edges, e, b, edges_geom = None, None, None, None, []
    if weight == 'Node-based':
        w, edges = node_weights(network, attribute=True)
	n = len(edges)
	e, b = np.zeros(n), np.zeros(n)
	for edge in edges:
            edges_geom.append(edges[edge][0])
	    e[edge] = edges[edge][event]
            b[edge] = getBase(edges, edge, base)
        w.id_order = edges.keys()
    elif dist is not None:
        id2edgepoints, id2attr, edge2id = edgepoints_from_network(network, attribute=True)
        for n1 in network:
            for n2 in network[n1]:
                network[n1][n2] = network[n1][n2][0]
        w, edges = dist_weights(network, id2edgepoints, edge2id, dist)
        n = len(id2attr)
	e, b = np.zeros(n), np.zeros(n)
        if base:
            base -= 1
	for edge in id2attr:
            edges_geom.append(edges[edge])
	    e[edge] = id2attr[edge][event - 1]
            b[edge] = getBase(id2attr, edge, base)
        w.id_order = id2attr.keys()

    Is, p_sim, Zs = None,None, None
    if sim_method == 'permutation':
        if lisa_func == pysal.esda.moran.Moran_Local:
	    lisa_i = lisa_func(e*1.0/b,w,transformation="r",permutations=sim_num)
            Is = lisa_i.Is
            Zs = lisa_i.q
        else:
	    lisa_i = lisa_func(e*1.0/b,w,transform="R",permutations=sim_num,star=star)
            Is = lisa_i.Gs
            Zs = lisa_i.Zs
        p_sim = lisa_i.p_sim
    else:
	sims = None
        if lisa_func == pysal.esda.moran.Moran_Local:
	    lisa_i = lisa_func(e*1.0/b,w,transformation="r",permutations=0)
            Is = lisa_i.Is
            Zs = lisa_i.q
        else:
	    lisa_i = lisa_func(e*1.0/b,w,transform="R",permutations=0,star=star)
	    Is = lisa_i.Gs
	    Zs = lisa_i.Zs
	if sim_method == 'binomial':
	    sims = unconditional_sim(e, b, sim_num)
	elif sim_method == 'poisson':
	    sims = unconditional_sim_poisson(e, b, sim_num)
	else:
	    sims = conditional_multinomial(e, b, sim_num)
        if lisa_func == pysal.esda.moran.Moran_Local:
	    for i in range(sim_num):
		sims[:,i] = lisa_func(sims[:,i]*1.0/b,w,transformation="r",permutations=0).Is
        else:
	    for i in range(sim_num):
		sims[:,i] = lisa_func(sims[:,i]*1.0/b,w,permutations=0,star=star).Gs
	sim_res = pseudo_pvalues(Is, sims)
	p_sim = sim_res[0]

    w.transform = 'O'
    return zip(edges_geom, e, b, Is, Zs, p_sim), w

        
