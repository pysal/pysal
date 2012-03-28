"""
A library of spatial network kernel density functions.
Not to be used without permission.

Contact: 

Andrew Winslow
GeoDa Center for Geospatial Analysis
Arizona State University
Tempe, AZ
Andrew.Winslow@asu.edu
"""

import operator
import unittest
import test
import priordict as priordict
import network as pynet
from math import exp, sqrt, pi
import time

def triangular(z):
    return 1 - abs(z)

def uniform(z):
    return abs(z)

def quadratic(z):
    return 0.75*(1 - z*z)

def quartic(z):
    return (3.0/pi)*(1-z*z)*(1-z*z)
    #return (15*1.0/16)*(1-z*z)*(1-z*z)

def gaussian(z):
    return sqrt(2*pi)*exp(-0.5*z*z)

def dijkstras_w_prev(G, start, r=1e600):
    D = {}  # dictionary of final distances
    P = {}  # dictionary of previous nodes
    Q = priordict.PriorityDictionary()   # est.dist. of non-final vert.
    Q[start] = 0
    P[start] = None
    for v in Q:
        D[v] = Q[v]
        if v == None or D[v] > r:
            break
        for w in G[v]:
            vwLength = D[v] + G[v][w]
            if w in D:
                pass
            elif w not in Q or vwLength < Q[w]:
                Q[w] = vwLength
                P[w] = v
    return (D, P)

def kernel_density(network, events, bandwidth, orig_nodes, kernel='quadratic'):
    """
    This function estimates Kernel densities on a planar undirected network. 
    It implements the equal-split discontinuous Kernel function developed by Okabe et al. (2009).
    Particularly, it computes Kernel densities by using equation 19 and 20 
    in the paper of Okabe et al. (2009). 

    Parameters
    ----------
    network: A dictionary of dictionaries like {n1:{n2:d12,...},...}
             A planar undirected network
             It is assumed that this network is divided by a certain cell size 
             and is restructured to incorporate the new nodes resulting from the division as well as 
             events. Therefore, nodes in the network can be classified into three groups:
             i) original nodes, 2) event points, and 3) cell points.  
    events: a list of tuples
            a tuple is the network-projected coordinate of an event
            that takes the form of (x,y)
    bandwidth: a float
            Kernel bandwidth
    orig_nodes: a list of tuples
            a tuple is the coordinate of a node that is part of the original base network
            each tuple takes the form of (x,y)
    kernel: string
            the type of Kernel function
            allowed values: 'quadratic', 'gaussian', 'quartic', 'uniform', 'triangular'

    Returns
    -------   
    A dictioinary where keys are node and values are their densities
    Example: {n1:d1,n2:d2,...}

    <tc>#is#kernel_density</tc>
    """

    # beginning of step i
    density = {}
    for n in network:
        density[n] = []
    # end of step i

    # beginning of step ii
    def compute_split_multiplier(prev_D, n):
        '''
        computes the demoninator of the formula 19

        Parameters
        ----------
        prev_D: a dictionary storing pathes from n to other nodes in the network
                its form is like: {n1:prev_node_of_n1(=n2), n2:prev_node_of_n2(=n3),...}
        n: a tuple containing the geographic coordinate of a starting point 
           its form is like: (x,y)

        Returns
        -------
        An integer

        '''
        split_multiplier = 1 
        p = prev_D[n] 
        while p != None:
            if len(network[p]) > 1:
                split_multiplier *= (len(network[p]) - 1)
            if p not in prev_D: 
                p = None
            else:
                p = prev_D[p]
        return split_multiplier
    # end of step ii

    kernel_funcs = {'triangular':triangular, 'uniform': uniform, 
                    'quadratic': quadratic, 'quartic':quartic, 'gaussian':gaussian}
    #t1 = time.time()
    # beginning of step iii
    kernel_func = kernel_funcs[kernel]
    for e in events:
        # beginning of step a
        src_D = pynet.dijkstras(network, e, bandwidth, True)
        # end of step a
        # beginning of step b
        density[e].append(kernel_func(0))
        # end of step b
        # beginning of step c
        for n in src_D[0]: # src_D[0] - a dictionary of nodes whose distance from e is smaller than e 
            if src_D[0][n] == 0: continue
            # src_D[1] - a dictionary from which a path from e to n can be traced
            d = src_D[0][n]
            if d <= bandwidth:
                n_degree = 2.0
                if n in events and n in orig_nodes and len(network[n]) > 0:
                    n_degree = len(network[n])
                unsplit_density = kernel_func(d*1.0/bandwidth*1.0) 
                # src_D[1] - a dictionary from which a path from e to n can be traced
                split_multiplier = compute_split_multiplier(src_D[1], n) 
                density[n].append((1.0/split_multiplier)*(2.0/n_degree)*unsplit_density)
                #if str(n[0]) == '724900.335127' and str(n[1]) == '872127.948935':
                #    print 'event', e
                #    print 'distance', d
                #    print 'unsplit_density', unsplit_density
                #    print 'n_degree', n_degree
                #    print 'split_multiplier', split_multiplier
                #    print 'density', (1.0/split_multiplier)*(2.0/n_degree)*unsplit_density
        # end of step c

    # beginning of step iv 
    #t1 = time.time()
    no_events = len(events)
    for node in density:
        if len(density[node]) > 0:
            #if str(node[0]) == '724900.335127' and str(node[1]) == '872127.948935':
            #    print density[node]
            density[node] = sum(density[node])/no_events
            #density[node] = sum(density[node])*1.0/len(density[node])
        else:
            density[node] = 0.0
    # end of step iv

    #for node in events:
    #    del density[node]

    #print 'normalizing density: %s' % (str(time.time() - t1))

    return density

    



