# GWR Kernel function specification
__author__ = "Qunshan Zhao qszhao@asu.edu"

from math import sqrt, exp
import numpy as np

#--0. Distance calculation--------------------------------------------------------------------

# Eucliean distance
def get_EucDist(pt1, pt2):
    """
    Get Euclidean distance
    
    Arguments:
        pt1: list/tuple, (x,y) coordinate of point 1
        pt2: list/tuple, (x,y) coordinate of point 2
                    
    Return:
        dist: double, Euclidean distance 
    """
    
    return sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

# Great circle distance
def get_latlongDist(pt1, pt2):
    """
    Get spherical distance
    
    Arguments:
        pt1: list/tuple, (x,y) coordinate of point 1
        pt2: list/tuple, (x,y) coordinate of point 2
                    
    Return:
        dist: double, spherical distance 
    """
    lat1 = pt1[1]
    lon1 = pt1[0]
    lat2 = pt2[1]
    lon2 = pt2[0]
    
    lat1 *= np.pi / 180
    lat2 *= np.pi / 180
    lon1 *= np.pi / 180
    lon2 *= np.pi / 180
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    work = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    #raddelta = 2.0 * np.arctan2( np.sqrt(work), np.sqrt(1.0-work) )
    rad = 2.0 * np.arcsin(np.sqrt(work))
    
    dis = 6371.009 * rad #raddelta
    
    return dis
    

#--1. Kernel specification--------------------------------------------------------------------

# Fixed Gaussian kernel (distance). 

def fix_Gaussian(dist, band):
    """
    Fixed Gaussian (distance)
    Methods: p56, (2.24), Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        dist: dictionary, dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n 
        band: double, bandwidth
            
    Return:
        weit: dictionary, n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij,        
    """
    w = {}
    npt = len(dist.keys())
    nval = len(dist[0])
    for i in range(npt):
        w[i] = []
        for j in range(nval):                
            w[i].append(exp(-0.5*(dist[i][j]/band)**2)) 
    return w

# Adaptive Gaussian kernel (nearest neighbors).
 
def adap_Gaussian(dist, nn):
    """
    Adaptive Gaussian (NN)
    
    Methods: p58, Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        dist: dictionary, dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n
        nn: Nth nearest neighbors of point i.
            
    Return:
        weit: dictionary, n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij,      
    """
    w = {}
    npt = len(dist.keys())
    nval = len(dist[0])
    nn = int(nn)
    
    for i in range(npt):
        w[i] = []
        
        lstDist = []
        for key, val in dist[i].items():
            lstDist.append((key, val))
        types = [('ptID', float), ('Distance', float)]
        arrDist = np.array(lstDist, dtype=types)
        arrDistSort = np.sort(arrDist, order='Distance') # sort based on distance        
        
        d_0 = arrDistSort[nn-1][1] # get distance threshold based on Nth neighbor
        for j in range(nval): # N includes point i self
            w[i].append(exp(-0.5*(dist[i][j]/d_0)**2))               
    return w    
    
# Fixed Bisquare kernel (distance). 

def fix_Bisquare(dist, band):
    """
    Fixed bi-square (distance)
    
    Methods: p57, (2.25), Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        dist: dictionary, dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n 
        band: double, bandwidth
            
    Return:
        weit: dictionary, n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij,       
    """
    w = {}
    npt = len(dist.keys())
    nval = len(dist[0])
    for i in range(npt):
        w[i] = []
        for j in range(nval):   
            if dist[i][j] <= band:
                w[i].append((1.0-(dist[i][j]/band)**2)**2)
            else:
                w[i].append(0.0)  
    return w
    

# Adaptive Bisquare kernel (nearest neighbors).

def adap_Bisquare(dist, nn):
    """
    Adaptive bi-square (NN)
    
    Methods: p58, (2.29), Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        dist: dictionary, dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n
        nn: Nth nearest neighbors of point i.
            
    Return:
        weit: dictionary, n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij,  
    """
    w = {}
    npt = len(dist.keys())
    nval = len(dist[0])
    nn = int(nn)
        
    for i in range(npt):
        w[i] = []
        
        lstDist = []
        for key, val in dist[i].items():
            lstDist.append((key, val))
        types = [('ptID', float), ('Distance', float)]
        arrDist = np.array(lstDist, dtype=types)
        arrDistSort = np.sort(arrDist, order='Distance') # sort based on distance        
        
        d_0 = arrDistSort[nn-1][1] # get distance threshold based on Nth neighbor
        for j in range(nval): # N includes point i self
            if dist[i][j] <= d_0:
                w[i].append((1.0-(dist[i][j]/d_0)**2)**2)   
            else:
                w[i].append(0.0)
    return w    

#--End of Kernel specification----------------------------------------------------------------

if __name__ == '__main__': 
    pass


