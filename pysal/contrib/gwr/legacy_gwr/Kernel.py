# Author: Jing Yao
# July, 2013
# Univ. of St Andrews, Scotland, UK

# For Kernel specification
from math import sqrt, exp
import numpy as np


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
            w[i].append(exp(-0.5*(dist[i][j]/band)**2)) #exp(-0.5*(dist[i][j]/band)**2) #w[i].append(exp(-0.5*(dist[i][j]/band)**2))
            
    ## reset wii
    #if wii == 0:
        #for i in w.keys():
            #w[i][i] = 0
        
    return w
    
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
            w[i].append(exp(-0.5*(dist[i][j]/d_0)**2))            #w[i].append(exp(-0.5*(dist[i][j]/d_0)**2))            
            
    ## reset wii
    #if wii == 0:
        #for i in w.keys():
            #w[i][i] = 0   
            
    return w    
    
    
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
                
    ## reset wii
    #if wii == 0:
        #for i in w.keys():
            #w[i][i] = 0    
        
    return w
    
    
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
                
    ## reset wii
    #if wii == 0:
        #for i in w.keys():
            #w[i][i] = 0    
        
    return w    

#--End of Kernel specification----------------------------------------------------------------


#--Global variables----------------------------------------------------------------
get_kernel = {0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare} # define kernel function
get_kernelName = {0: 'Fixed Gaussian', 1: 'Adaptive Gaussian', 2: 'Fixed bi-square', 3: 'Adaptive bi-square'} # define kernel name

#----------------------------------------------------------------------------------

def get_pairDist(data, flag=0):
    """
    get pairwise distance between all the data points
    
    Arguments:
        data: dictionary
              including (x,y) coordinates involved in the weight evaluation (including point i) 
        flag: dummy,
              0 or 1, 0: Euclidean distance; 1: spherical distance
       
    return:
        dist: dictionary
              dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n    
    """
    dist = {}
    npt = len(data.keys())
    for i in range(npt): # get distance dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n
        dist[i] = {}
        for j in range(npt):
            if j == i:
                dist[i][j] = 0.0
            elif j > i:
                if flag == 0:
                    dist[i][j] = get_EucDist(data[i],data[j])
                else:
                    dist[i][j] = get_latlongDist(data[i],data[j])
            else:
                dist[i][j] = dist[j][i]
    return dist
    
def get_focusDist(pt, data, flag=0):
    """
    get distance from pt to all the points in "data" -- for use in the prediction of umsampled locations
    
    Arguments:
        pt  : tuple
              (x,y) of focused location
        data: dictionary
              including (x,y) coordinates involved in the weight evaluation
        flag: dummy,
              0 or 1, 0: Euclidean distance; 1: spherical distance
       
    return:
        dist: dictionary
              dist={0:d_00,1:d_01,2:d_02...}, n  
    """
    dist = {}
    npt = len(data.keys())
    for i in range(npt): # get distance dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n
        if flag == 0:
            dist[i] = get_EucDist(pt, data[i])
        else:
            dist[i] = get_latlongDist(pt, data[i])
            
    return dist
    

def set_Kernel(dist, band, types):
    """
    Define kernel weight according to the type
    
    Arguments:
        dist: dictionary, dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n #including (x,y) coordinates involved in the weight evaluation (including point i) 
        band: float, bandwidth for single bandwidth
        types: integer, define which kernel function to use, refer to "get_kernel"
    return:
        weit: dictionary, n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...} .
    """
    #dist = {}
    #dist = get_pairDist(data)
    #npt = len(data.keys())
    #for i in range(npt): # get distance dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n
        #dist[i] = {}
        #for j in range(npt):
            #if j == i:
                #dist[i][j] = 0.0
            #elif j > i:
                #dist[i][j] = get_EucDist(data[i],data[j])
            #else:
                #dist[i][j] = dist[j][i]
                
        
    
    return get_kernel[types](dist, band) 

def set_Bandwidth(data, method=1, band=0.0, criterion=0.0, maxVal=0.0, minVal=0.0, interval=0.0):
    """
    Set bandwidth using specified method and parameters
        
    Arguments:
        data: dictionary, including (x,y) coordinates involved in the weight evaluation (including point i) 
        band: float, bandwidth for single bandwidth
        criterion: integer, criteria for bandwidth selection
                   0: AICc
                   1: AIC
                   2: BIC/MDL
                   3: CV
        method: integer, method to use
                0: Gloden section search
                1: Single bandwith 
                2: Interval Search
        maxVal: max value for method 0 or 2
        minVal: min value for method 0 or 2
        interval: float, for Interval Search        
            
    Return:
        band: float, bandwidth       
    """
    if method == 1:
        return band
    if method == 0:
        return band_Golden(data, criterion, maxVal, minVal)
    if method == 2:
        return band_Interval(data, criterion, maxVal, minVal, interval)
    
    
def band_Golden(data, criterion, maxVal, minVal):
    """
    Set bandwidth using golden section search
    
    Methods: p212-213, section 9.6.4, Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        data: dictionary, including (x,y) coordinates involved in the weight evaluation (including point i) 
        criterion: integer, criteria for bandwidth selection
                   0: AICc
                   1: AIC
                   2: BIC/MDL
                   3: CV
        maxVal: max value for method 0 or 2
        minVal: min value for method 0 or 2       
            
    Return:
        band: float, bandwidth
    """
    a, b, c
    b = (1-0.618) * abs(c-a)
    
    get_criteria[criterion]()

def band_Interval(data, criterion, maxVal, minVal, interval):
    """
    Set bandwidth using interval search
    
    Methods: p61, (2.34), Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        data: dictionary, including (x,y) coordinates involved in the weight evaluation (including point i) 
        criterion: integer, criteria for bandwidth selection
                   0: AICc
                   1: AIC
                   2: BIC/MDL
                   3: CV
        maxVal: max value for method 0 or 2
        minVal: min value for method 0 or 2
        interval: float, for Interval Search        
            
    Return:
        band: float, bandwidth
    """  
    get_criteria[criterion]()
    
class GWR_W(object):
    """
    set kernel weight
    
    Parameters
    ----------
    coords    : dictionary               
                including (x,y) coordinates involved in the weight evaluation (including point i)     
    band      : float
                if it is fixed kernel, band is the bandwidth; if it is adaptive kernel, it is Nth nearest neighbors    
    wType     : integer
                define which kernel function to use, get_kernel = {0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare} # define kernel function    
    dist:     : dictionary
                dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n
                

    Attributes
    ----------
    w         : dictionary
                n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...} .  
    dist      : dictionary
                dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n 
    band      : float
                bandwidth
    wType     : integer
                define which kernel function to use  
    wName     : string
                define the kernel function Name   
    
    """
    def __init__(self, coords, band, wType=0, dist=None, flag=0):
        """
        Initialize class
        """   
        self.coords = coords
        self.band = band
        self.wType = wType  
        self.wName = get_kernelName[self.wType]
        if dist is None:
            self.dist = get_pairDist(coords,flag)
        else:
            self.dist = dist
        
        self.w = set_Kernel(self.dist, self.band, self.wType)        



#--End of Bandwidth selection ----------------------------------------------------------------

if __name__ == '__main__': 
    
    # Examples
    
    #print get_EucDist((0,3),(3,7))
    pass
    #--read shapefile------------------------------
    #--read shapefile------------------------------
    #--read shapefile------------------------------
    #--read shapefile------------------------------
    #--read shapefile------------------------------
    #--read shapefile------------------------------

