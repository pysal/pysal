"""
google: Tools for extracting information through Google Places API

Author(s):
    Luc Anselin luc.anselin@asu.edu
    Xun Li xun.li@asu.edu

"""

__author__ = "Luc Anselin <luc.anselin@asu.edu, Xun Li <xun.li@asu.edu"

import urllib2
import json
from pysal.cg import lonlat,geogrid,harcdist

__all__ = ['querypoints','queryradius','googlepoints','ptdict2geojson']

def querypoints(p,bbox=True,lonx=False,k=5):
    """
    Utility function to generate a list of grid query points from a bounding box
    essentially a wrapper around geogrid with a check for bounding box dimension
    
    Parameters
    ----------
    p      : list of points in alt-lon or lon-lat
    bbox   : flag for bounding box (upper left, lower right), default = True
    lonx   : flag for lat-lon order, default is False for lat-lon (True is lon-lat)
    k      : grid size (points grid will be k+1 by k+1)
    
    Returns
    -------
    grid   : list of lat-lon of a grid of query points, row by row, from top to bottom
    
    """
    if bbox:
        if not len(p)==2:
            raise Exception, "invalid format for bounding box"
        # create grid points from bounding box
        grid = geogrid(p[0],p[1],k=k,lonx=lonx)
        # the list must be in lat-lon format
        if lonx:
            grid = lonlat(grid)
    else:
        return p
    return grid
    
def queryradius(grid,k,lonx=False):
    """
    Utility function to find a good query range distance for grid points
    
    Parameters
    ----------
    grid    : list of lat-lon tuplets for regular grid points
              (created with querypoints)
    k       : dimension of the grid, the list will consist of k+1 x k+1 tuplets
    lonx    : flag for lat-lon order, default is False for lat-lon (lon-lat is True)
    
    Returns
    -------
            : maximum distance from the horizontal and vertical sides of
              the lower right grid cell in meters
              (to be used as the sradius argument in googlepoints)
    
    """
    # compute distances in horizontal and vertical direction
    # use the lower right grid cell
    # along longitude
    p0 = grid[-2]
    p1 = grid[-1]
    # along latitude
    p3 = grid[-(k+2)]
    # horizontal distance
    dist1 = harcdist(p0,p1,lonx=lonx)
    # vertical distance
    dist2 = harcdist(p3,p1,lonx=lonx)
    return round(max([dist1,dist2])*1000.0)

def googlepoints(querypoints,apikey,sradius,stype,newid=True,verbose=True):
    """
    Queries the Google Places API for locations (lat-lon) of a 
    given type of facility
    
    Parameters
    ----------
    querypoints  : a list of tuples with point coordinates in lat-lon format 
                   (not lon-lat!)
    apikey       : the user's Google Places API key
    sradius      : search radius in meters
    stype        : facility type from list supported by Google Places API
    newid        : flag for sequential integer as new key
                   default = True
                   should be set to False when result dictionary will
                   be used as input to googlevalues
    verbose      : flag for amount of detail in output
                   default = True, for each point query, the status and
                   number of points found is printed
                   False: only a warning is printed when search is truncated
                   at 200
    
    Returns
    -------
    findings     : a directory with the facility Google ID as the key
                   contains lat and lng
                   
    Example
    -------
    
    It is not possible to give an actual example, since that requires the
    user's unique API key. However, to illustrate the type of call and
    return consider:
    
    liquorlist = googlepoints([(41.981417, -87.893517)],'apikeyhere',5000,'liquor_store')
    
    which will query a radius of 5000m (5km) around the specified query point for the
    locations of all liquor stores in the Google data base
    
    This retuns a dictionary:
    
    {u'ChIJ-etp1PLJD4gRLYGl7Jm1oG4': {u'lat': 41.986742, u'lng': -87.836312},
     u'ChIJ0fyhh9awD4gR6-vaMEbGNBE': {u'lat': 42.022235, u'lng': -87.941302},
    ...
     u'ChIJxfyRaki2D4gRiaHLJANgJns': {u'lat': 42.010285, u'lng': -87.875896}}
     
    The key in the dictionary is the Google ID of the facility
    """
    base_url = 'https://maps.googleapis.com/maps/api/place/radarsearch/json?location=%s,%s&radius=%s&types=%s&key=%s'
    # list of urls to query
    query_urls = [base_url%(lat,lon,sradius,stype,apikey) for lat,lon in querypoints]
    findings = {}
    
    for url in query_urls:
        rsp = urllib2.urlopen(url)
        content = rsp.read()
        data = json.loads(content)
        if 'results' in data:         # avoid empty records
            results = data['results']
            if verbose:
                print data['status'], len(results)
            else:
                if len(results) == 200:
                    print "WARNING: query truncated at 200"
            for item in results:
                place_id = item['place_id']
                loc = item['geometry']['location']
                # use place id as key in the dictionary
                findings[place_id] = loc
            
    # replace key with integer sequence number
    if newid:
        ii = 0
        newdict = {}
        for placeid in findings.keys():
            dd = findings[placeid]
            dd['placeid']=placeid  
            newdict[ii]=dd
            ii = ii + 1
    else:
        return findings
    
    return newdict
    
def ptdict2geojson(d,location=["lng","lat"],values=[],ofile="output.geojson"):
    """
    Turns the dictionary output from googlepoints into a geojson point file
    
    Parameters
    ----------
    d          : dict object created by googlepoints
    location   : list with long, lat coordinates in decimal degrees
                 (no check for order)
    values     : list of keys for data items other than the id
                 (do not include the id)
                 default: no values (id only)
    ofile      : file name for the output file (include geojson extension)
                 default: "output.geojson"
    
    Returns
    -------
               : geojson point file
    
    Remarks
    -------
    Assumes that the dictionary is complete, i.e., every key has a value associated
    with it. No checks are made for missing keys.
    
    """
    # extract location info
    lng = location[0]
    lat = location[1]
    # initialize geo dictionary for point features
    geo = {"type": "FeatureCollection","features":[]}
    # loop over dictionary
    for id,loc in d.iteritems():
        feature = { "type" : "Feature", "geometry": { "type": "Point", "coordinates": [ loc['lng'],loc['lat']]}}
        properties = {"id": id}
        for info in values:
            properties[info] = loc[info]
        feature["properties"] = properties
        geo["features"].append(feature)
    # output file
    with open(ofile,'w') as outfile:
        json.dump(geo,outfile,sort_keys=True,indent=4,ensure_ascii=False)
    outfile.close()
    return
    

    
