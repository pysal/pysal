import numpy as np
import folium as fm
import pysal as ps
import pandas as pd
import geojson as gj
import os as os
from IPython.display import HTML

def build_features(shp, dbf):
    '''
    Builds a GeoJSON object from a PySAL shapefile and DBF object

    shp - shapefile opened using pysal.open(file)
    dbf - dbase table opened using pysal.open(file)

    Only polygonal lattices are supported. 
    '''



    shp_bak = ps.open(shp.dataPath)
    dbf_bak = ps.open(dbf.dataPath)
        
    chains = shp_bak.read()
    dbftable = dbf_bak.read() 
    
    shp_bak.close()
    dbf_bak.close()

    #shptype = str(shp_bak.type).strip("<class 'pysal.cg.shapes.").strip("'>")
    
    if 'Polygon' in str(shp_bak.type):
        ftype = 'Polygon'
    elif 'Point' in str(type(shp_bak.type)):
        raise NotImplementedError('Point data is not implemented yet')

    if ftype == "Polygon":
        feats = []
        for idx in range(len(chains)):
            chain = chains[idx]
            if len(chain.parts) > 1:
                #shptype = 'MultiPolygon'
                geom = gj.MultiPolygon([ [[ list(coord) for coord in part]] for part
                    in chain.parts])
            else:
                #shptype = 'Polygon'
                geom = gj.Polygon(coordinates = [ [ list(coord) for coord in
                    part] for part in chain.parts])
            prop = {head: val for head,val in zip(dbf_bak.header,
                    dbftable[idx])}
            bbox = chain.bbox
            feats.append(gj.Feature(None, geometry=geom, properties=prop, bbox=bbox))
        
    return gj.FeatureCollection(feats, bbox = shp_bak.bbox )

def json2df(jsonobj, index_on = ''):
    '''
    Reads a json file and constructs a pandas dataframe from it.

    jsonobj - the filepath to a JSON file. 
    index_on - a fieldname which the final pandas dataframe will be indexed on.

    '''
    n = len(jsonobj['features'])
    rows = [ jsonobj['features'][i]['properties'] for i in range(n) ]
    try:
        idxs = [ jsonobj['features'][i]['properties'][index_on] for i in range(n) ] 
        result = pd.DataFrame(rows, index=idxs ) 
    except KeyError:
        result = pd.DataFrame(rows)
    return result

def flip(fname, shp, dbf):
    with open(fname, 'w') as out:
        gj.dump(build_features(shp, dbf), out)
            
def bboxsearch(jsonobj):
    '''
    Searches over a list of coordinates in a pandas dataframe to construct a
    bounding box
     
    df - pandas dataframe with fieldname "geom_name", ideally constructed from
    json using json2df
    geom_name - the name of the geometry field to be used.
    '''
    max_x = -180
    max_y = -90
    min_x = 180
    min_y = 90

    for feat in jsonobj.features:
        geom = feat.geometry.coordinates
        for chain in geom:
            for piece in chain:
                if type(piece[0]) != float:                
                    for point in piece:
                        if point[0] > max_x:
                            max_x = point[0]
                        elif point[0] < min_x:
                            min_x = point[0]
                        if point[1] > max_y:
                            max_y = point[1]
                        elif point[1] < min_y:
                            min_y = point[1]
                else:
                    if piece[0] > max_x:
                        max_x = piece[0]
                    elif piece[0] < min_x:
                        min_x = piece[0]
                    if piece[1] > max_y:
                        max_y = piece[1]
                    elif piece[1] < min_y:
                        min_y = piece[1]
    return [min_x, min_y, max_x, max_y] 

def choropleth_map(jsonpath, key, attribute, df = None, 
                   classification = "Quantiles", classes = 5, bins = None, std = None,
                   centroid = None, zoom_start = 5, tiles = 'OpenStreetMap',
                   fill_color = "YlGn", fill_opacity = .5, 
                   line_opacity = 0.2, legend_name = '', 
                   save = True):
    '''
    One-shot mapping function for folium-based choropleth mapping. 

    jsonpath - the filepath to a JSON file
    key - the field upon which the JSON and the dataframe will be linked
    attribute - the attribute to be mapped

    The rest of the arguments are keyword:
    
    classification - type of classification scheme to be used
    classes - number of classes used
    bins - breakpoints, if manual classes are desired


    '''
    
    #Polymorphism by hand...

    if isinstance(jsonpath, str): 
        if os.path.isfile(jsonpath):
            sjson = gj.load(open(jsonpath))
        else:
            raise IOError('File not found')

    if isinstance(jsonpath, dict):
        raise NotImplementedError('Direct mapping from dictionary not yet supported')
        #with open('tmp.json', 'w') as out:
        #    gj.dump(jsonpath, out)
        #    sjson = gj.load(open('tmp.json'))

    if isinstance(jsonpath, tuple):
        if 'ShpWrapper' in str(type(jsonpath[0])) and 'DBF' in str(type(jsonpath[1])):
            flip('tmp.json', jsonpath[0], jsonpath[1])
            sjson = gj.load(open('tmp.json'))
            jsonpath = 'tmp.json'

        elif 'ShpWrapper' in str(type(jsonpath[1])) and 'DBF' in str(type(jsonpath[0])):
            flip('tmp.json', jsonpath[1], jsonpath[0])
            sjson = gj.load(open('tmp.json'))
            jsonpath = 'tmp.json'

        else:
            raise IOError('Inputs must be GeoJSON filepath, GeoJSON dictionary in memory, or shp-dbf tuple')

    #key construction
    if df is None:
        df = json2df(sjson)
    dfkey = [key, attribute]
    
    #centroid search
    if centroid == None:
        if 'bbox' in sjson.keys():
            bbox = sjson.bbox
        bbox = bboxsearch(sjson)
        xs = sum([bbox[0], bbox[2]])/2.
        ys = sum([bbox[1], bbox[3]])/2.
        centroid = [ys, xs]
    jsonkey = 'feature.properties.' + key
    
    choromap = fm.Map(location = centroid, zoom_start = zoom_start, tiles=tiles) # all the elements you need to make a choropleth 
    
    #standardization 
    if std != None:
        if isinstance(std, int) or isinstance(std, float):
            y = np.array(df[attribute]/std)
        elif type(std) == str:
            y = np.array(df[attribute]/df[std])
        elif callable(std):
            raise NotImplementedError('Functional Standardizations are not implemented yet')
        else:
            raise ValueError('Standardization must be integer, float, function, or Series')
    else:
        y = np.array(df[attribute].tolist())
    
    #For people who don't read documentation...
    if isinstance(classes, list):
        bins = classes
        classes = len(bins)
    elif isinstance(classes, float):
        try:
            classes = int(classes)
        except:
            raise ValueError('Classes must be coercable to integers')

    #classification passing
    if classification != None:
        if classification == "Maximum Breaks": #there is probably a better way to do this, but it's a start. 
            mapclass = ps.Maximum_Breaks(y, k=classes).bins.tolist()
        elif classification == 'Quantiles':
            mapclass = ps.Quantiles(y, k=classes).bins.tolist()
        elif classification == 'Fisher-Jenks':
            mapclass = ps.Fisher_Jenks(y, k=classes).bins
        elif classification == 'Equal Interval':
            mapclass = ps.Equal_Interval(y, k=classes).bins.tolist()
        elif classification == 'Natural Breaks':
            mapclass = ps.Natural_Breaks (y, k=classes).bins
        elif classification == 'Jenks Caspall Forced':
            raise NotImplementedError('Jenks Caspall Forced is not implemented yet.') 
        #   mapclass = ps.Jenks_Caspall_Forced(y, k=classes).bins.tolist()
        elif classification == 'Jenks Caspall Sampled':
            raise NotImplementedError('Jenks Caspall Sampled is not implemented yet')
        #   mapclass = ps.Jenks_Caspall_Sampled(y, k=classes).bins.tolist()
        elif classification == 'Jenks Caspall':
           mapclass = ps.Jenks_Caspall (y, k=classes).bins.tolist()
        elif classification == 'User Defined':
            mapclass = bins
        elif classification == 'Standard Deviation':
            if bins == None:
                l = classes / 2
                bins = range(-l, l+1)
                mapclass = list(ps.Std_Mean(y, bins).bins)
            else:
                mapclass = list(ps.Std_Mean(y, bins).bins)
        elif classification == 'Percentiles':
            if bins == None:
                bins = [1,10,50,90,99,100]
                mapclass = list(ps.Percentiles(y, bins).bins)
            else:
                mapclass = list(ps.Percentiles(y, bins).bins)
        elif classification == 'Max P':
            #raise NotImplementedError('Max-P classification is not implemented yet')
            mapclass = ps.Max_P_Classifier(y, k=classes).bins.tolist()
        else:
            raise NotImplementedError('Your classification is not supported or was not found. Supported classifications are:\n "Maximum Breaks"\n "Quantiles"\n "Fisher-Jenks"\n "Equal Interval"\n "Natural Breaks"\n "Jenks Caspall"\n "User Defined"\n "Percentiles"\n "Max P"')
    else:
        print('Classification forced to None. Defaulting to Quartiles')
        mapclass = ps.Quantiles(y, k=classes).bins.tolist()

    #folium call, try abstracting to a "mapper" function, passing list of args
    choromap.geo_json(geo_path=jsonpath, key_on = jsonkey, 
                      data = df, columns = dfkey, 
                      fill_color = fill_color, fill_opacity = fill_opacity,
                      line_opacity = line_opacity, threshold_scale = mapclass[:-1] , legend_name = legend_name
                      )

    if save:
        fname = jsonpath.rstrip('.json') + '_' + attribute + '.html'
        choromap.save(fname)
    
    return choromap

