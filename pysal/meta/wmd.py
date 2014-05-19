"""
Weights Meta Data

Prototyping meta data functions and classes for weights provenance


"""
__author__ = "Sergio J. Rey <srey@asu.edu>, Wenwen Li <wenwen@asu.edu>"
import pysal as ps
import io, json
import httplib
from urlparse import urlparse
import urllib2 as urllib


def contiguity(arg_dict):
    """
    General handler for building contiguity weights from shapefiles

    """
    input1 = arg_dict['input1']
    uri = input1['uri']
    weight_type = arg_dict['weight_type']
    weight_type = weight_type.lower()
    if weight_type == 'rook':
        w = ps.rook_from_shapefile(uri)
    elif weight_type == 'queen':
        w = ps.queen_from_shapefile(uri)
    else:
        print "Unsupported contiguity criterion: ",weight_type
        return None
    if 'parameters' in arg_dict:
        order = arg_dict['parameters'].get('order',1) # default to 1st order
        lower = arg_dict['parameters'].get('lower',0) # default to exclude lower orders
        if order > 1:
            w_orig = w
            w = ps.higher_order(w,order)
            if lower:
                for o in xrange(order-1,1,-1):
                    w = ps.weights.w_union(ps.higher_order(w_orig,o), w)
                w = ps.weights.w_union(w, w_orig)
        parameters = arg_dict['parameters']
    else:
        parameters = {'lower': 0, 'order':1 }
    w = WMD(w.neighbors, w.weights)
    w.meta_data = {}
    w.meta_data["input1"] = {"type": 'shp', 'uri':uri}
    w.meta_data["transform"] = w.transform
    w.meta_data["weight_type"] =  weight_type
    w.meta_data['parameters'] = parameters
    return w


def kernel(arg_dict):
    """
    General handler for building kernel based weights from shapefiles
    """
    input1 = arg_dict['input1']
    uri = input1['uri']
    weight_type = arg_dict['weight_type']
    weight_type = weight_type.lower()
    k = 2
    bandwidths = None
    function = 'triangular'
    if 'parameters' in arg_dict:
        k = arg_dict['parameters'].get('k',k) # set default to 2
        bandwidths = arg_dict['parameters'].get('bandwidths',bandwidths) 
        function = arg_dict['parameters'].get('function', function) 
    else:
        parameters = {}
        parameters['k'] = k
        parameters['bandwidths'] = bandwidths
        parameters['function'] = function
        arg_dict['parameters'] = parameters


    if weight_type == 'akernel':
        # adaptive kernel
        w = ps.adaptive_kernelW_from_shapefile(uri, bandwidths = bandwidths,
                k=k, function = function)
    elif weight_type == 'kernel':
        w = ps.kernelW_from_shapefile(uri, k=k, function = function)
    else:
        print "Unsupported kernel: ",weight_type
        return None
    w = WMD(w.neighbors, w.weights)
    w.meta_data = {}
    w.meta_data["input1"] = {"type": 'shp', 'uri':uri}
    w.meta_data["transform"] = w.transform
    w.meta_data["weight_type"] =  weight_type
    w.meta_data['parameters'] = arg_dict['parameters']
    return w


def distance(arg_dict):
    """
    General handler for distance based weights obtained from shapefiles
    """
    input1 = arg_dict['input1']
    uri = input1['uri']
    weight_type = arg_dict['weight_type']
    weight_type = weight_type.lower()
    k = 2
    id_variable = None
    p = 2
    radius = None
    if 'parameters' in arg_dict:
        k = arg_dict['parameters'].get('k',k) # set default to 2
        id_variable = arg_dict['parameters'].get('id_variable', id_variable)
        p = arg_dict['parameters'].get('p',p) 
        radius = arg_dict['parameters'].get('radius', radius)

    else:
        parameters = {}
        parameters['k'] = 2
        parameters['id_variable'] = None
        parameters['radius'] = None
        parameters['p'] = 2
        arg_dict['parameters'] = parameters

    if weight_type == 'knn':
        w = ps.knnW_from_shapefile(uri,k=k,p=p,idVariable=id_variable,
                radius=radius)
        w = WMD(w.neighbors, w.weights)
        w.meta_data = {}
        w.meta_data["input1"] = {"type": 'shp', 'uri':uri}
        w.meta_data["weight_type"] =  'knn'
        w.meta_data["transform"] = w.transform
        w.meta_data['parameters'] = arg_dict['parameters']
        return w

def higher_order(arg_dict):
    wmd = arg_dict['wmd']
    order = 2
    if 'parameters' in arg_dict:
        order = arg_dict['parameters'].get('order', order)
    else:
        parameters = {}
        parameters['order'] = order
        arg_dict['parameters'] = parameters


    w = ps.higher_order(wmd, order)
    w = WMD(w.neighbors, w.weights)
    w.meta_data = {}
    w.meta_data['input1'] = arg_dict['input1']
    w.meta_data['parameters'] = arg_dict['parameters']

    return w

def geojsonf(arg_dict):
    """
    Handler for local geojson files
    """
    input1 = arg_dict['input1']
    uri = input1['uri']
    weight_type = arg_dict['weight_type']
    weight_type = weight_type.lower()
    id_variable = None

    if weight_type == 'queen_geojsonf':
        w = ps.weights.user.queen_from_geojsonf(uri)
        w.meta_data = {}
        w.meta_data["input1"] = {"type": 'geojsonf', 'uri':uri}
        w.meta_data["weight_type"] =  'queen'
        w.meta_data["transform"] = w.transform
        return w




# wrapper dict that maps specific weights types to a handler function that
# builds the specific weights instance
WEIGHT_TYPES = {}
WEIGHT_TYPES['rook'] = contiguity
WEIGHT_TYPES['queen'] = contiguity
WEIGHT_TYPES['akernel'] = kernel
WEIGHT_TYPES['kernel'] = kernel
WEIGHT_TYPES['knn'] = distance
WEIGHT_TYPES['higher_order'] = higher_order
#WEIGHT_TYPES['queen_geojson'] = geojson
#WEIGHT_TYPES['queen_geojsonf'] = geojsonf
#WEIGHT_TYPES['geojsons'] = geojsons




def uri_reader(uri):
    j = json.load(urllib.urlopen(uri))
    return j

def wmd_read_only(fileName):
    try:
        meta_data = uri_reader(fileName)
    except:
        fp = open(fileName)
        meta_data = json.load(fp)
        fp.close()
    return meta_data


def wmd_reader(fileName):
    try:
        meta_data = uri_reader(fileName)
    except:
        fp = open(fileName)
        meta_data = json.load(fp)
        fp.close()
    w = wmd_parser(meta_data)
    return w
    

def wmd_writer(wmd_object, fileName, data=False):
    #print json.dumps(wmd_object.meta_data, 
    #        indent=4,
    #        separators=(',', ': '))
    fp = open(fileName, 'w')
    if data:
        wmd_object.meta_data['data'] = {}
        wmd_object.meta_data['data']['weights'] = wmd_object.weights
        wmd_object.meta_data['data']['neighbors'] = wmd_object.neighbors
    json.dump(wmd_object.meta_data, 
            fp,
            indent=4,
            separators=(',', ': '))
    fp.close()



def wmd_parser(wmd_object):

    weight_type = wmd_object['weight_type'].lower()

    # check if data for weights and neighbors is included already

    if 'data' in wmd_object:
        # check if keys got cast to strings in json writer
        neighbors = wmd_object['data']['neighbors']
        keys = neighbors.keys()
        ktype = type(keys[0])
        vtype = type(neighbors[keys[0]][0])
        if ktype != vtype:
            n_1 = {}
            w_1 = {}
            for key in keys:
                n_1[int(key)] = neighbors[key]
                w_1[int(key)] = wmd_object['data']['weights'][key]
            wmd_object['data']['neighbors'] = n_1
            wmd_object['data']['weights'] = w_1
        w = WMD(neighbors = wmd_object['data']['neighbors'],
                weights = wmd_object['data']['weights'])
        w.meta_data = {}
        w.meta_data['input1'] = wmd_object['input1']
        w.meta_data['weight_type'] = wmd_object['weight_type']
        w.meta_data['transform'] = w.transform
        if 'parameters' in wmd_object:
            w.meta_data['parameters'] = wmd_object['parameters']
        return w

    # check input type
    itype = wmd_object['input1']['type']
    if itype == 'prov':
        #      call wmd_reader
        uri = wmd_object['input1']['uri']
        meta_data = wmd_read_only(uri)
        wmd = wmd_parser(meta_data)
        wmd_object['wmd'] = wmd
    else:
        # handle distributed files
        uri = wmd_object['input1']['uri']
        try:
            tmp = open(uri)
            print ' tmp: ', tmp
            wmd_object['input1']['uri'] = uri
        except:
            download_shapefiles(uri)
            uri = uri.split("/")[-1]
            wmd_object['input1']['uri'] = uri # use local copy


    # check for weight_type
    if weight_type in WEIGHT_TYPES:
        print weight_type
        return  WEIGHT_TYPES[weight_type](wmd_object)
    else:
        print 'Unsupported weight type: ', weight_type
        return None


def download_shapefiles(file_name):

    file_parts = file_name.split("/")
    file_prefix = file_parts[-1].split(".")[0]
    exts = [ ".shp", ".dbf", ".shx" ]
    for ext in exts:
        # rebuild url
        file_name = file_prefix + ext
        file_parts[-1] = file_name
        new_url = "/".join(file_parts)
        print file_name, new_url
        u = urllib.urlopen(new_url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print "Downloading: %s Bytes: %s" % (file_name, file_size)
        file_size_dl = 0
        block_sz = 8192
        while True:
            bf = u.read(block_sz)
            if not bf:
                break
            file_size_dl += len(bf)
            f.write(bf)
            status = r"%10d [%3.2f%%]" % (file_size_dl, file_size_dl * 100. /
                    file_size)
            status = status + chr(8)* (len(status)+1)
        print status, f.close()


class WMD(ps.W):
    """Weights Meta Data Class """
    def __init__(self, neighbors=None,  weights=None, id_order=None):
        super(WMD, self).__init__(neighbors, weights, id_order)



if __name__ == '__main__':


    # distributed file
    w1 = wmd_reader("http://toae.org/pub/wrook1.wmd")

    # order
    w1o = wmd_reader('wrooko1.wmd')
    w2o = wmd_reader('wrooko2.wmd')
    w2ol = wmd_reader('wrooko2l.wmd')

    # kernels
    ak1 = wmd_reader('akern1.wmd')
    kern = wmd_reader('kernel.wmd')

    # knn
    knn = wmd_reader('knn.wmd')



    # moran workflow
    import pysal as ps


    # geojson
    #wj = wmd_reader("wgeojson.wmd")


    # here we test chaining
    #r1 = wmd_reader('chain.wmd')

    r2 = wmd_reader('chain2.wmd')
    

