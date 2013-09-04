import pysal as ps
import numpy as np

def getAttributeNames(dbf_file_name):

    fp = ps.open(dbf_file_name)
    attribute_names = fp.header
    fp.close()
    return attribute_names 



def local_moran(dbf_file_name, shp_file_name, attribute_name,
        contiguity_type='QUEEN'):

    fp = ps.open(dbf_file_name)
    y = np.array(fp.by_col(attribute_name))
    
    ct = contiguity_type.upper()

    if ct == 'QUEEN':
        w = ps.queen_from_shapefile(shp_file_name)
    elif ct == 'ROOK':
        w = ps.rook_from_shapefile(shp_file_name)
    else:
        print 'Unsupported contiguity type: ', contiguity_type
        return None

    w.transform = 'r' # row standardize

    lm = ps.Moran_Local(y, w, permutations = 99)

    # create a binary attribute to indicate which shapes have significant
    # lisas

    sig_lisas = 1* (lm.p_sim < 0.05)

    # write out data for scatter plot (x-axis is the attribute, y is the
    # spatial lag)
    wy = ps.lag_spatial(w,y)
    scatter_data = np.vstack((y, wy)).T
    output_file = "_".join(['scatter', contiguity_type, \
            attribute_name])+".txt"
    np.savetxt(output_file, scatter_data, fmt="%12.8f")


    # write out the binary data for significant lisas
    output_file = contiguity_type + "_" + attribute_name + ".txt"
    np.savetxt(output_file, sig_lisas, fmt="%d")


    # write out histogram on adjacency cardinalities
    # first value the number of neighbors, second value, the number of
    # polygons having that many neighbors
    output_file = 'histogram_%s.txt'%contiguity_type
    np.savetxt(output_file, w.histogram, "%d")



if __name__ == '__main__':


    # Step 0
    # set random number generator seed
    import numpy as np
    np.random.seed(12345)

    # Step 1 get attribute names
    # set dbf file name here
    # for now we use canned national example
    # this could be swapped out for a user defined file

    dbf_file_name = ps.examples.get_path("NAT.dbf")
    names = getAttributeNames(dbf_file_name)

    # names is a python list of strings with attribute names
    # populate a list box to let the user select the attribute of interest


    # Step 2 carry out local Moran's I

    # prompt user for 
    # dbf_file_name
    # shp_file_name
    # type of contiguity [QUEEN|ROOK]

    # attribute name (found in step one so we don't need to prompt)

    shp_file_name = ps.examples.get_path("NAT.shp")

    # example call
    local_moran(dbf_file_name, shp_file_name, 'HR80', contiguity_type='QUEEN')

    # this writes out three files
    # 1. QUEEN_HR80.txt - binary indicator if shape has significant lisa or not
    # 2. scatter_QUEEN_HR80.txt - data for the scatter plot, first column is
    #    attribute value, second column is value of spatial lag
    # 3. histogram_QUEEN.txt values for a histogram, first column is number of
    # neighbors, second column is the count of polygons with that many
    # neighbors

    # now for rook

    local_moran(dbf_file_name, shp_file_name, 'HR80', contiguity_type='ROOK')

    # three more files as in the case for queen, only substitute ROOK for
    # QUEEN

