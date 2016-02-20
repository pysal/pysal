# Author: Jing Yao
# July, 2013
# Univ. of St Andrews, Scotland, UK

# For file input and output

#import shapelib, dbflib
import numpy as np
#import shapely
from shapely import *#geometry
import csv

#--Global variables----------------------------------------------------------------
#fldTypes = {0: dbflib.FTDouble, 1:dbflib.FTInteger, 2:dbflib.FTString} # field type for dbf file

#----------------------------------------------------------------------------------

#--1. read file--------------------------------------------------------------------

def read_SHP(fleName):
    """
    Read shapefile(point/polygon) and return (x,y) and all the attribute values

    Arguments:
        fleName: text, full directory of shapefile name, including ".shp"

    Return:
        dicFeat: dictionary, key: id of feature, value: (x, y) coordinates
        lstFlds: list, fields in the file
        dicAttr: dictionary, key: id of feature, value: tuple of attribute values
    """
    #fleSHP = shapelib.ShapeFile(fleName)
    nrec = fleSHP.info()[0]
    geoType = fleSHP.info()[1]
    dbfName = fleName.split(".")[0] + ".dbf"

    dicFeat = {}
    lstFlds = []
    dicAttr = {}

    # get (x,y)
    #--for polygon, use centroid------------------
    if geoType == 5:
        for i in range(nrec):
            vert = fleSHP.read_object(i)
            poly = shapely.geometry.Polygon(vert.vertices()[0])
            dicFeat[i] = (poly.centroid.x, poly.centroid.y)

    #--for point----------------------------------
    if geoType == 1:
        for i in range(nrec):
            vert = fleSHP.read_object(i)
            dicFeat[i] = vert.vertices()[0]

    # get record info
    lstFlds, dicAttr = read_DBF(dbfName)

    return dicFeat, lstFlds, dicAttr


def read_DBF(fleName):
    """
    Read DBF file and return all the attribute values

    Arguments:
        fleName: text, full directory of DBF name, including ".dbf"

    Return:
        lstFlds: list, fields in the file
        dicAttr: dictionary, key: id of feature, value: tuple of attribute values
    """
    #fleDBF = dbflib.DBFFile(fleName)
    nrec = fleDBF.record_count()

    lstFlds = []
    dicAttr = {}
    nFlds = fleDBF.field_count()
    for i in range(nFlds):
        lstFlds.append(fleDBF.field_info(i)[1])
    #lstFlds = fleDBF.read_record(0).keys()
    for i in range(nrec):
        rec = fleDBF.read_record(i)
        lst = []
        for fld in lstFlds:
            lst.append(rec[fld])
        dicAttr[i] = tuple(lst)

    return lstFlds, dicAttr

def read_CSV(fleName):
    """
    Read CSV file and return all the attribute values

    Arguments:
        fleName: text, full directory of CSV name, including ".csv"

    Return:
        lstFlds: list, fields in the file
        dicAttr: dictionary, key: id of feature, value: tuple of attribute values
    """
    lstFlds = []
    dicAttr = {}

    with open(fleName, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        rownum = 0
        for row in spamreader:
            # Save header row.
            if rownum == 0:
                header = row
                lstFlds = header[0].split(',')
            else:
                dicAttr[rownum-1] = tuple(row[0].split(','))      # note: all value type is string, need to change to number for future analysis
            rownum += 1
    csvfile.close()

    return lstFlds, dicAttr

def read_TXT(fleName):
    """
    Read TXT file (space separated) and return all the attribute values

    Arguments:
        fleName: text, full directory of TXT name, including ".txt"

    Return:
        lstFlds: list, fields in the file
        dicAttr: dictionary, key: id of feature, value: tuple of attribute values
    """
    lstFlds = []
    dicAttr = {}

    with open(fleName, 'rb') as txtfile:
        header = txtfile.readline() # Save header row.
        lstFlds = header.strip().split()
        rownum = 0
        for row in txtfile:
            dicAttr[rownum] = tuple(row.strip().split())      # note: all value type is string, need to change to number in future analysis
            rownum += 1
    txtfile.close()

    return lstFlds, dicAttr

#--End of read file--------------------------------------------------------------------

#--2. write file-----------------------------------------------------------------------
def write_SHP(fleName, geoType, geoList, fldList, attList):
    """
    Write shapefile(point/polygon) and associated DBF file

    Arguments:
        fleName: text, full directory of shapefile name, including ".shp"
        geoType: integer, type of geometry, either "POINT" (1) or "POLYGON" (2)
        geoList: list, including (x,y) series for each geometry.
        fldList: list of tuples/lists, including names, types, precisions, scale of fields, [(name, type, precision, scale), (), ... ()]
                 field type mapping is given by global variable "fldTypes".
        attList: list of tuples, including values of attributes, [(val1, val2, ... valn), (), ... ()]

    Return:
        None
    """
    # check geometry type
    if geoType == 1:
        pass
        #geo = shapelib.SHPT_POINT
    if geoType == 2:
        pass
        #geo = shapelib.SHPT_POLYGON
    if not (geoType == 1 or geoType == 2):
        print "Please input a correct geometry type!"
        exit

    # create new shapefile
    #fleSHP = shapelib.create(fleName, geo)
    for elem in geoList:
        #obj = shapelib.SHPObject(geo, 1, [[elem]])
        fleSHP.write_object(-1, obj)

    # create new dbf file
    dbfName = fleName.split(".")[0] + ".dbf"
    write_DBF(dbfName, fldList, attList)


def write_DBF(fleName, fldList, attList):
    """
    Write DBF file

    Arguments:
        fleName: text, full directory of shapefile name, including ".dbf"
        fldList: list of tuples/lists, including names, types, precisions, scale of fields, [(name, type, precision, scale), (), ... ()]
                 field type mapping is given by global variable "fldTypes".
        attList: list of tuples, including values of attributes, [(val1, val2, ... valn), (), ... ()]

    Return:
        None
    """
    #fleDBF = dbflib.create(fleName)

    # add fields
    for fld in fldList:
        fleDBF.add_field(fld[0],fldTypes[fld[1]],fld[2],fld[3])

    # add values
    i = 0
    for val in attList:
        fleDBF.write_record(i, val)
        i+=1


def write_CSV(fleName, fldList, attList):
    """
    Write CSV file

    Arguments:
        fleName: text, full directory of shapefile name, including ".csv"
        fldList: list, including names of fields
        attList: list of tuples/lists, including values of attributes, [(val1, val2, ... valn), (), ... ()]

    Return:
        None
    """
    with open(fleName, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(fldList)
        for att in attList:
            writer.writerow(att)

    f.close()



def write_TXT(fleName, fldList, attList):
    """
    Write TXT file

    Arguments:
        fleName: text, full directory of shapefile name, including ".txt"
        fldList: list/tuple, including names of fields
                 *alternatively, for general text output, fldList can be used to store summary information for the text
        attList: list of tuples/lists, including values of attributes, [(val1, val2, ... valn), (), ... ()]
                 *alternatively, for general text output, attList can be used to store detailed information for the text, one element one line

    Return:
        None
    """
    with open(fleName, 'w') as txtfile:
        txtfile.write('  '.join([str(fld) for fld in fldList])) # write header
        txtfile.write('\n')
        for row in attList:
            txtfile.write('  '.join([str(att) for att in row])) # separated by two spaces
            txtfile.write('\n')

    txtfile.close()


#--End of write file-------------------------------------------------------------------

#--3. Get subset of data---------------------------------------------------------------
def get_subset(lstFlds, dicAttr, flds=None, rows=None):
    """
    Get subset of the data based on specified fields and rows

    Arguments:
        lstFlds: list, fields in the file
        dicAttr: dictionary, key: id of feature, value: tuple of attribute values
        flds: list, list of field names
        rows: list, list of row indices

    Return:
        dicAttr_sub: dictionary, key: id of feature, value: tuple of attribute values
    """
    #fleType = fleName.split('.')[1]
    #lstFlds = []
    #dicAttr = {}
    dicAttr_sub = {}

    ## read file
    #if fleType == 'shp':
        #lstFlds = read_SHP(fleName)[1]
        #dicAttr = read_SHP(fleName)[2]
    #if fleType == 'dbf':
        #lstFlds, dicAttr = read_DBF(fleName)
    #if fleType == 'csv':
        #lstFlds, dicAttr = read_CSV(fleName)
    #if fleType == 'txt':
        #lstFlds, dicAttr = read_TXT(fleName)

    # read subset
    flds_set = []
    rows_set = []
    # check  fields
    if flds is None:
        flds_set = range(len(lstFlds))
    else:
        flds_set = [lstFlds.index(fld) for fld in flds]
    # check rows
    if rows is None:
        rows_set = dicAttr.keys()
    else:
        rows_set = rows

    for row in rows_set:
        dicAttr_sub[row] = tuple(dicAttr[row][val] for val in flds_set)

    return dicAttr_sub


#--End of Get subset of data-----------------------------------------------------------

#--Global variables----------------------------------------------------------------
read_FILE = {0: read_CSV, 1: read_DBF, 2: read_TXT, 3: read_SHP} # use one function to read different types of files
write_FILE = {0: write_CSV, 1: write_DBF, 2: write_TXT, 3: write_SHP} # use one function to write different types of files
#----------------------------------------------------------------------------------


if __name__ == '__main__':

    # Examples

    ##--read shapefile------------------------------

    #shp_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262.shp" #"E:/Research/GWR/Code/Sample data/StAndrews/StAndrewsHousePrice.shp"
    #rs = read_SHP(shp_name)
    #feat = rs[0]
    #flds = rs[1]
    #attr = rs[2]
    #print "number of features: %d" % len(feat.keys())
    #print "(x,y) of first feature: (%.6f, %.6f)" % feat[0]
    #print "number of records: %d" % len(attr.keys())
    #print "number of fields: %d" % len(flds)
    #print "name of fields: "
    #print flds
    #print "first record: "
    #print attr[0]

    ##--read dbf------------------------------------

    #dbf_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262.dbf"
    #rs = read_DBF(dbf_name)
    #print "number of records: %d" % len(rs[1].keys())
    #print "number of fields: %d" % len(rs[0])
    #print "name of fields: "
    #print rs[0]
    #print "first record: "
    #print rs[1][0]

    ##--read csv------------------------------------

    #csv_name = "E:/Research/GWR/Code/Sample data/Census/Census.csv"
    #rs = read_CSV(csv_name)
    #print "number of records: %d" % len(rs[1].keys())
    #print "number of fields: %d" % len(rs[0])
    #print "name of fields: "
    #print rs[0]
    #print "first record: "
    #print rs[1][0]

    ##--read txt------------------------------------

    #txt_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt"
    #rs = read_TXT(txt_name)
    #print "number of records: %d" % len(rs[1].keys())
    #print "number of fields: %d" % len(rs[0])
    #print "name of fields: "
    #print rs[0]
    #print "first record: "
    #print rs[1][0]

    #--get subset data-----------------------------

    dbf_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262.dbf"
    fields = ['AreaID','GEOCODE']
    rs = read_FILE[1](dbf_name)
    rs_sub = get_subset(rs[0], rs[1], fields, range(10))
    print "number of records: %d" % len(rs_sub.keys())
    print "number of fields: %d" % len(rs_sub.values()[0])
    print "All records: "
    print rs_sub

    ##--write shapefile/write dbf------------------------------

    #shp_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262_pt.shp"
    #rs = read_SHP(shp_name)
    #feat = rs[0]
    #flds = rs[1] #['AreaID', 'ORIG_FID', 'GEOCODE', 'Y', 'X', 'AREANAME']
    #attr = rs[2] # (0, 0, '08203', 17310.407256, 378906.831885, 'Tsuchiura-shi')
    #out_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262_pt_out.shp"


    #flds_ex = [[1,20,0],[1,20,0],[2,50,0],[0,20,6],[0,20,6],[2,50,0]]
    #nflds = len(flds)
    #for i in range(nflds): # populate fields
        #flds_ex[i].insert(0,flds[i])
    #write_SHP(out_name,1,feat.values(),flds_ex,attr.values())
    #print "ok"

    ##--write csv-----------------------------------------
    #dbf_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262.dbf"
    #rs = read_DBF(dbf_name)
    #out_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262_out.csv"
    #write_CSV(out_name, rs[0],rs[1].values())
    #print "ok"

    ##--write txt-----------------------------------------
    #dbf_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262.dbf"
    #rs = read_DBF(dbf_name)
    #out_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262_out.txt"
    #write_TXT(out_name, rs[0],rs[1].values())
    #print "ok"

    ##--use global variables-----------------------------------------
    #dbf_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262.dbf"
    #rs = read_FILE[1](dbf_name)
    #out_name = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/tokyomet262_out.txt"
    #write_FILE[3](out_name, rs[0],rs[1].values())
    #print "ok"
