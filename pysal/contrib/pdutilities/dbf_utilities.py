"""miscellaneous file manipulation utilities

"""

import numpy as np
import pysal as ps
import pandas as pd

def check_dups(li):
    """checks duplicates in list of ID values
       ID values must be read in as a list

       __author__  = "Luc Anselin <luc.anselin@asu.edu> "
       
       Arguments
       ---------
       li      : list of ID values
       
       Returns
       -------
       a list with the duplicate IDs
    """
    return list(set([x for x in li if li.count(x) > 1]))
    
    
def dbfdups(dbfpath,idvar):
    """checks duplicates in a dBase file
       ID variable must be specified correctly

       __author__  = "Luc Anselin <luc.anselin@asu.edu> " 
       
       Arguments
       ---------
       dbfpath  : file path to dBase file
       idvar    : ID variable in dBase file
       
       Returns
       -------
       a list with the duplicate IDs
    """
    db = ps.open(dbfpath,'r')
    li = db.by_col(idvar)
    return list(set([x for x in li if li.count(x) > 1]))   
    

def df2dbf(df, dbf_path, my_specs=None):
    '''
    Convert a pandas.DataFrame into a dbf.

    __author__  = "Dani Arribas-Bel <darribas@asu.edu>, Luc Anselin <luc.anselin@asu.edu>"
    ...

    Arguments
    ---------
    df          : DataFrame
                  Pandas dataframe object to be entirely written out to a dbf
    dbf_path    : str
                  Path to the output dbf. It is also returned by the function
    my_specs    : list
                  List with the field_specs to use for each column.
                  Defaults to None and applies the following scheme:
                    * int: ('N', 14, 0) - for all ints
                    * float: ('N', 14, 14) - for all floats
                    * str: ('C', 14, 0) - for string, object and category
                  with all variants for different type sizes
                  
    Note: use of dtypes.name may not be fully robust, but preferred apprach of using
    isinstance seems too clumsy
    '''
    if my_specs:
        specs = my_specs
    else:
        """
        type2spec = {int: ('N', 20, 0),
                     np.int64: ('N', 20, 0),
                     np.int32: ('N', 20, 0),
                     np.int16: ('N', 20, 0),
                     np.int8: ('N', 20, 0),
                     float: ('N', 36, 15),
                     np.float64: ('N', 36, 15),
                     np.float32: ('N', 36, 15),
                     str: ('C', 14, 0)
                     }
        types = [type(df[i].iloc[0]) for i in df.columns]
        """
        # new approach using dtypes.name to avoid numpy name issue in type
        type2spec = {'int': ('N', 20, 0),
                     'int8': ('N', 20, 0),
                     'int16': ('N', 20, 0),
                     'int32': ('N', 20, 0),
                     'int64': ('N', 20, 0),
                     'float': ('N', 36, 15),
                     'float32': ('N', 36, 15),
                     'float64': ('N', 36, 15),
                     'str': ('C', 14, 0),
                     'object': ('C', 14, 0),
                     'category': ('C', 14, 0)
                     }
        types = [df[i].dtypes.name for i in df.columns]
        specs = [type2spec[t] for t in types]
    db = ps.open(dbf_path, 'w')
    db.header = list(df.columns)
    db.field_spec = specs
    for i, row in df.T.iteritems():
        db.write(row)
    db.close()
    return dbf_path

def dbf2df(dbf_path, index=None, cols=False, incl_index=False):
    '''
    Read a dbf file as a pandas.DataFrame, optionally selecting the index
    variable and which columns are to be loaded.

    __author__  = "Dani Arribas-Bel <darribas@asu.edu> "
    ...

    Arguments
    ---------
    dbf_path    : str
                  Path to the DBF file to be read
    index       : str
                  Name of the column to be used as the index of the DataFrame
    cols        : list
                  List with the names of the columns to be read into the
                  DataFrame. Defaults to False, which reads the whole dbf
    incl_index  : Boolean
                  If True index is included in the DataFrame as a
                  column too. Defaults to False

    Returns
    -------
    df          : DataFrame
                  pandas.DataFrame object created
    '''
    db = ps.open(dbf_path)
    if cols:
        if incl_index:
            cols.append(index)
        vars_to_read = cols
    else:
        vars_to_read = db.header
    data = dict([(var, db.by_col(var)) for var in vars_to_read])
    if index:
        index = db.by_col(index)
        db.close()
        return pd.DataFrame(data, index=index, columns=vars_to_read)
    else:
        db.close()
        return pd.DataFrame(data,columns=vars_to_read)

    
def dbfjoin(dbf1_path,dbf2_path,out_path,joinkey1,joinkey2):
    '''
    Wrapper function to merge two dbf files into a new dbf file. 

    __author__  = "Luc Anselin <luc.anselin@asu.edu> "
    
    Uses dbf2df and df2dbf to read and write the dbf files into a pandas
    DataFrame. Uses all default settings for dbf2df and df2dbf (see docs
    for specifics).
    ...

    Arguments
    ---------

    dbf1_path   : str
                  Path to the first (left) dbf file
    dbf2_path   : str
                  Path to the second (right) dbf file
    out_path    : str
                  Path to the output dbf file (returned by the function)
    joinkey1    : str
                  Variable name for the key in the first dbf. Must be specified.
                  Key must take unique values.
    joinkey2    : str
                  Variable name for the key in the second dbf. Must be specified.
                  Key must take unique values.
                  
    Returns
    -------
    dbfpath     : path to output file
    
    '''
    df1 = dbf2df(dbf1_path,index=joinkey1)
    df2 = dbf2df(dbf2_path,index=joinkey2)
    dfbig = pd.merge(df1,df2,left_on=joinkey1,right_on=joinkey2,sort=False)
    dp = df2dbf(dfbig,out_path)
    return dp


def dta2dbf(dta_path,dbf_path):
    """
    Wrapper function to convert a stata dta file into a dbf file. 

    __author__  = "Luc Anselin <luc.anselin@asu.edu> "
    
    Uses df2dbf to write the dbf files from a pandas
    DataFrame. Uses all default settings for df2dbf (see docs
    for specifics).
    ...

    Arguments
    ---------

    dta_path    : str
                  Path to the Stata dta file
    dbf_path    : str
                  Path to the output dbf file
                  
    Returns
    -------
    dbf_path    : path to output file
    """
    
    db = pd.read_stata(dta_path)
    dp = df2dbf(db,dbf_path)
    return dp    
