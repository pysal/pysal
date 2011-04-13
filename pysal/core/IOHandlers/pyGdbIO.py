"""Geodatabase extension handler for PySAL."""
import os
import pysal
from pysal.core.util.sql import *

__author__ = 
__all__ = ['pgdb2shp', 'shp2pgdb', 'rookFromSql', 'queenFromSql']

#try: os.pgsql2shp, os.shp2pgsql


def getWKB(Table_repr):
    cur = session.cursor() # is there a better way to keep session in namespace?
    #cur.execute("expression to extract" Table_repr "as WKB")
    return wkb

#helper utilities to expose common use such as QueenFromSQL or RookFromSQL --
#may push to "utils" to expose at the top level

def queenFromSql(wkb):
    #read wkb into pysal via fileio
    #generate W using weights
    return W

def rookFromSql(wkb):
    #read wkb into pysal via fileio
    #generate W using weights
    return W

# Utilities
def pgdb2shp(outpath, tablename):
    """This utility function wraps PostGIS command line utility pgsql2shp.
    Unless specified, this function tells PostGIS to dump your table to the
    current directory."""
    #check for command line utility
    try:
        os.system("pgsql2shp -f "+outpath+"-P '"+self.dbpass+"' "+self.db+" "+tablename+"")
    except NameError as e:
        print "PostGIS command line utility pgsql2shp not found."


def shp2pgdb(shpname, tablename):
    """This utility function wraps PostGIS command line utility shp2pgsql."""

    try:
        os.system("shp2pgsql -D "+shpname+" "+tablename+" "+self.db+" | psql "+self.db+"")
    except NameError as e:
        print "PostGIS command line utility shp2pgsql not found."
