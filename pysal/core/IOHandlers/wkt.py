import pysal.core.FileIO as FileIO
from pysal import cg
import re

__author__ = "Charles R Schmidt <Charles.R.Schmidt@asu.edu>"
__all__ = ['WKTReader', 'WKTParser']
#####################################################################
## ToDo: Add Well-Known-Binary support...
##       * WKB spec:
##  http://webhelp.esri.com/arcgisserver/9.3/dotNet/index.htm#geodatabases/the_ogc_103951442.htm 
##
##
#####################################################################



class WKTReader(FileIO.FileIO):
    MODES = ['r']
    FORMATS = ['wkt']
    def __init__(self,*args,**kwargs):
        FileIO.FileIO.__init__(self,*args,**kwargs)
        self.__idx = {}
        self.__pos = 0
        self.__open()
    def open(self):
        self.__open()
    def __open(self):
        self.dataObj = open(self.dataPath,self.mode)
        self.wkt = WKTParser()
    def _read(self):
        FileIO.FileIO._complain_ifclosed(self.closed)
        if self.__pos not in self.__idx:
            self.__idx[self.__pos] = self.dataObj.tell()
        line = self.dataObj.readline()
        if line:
            shape = self.wkt.fromWKT(line)
            shape.id = self.pos
            self.__pos += 1
            self.pos += 1
            return shape
        else:
            self.seek(0)
            return None
    def seek(self,n):
        FileIO.FileIO.seek(self,n)
        pos = self.pos
        if pos in self.__idx:
            self.dataObj.seek(self.__idx[pos])
            self.__pos = pos
        else:
            while pos not in self.__idx:
                s = self._read()
                if not s:
                    raise IndexError, "%d not in range(0,%d)"%(pos,max(self.__idx.keys()))
            self.pos = pos
            self.__pos = pos
            self.dataObj.seek(self.__idx[pos])
    def close(self):
        self.dataObj.close()
        FileIO.FileIO.close(self)
        

class WKTParser:
    """ Class to represent OGC WKT, supports reading and writing
        Modified from...
        # URL: http://dev.openlayers.org/releases/OpenLayers-2.7/lib/OpenLayers/Format/WKT.js
        #Reg Ex Strings copied from OpenLayers.Format.WKT

    Example
    -------
    >>> from pysal.core.IOHandlers import wkt
    >>> import pysal
    
    Create some Well-Known Text objects

    >>> p = 'POLYGON((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2))'
    >>> pt = 'POINT(6 10)'
    >>> l = 'LINESTRING(3 4,10 50,20 25)'
    
    Instantiate the parser

    >>> wkt = WKTParser()

    Inspect our WKT polygon 

    >>> wkt(p).parts
    [[(1.0, 1.0), (1.0, 5.0), (5.0, 5.0), (5.0, 1.0), (1.0, 1.0)], [(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]
    >>> wkt(p).centroid
    (2.9705882352941173, 2.9705882352941173)
    >>> wkt(p).area
    17.0

    Inspect pt, our WKT point object

    >>> wkt(pt)
    (6.0, 10.0)

    Inspect our WKT linestring

    >>> wkt(l).len
    73.455384532199886
    >>> wkt(l).parts
    [[(3.0, 4.0), (10.0, 50.0), (20.0, 25.0)]]

    Read in WKT from a file

    >>> f = pysal.open('../../examples/sample.wkt')
    >>> f.mode
    'r'
    >>> f.header
    []

    See local doctest output for the items not tested...

    """
    regExes = { 'typeStr': re.compile('^\s*(\w+)\s*\(\s*(.*)\s*\)\s*$'),
        'spaces': re.compile('\s+'),
        'parenComma': re.compile('\)\s*,\s*\('),
        'doubleParenComma': re.compile('\)\s*\)\s*,\s*\(\s*\('),  # can't use {2} here
        'trimParens': re.compile('^\s*\(?(.*?)\)?\s*$') }
    def __init__(self):
        self.parsers = p = {}
        p['point'] = self.Point
        p['linestring'] = self.LineString
        p['polygon'] = self.Polygon
    def Point(self,geoStr):
        coords = self.regExes['spaces'].split(geoStr.strip())
        return cg.Point((coords[0],coords[1]))
    def LineString(self,geoStr):
        points = geoStr.strip().split(',')
        points = map(self.Point,points)
        return cg.Chain(points)
    def Polygon(self,geoStr):
        rings = self.regExes['parenComma'].split(geoStr.strip())
        for i,ring in enumerate(rings):
            ring = self.regExes['trimParens'].match(ring).groups()[0]
            ring = self.LineString(ring).vertices
            rings[i] = ring
        return cg.Polygon(rings)
    def fromWKT(self,wkt):
        matches = self.regExes['typeStr'].match(wkt)
        if matches:
            geoType,geoStr = matches.groups()
            geoType = geoType.lower()
            try:
                return self.parsers[geoType](geoStr)
            except KeyError:
                raise NotImplementedError, "Unsupported WKT Type: %s"%geoType
        else:
            return None
    __call__ = fromWKT
if __name__=='__main__':
    p = 'POLYGON((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2))'
    pt = 'POINT(6 10)'
    l = 'LINESTRING(3 4,10 50,20 25)'
    wktExamples = ['POINT(6 10)',
            'LINESTRING(3 4,10 50,20 25)',
            'POLYGON((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2))',
            'MULTIPOINT(3.5 5.6,4.8 10.5)',
            'MULTILINESTRING((3 4,10 50,20 25),(-5 -8,-10 -8,-15 -4))',
            'MULTIPOLYGON(((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2)),((3 3,6 2,6 4,3 3)))',
            'GEOMETRYCOLLECTION(POINT(4 6),LINESTRING(4 6,7 10))',
            'POINT ZM (1 1 5 60)',
            'POINT M (1 1 80)',
            'POINT EMPTY',
            'MULTIPOLYGON EMPTY']
    wkt = WKTParser()
    
def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == '__main__':
    _test()

