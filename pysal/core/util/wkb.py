"""
Load WKB into pysal shapes.

Where pysal shapes support multiple parts, 
"MULTI"type shapes will be converted to a single multi-part shape:
    MULTIPOLYGON -> Polygon
    MULTILINESTRING -> Chain

Otherwise a list of shapes will be returned:
    MULTIPOINT -> [pt0, ..., ptN]

Some concepts aren't well supported by pysal shapes.
For example:
    wkt = 'MULTIPOLYGON EMPTY' -> '\x01   \x06\x00\x00\x00   \x00\x00\x00\x00'
                                  |  <  | WKBMultiPolygon |    0 parts      |
    pysal.cg.Polygon does not support 0 part polygons.
    None is returned in this case.

REFERENCE MATERIAL:
SOURCE: http://webhelp.esri.com/arcgisserver/9.3/dotNet/index.htm#geodatabases/the_ogc_103951442.htm

 Basic Type definitions
 byte : 1 byte
 uint32 : 32 bit unsigned integer  (4 bytes)
 double : double precision number (8 bytes)

 Building Blocks : Point, LinearRing


"""
from cStringIO import StringIO
from pysal import cg
import sys
import array
import struct

__author__ = 'Charles R Schmidt <schmidtc@gmail.com>'
__all__ = ['loads']

"""
enum wkbByteOrder {
    wkbXDR = 0,              Big Endian
    wkbNDR = 1               Little Endian
};
"""
DEFAULT_ENDIAN = '<' if sys.byteorder == 'little' else '>'
ENDIAN = {'\x00': '>', '\x01': '<'}

def load_ring_little(dat):
    """
    LinearRing   {
        uint32  numPoints;
        Point   points[numPoints];
    }
    """
    npts = struct.unpack('<I', dat.read(4))[0]
    xy = struct.unpack('<%dd'%(npts*2), dat.read(npts*2*8))
    return [cg.Point(xy[i:i+2]) for i in xrange(0,npts*2,2)]
    
def load_ring_big(dat):
    npts = struct.unpack('>I', dat.read(4))[0]
    xy = struct.unpack('>%dd'%(npts*2), dat.read(npts*2*8))
    return [cg.Point(xy[i:i+2]) for i in xrange(0,npts*2,2)]
    

def loads(s):
    """
    WKBGeometry  {
        union {
            WKBPoint                        point;
            WKBLineString               linestring;
            WKBPolygon                  polygon;
            WKBGeometryCollection   collection;
            WKBMultiPoint               mpoint;
            WKBMultiLineString      mlinestring;
            WKBMultiPolygon         mpolygon;
        }
    };
    """
    # To allow recursive calls, read only the bytes we need.
    if hasattr(s, 'read'):
        dat = s
    else:
        dat = StringIO(s)
    endian = ENDIAN[dat.read(1)]
    typ = struct.unpack('I', dat.read(4))[0]
    if typ == 1:
        """
        WKBPoint {
            byte                byteOrder;
            uint32          wkbType;                 1
            Point               point;
        }
        Point {
            double x;
            double y;
        };
        """
        x,y = struct.unpack(endian+'dd', dat.read(16))
        return cg.Point((x,y))
    elif typ == 2:
        """
        WKBLineString {
            byte                byteOrder;
            uint32          wkbType;                     2
            uint32          numPoints;
            Point               points[numPoints];
        }
        """
        n = struct.unpack(endian+'I', dat.read(4))[0]
        xy = struct.unpack(endian+'%dd'%(n*2), dat.read(n*2*8))
        return cg.Chain([cg.Point(xy[i:i+2]) for i in xrange(0,n*2,2)])
    elif typ == 3:
        """
        WKBPolygon  {
            byte                byteOrder;
            uint32          wkbType;                     3
            uint32          numRings;
            LinearRing      rings[numRings];
        }

        WKBPolygon has exactly 1 outer ring and n holes.
            multipart Polygons are NOT support by WKBPolygon.
        """
        nrings = struct.unpack(endian+'I', dat.read(4))[0]
        load_ring = load_ring_little if endian == '<' else load_ring_big
        rings = [load_ring(dat) for _ in xrange(nrings)]
        return cg.Polygon(rings[0], rings[1:])
    elif typ == 4:
        """
        WKBMultiPoint   {
            byte                byteOrder;
            uint32          wkbType;                     4
            uint32          num_wkbPoints;
            WKBPoint            WKBPoints[num_wkbPoints];
        }
        """
        npts = struct.unpack(endian+'I', dat.read(4))[0]
        return [loads(dat) for _ in xrange(npts)]
    elif typ == 5:
        """
        WKBMultiLineString  {
            byte                byteOrder;
            uint32          wkbType;                     5
            uint32          num_wkbLineStrings;
            WKBLineString   WKBLineStrings[num_wkbLineStrings];
        }
        """
        nparts = struct.unpack(endian+'I', dat.read(4))[0]
        chains = [loads(dat) for _ in xrange(nparts)]
        return cg.Chain(sum([c.parts for c in chains],[]))
    elif typ == 6:
        """
        wkbMultiPolygon {               
            byte                byteOrder;                              
            uint32          wkbType;                     6
            uint32          num_wkbPolygons;
            WKBPolygon      wkbPolygons[num_wkbPolygons];
        }

        """
        npolys = struct.unpack(endian+'I', dat.read(4))[0]
        polys = [loads(dat) for _ in xrange(npolys)]
        parts = sum([p.parts for p in polys], [])
        holes = sum([p.holes for p in polys if p.holes[0]], [])
        # MULTIPOLYGON EMPTY, isn't well supported by pysal shape types.
        if not parts:
            return None
        return cg.Polygon(parts, holes)
    elif typ == 7:
        """
        WKBGeometryCollection {
            byte                byte_order;
            uint32          wkbType;                     7
            uint32          num_wkbGeometries;
            WKBGeometry     wkbGeometries[num_wkbGeometries]
        }
        """
        ngeoms = struct.unpack(endian+'I', dat.read(4))[0]
        return [loads(dat) for _ in xrange(ngeoms)]
        
    raise TypeError('Type (%d) is unknown or unsupported.'%typ)



if __name__ == '__main__':
    # TODO: Refactor below into Unit Tests
    wktExamples = ['POINT(6 10)',
                   'LINESTRING(3 4,10 50,20 25)',
                   'POLYGON((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2))',
                   'MULTIPOINT(3.5 5.6,4.8 10.5)',
                   'MULTILINESTRING((3 4,10 50,20 25),(-5 -8,-10 -8,-15 -4))',
                    # This MULTIPOLYGON is not valid, the 2nd shell instects the 1st.
                   #'MULTIPOLYGON(((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2)),((3 3,6 2,6 4,3 3)))',
                   'MULTIPOLYGON(((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2)),((5 3,6 2,6 4,5 3)))',
                   'GEOMETRYCOLLECTION(POINT(4 6),LINESTRING(4 6,7 10))',
                   #'POINT ZM (1 1 5 60)',  <-- ZM is not supported by WKB ?
                   #'POINT M (1 1 80)',     <-- M is not supported by WKB ?
                   #'POINT EMPTY',          <-- NOT SUPPORT
                   'MULTIPOLYGON EMPTY']
    # shapely only used for testing.
    try:
        import shapely.wkt, shapely.geometry
        from pysal.contrib.shapely_ext import to_wkb
    except ImportError:
        print "shapely is used to test this module."
        raise
    for example in wktExamples:
        print example
        shape0= shapely.wkt.loads(example)
        shape1 = loads(shape0.to_wkb())
        if example.startswith('MULTIPOINT'):
            shape2 = shapely.geometry.asMultiPoint(shape1)
        elif example.startswith('GEOMETRYCOLLECTION'):
            shape2 = shapely.geometry.collection.GeometryCollection(map(shapely.geometry.asShape,shape1))
        elif example == 'MULTIPOLYGON EMPTY':
            #Skip Test
            shape2 = None
        else:
            shape2 = shapely.geometry.asShape(shape1)


        print shape1
        if shape2:
            assert shape0.equals(shape2)
            print shape0.equals(shape2)
        else:
            print "Skip"

        print ""
