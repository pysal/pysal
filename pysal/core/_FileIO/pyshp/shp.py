"""
A library for reading and writing to 'shapefiles'.
More specifically, shp/shx and dbf components.
Not to be used without permission.

Contact: 

Andrew Winslow
GeoDa Center for Geospatial Analysis
Arizona State University
Tempe, AZ
Andrew.Winslow@asu.edu
"""

import pysal.core._shp as C_Net

def create_dbf_file(filename, fields):
    """
    Creates a dbf file with name _filename_ and the 
    field names and types specified by _fields_ as a list like: 
    [('X', 1.0), ('POP', 52), ('NAME', 'JONES')] 
    """
    C_Net.create_dbf_file(filename, fields)
    return Dbffile(filename)


class Dbffile(object):

   
    def __init__(self, filename):
	self.filename = filename

    def read(self, start_rec, end_rec):
	"""
	Returns a list of dbf records between indices _start_rec_ and _end_rec_, inclusive.
	Each record is a list of elements in the same order as the header specified by
	field_names().
	"""
	return C_Net.read_dbf_records(self.filename, start_rec, end_rec)

    def size(self):
	"""
        Returns the number of records in the dbf file.
        """
	return C_Net.read_dbf_record_count(self.filename)

    def field_names(self):	
	"""
	Returns a list of the names of the fields in the dbf file
        in the order they occur.
        """
	return C_Net.read_dbf_header(self.filename)

    def write(self, records):
	"""
	Writes a list of dbf records to the end of existing records.
	Each record should be a list of elements of the correct type and order
	as seen in field_names() and field_types().
	"""
	C_Net.write_dbf_records(self.filename, records)	

    
def create_shpshx_file(filename, type):
    """
    Creates a shp/shx file with name _filename_ and type one of:
    'POINT', 'ARC', 'POLYGON', 'MULTIPOINT'. 
    """
    if type not in ['POINT', 'POINTZ', 'POINTM',
                    'ARC', 'ARCZ', 'ARCM',
                    'POLYGON', 'POLYGONZ', 'POLYGONM',
                    'MULTIPOINT', 'MULTIPOINTZ', 'MULTIPOINTM',
                    'MULTIPATCH']:
	raise Exception, 'Attempt to create shp/shx file of invalid type'
    C_Net.create_shpshx_file(filename, type)
    return Shpfile(filename)


class Shpfile(object):

    def __init__(self, filename):
	self.filename = filename

    def size(self):
	"""
        Returns the number of records in the shp/shx file.
        """
	return C_Net.read_shpshx_record_count(self.filename)

    def type(self):	
        """
        Returns the type of the shp/shx file as a string.
        """
	return C_Net.read_shpshx_type(self.filename)
    
    def read(self, start_rec, end_rec):
	"""
	Returns the set of records between 1-based indices _start_rec_ and _end_rec_, inclusive.

	Each record has the form:

	[type, record_id, pan_parts, vertices, mins, maxs]

	Examples:

	['POINT', 12, [], [(17.2, 16.5)], (17.2, 16.5, 0, 0), (17.2, 16.5, 0, 0)]
	['ARC', 6, [], [(4.5, 6.5), (9.5, 10.2)], (4.5, 6.5, 0, 0), (9.5, 10.2, 0, 0)]
	['POLYGON', 42, [(0, 'MULTIPATCH'), (2, 'MULTIPATCH')], [(2, 3), (4, 1), (4, 2), (2,2)], (2,2,0,0), (4,3,0,0)]
	"""
	return C_Net.read_shpshx_records(self.filename, start_rec, end_rec)

    def write(self, records):
	"""
	Writes a set of records _records_ to the shp/shx file. Each record either 
	replaces a null	record in the existing shp/shx file or appends a new 
	record to the end. Each record has the form:

	[type, record_id, pan_parts, vertices, mins, maxs]

	Examples:
	
	['POINT', 12, [], [(17.2, 16.5)], (17.2, 16.5, 0, 0), (17.2, 16.5, 0, 0)]
	['ARC', 6, [], [(4.5, 6.5), (9.5, 10.2)], (4.5, 6.5, 0, 0), (9.5, 10.2, 0, 0)]
	['POLYGON', 42, [(0, 'MULTIPATCH'), (2, 'MULTIPATCH')], [(2, 3), (4, 1), (4, 2), (2,2)], (2,2,0,0), (4,3,0,0)]

	Note: record_id, mins and maxs are currently not used when writing. They are present
	only to maintain consistency with read() output. 
	"""
	return C_Net.write_shpshx_records(self.filename, records)	

    def bounding_box(self):
        """
        Returns a 4-tuple (left, right, lower, upper) of the bounding box
        of the interval of records from start_rec to end_rec 1-based and inclusive.
        """
        bb = (1e300, -1e300, 1e300, -1e300)
        size = self.size()
        for rec_intv in zip(range(1, size, 50000), range(50000, size + 50000, 50000)):
            recs = self.read(rec_intv[0], rec_intv[1])
            bb = (min(bb[0], min([r[4][0] for r in recs])), 
                  max(bb[1], max([r[5][0] for r in recs])), 
                  min(bb[2], min([r[4][1] for r in recs])),
                  max(bb[3], max([r[5][1] for r in recs])))
        return bb
  
