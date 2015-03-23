import pysal.core.Tables
import datetime
import struct
import itertools
from warnings import warn
import pysal

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ['DBF']


class DBF(pysal.core.Tables.DataTable):
    """
    PySAL DBF Reader/Writer

    This DBF handler implements the PySAL DataTable interface.

    Attributes
    ----------

    header      : list
                  A list of field names. The header is a python list of
                  strings.  Each string is a field name and field name must
                  not be longer than 10 characters.
    field_spec  : list
                  A list describing the data types of each field. It is
                  comprised of a list of tuples, each tuple describing a
                  field. The format for the tuples is ("Type",len,precision).
                  Valid Types are 'C' for characters, 'L' for bool, 'D' for
                  data, 'N' or 'F' for number.

    Examples
    --------

    >>> import pysal
    >>> dbf = pysal.open(pysal.examples.get_path('juvenile.dbf'), 'r')
    >>> dbf.header
    ['ID', 'X', 'Y']
    >>> dbf.field_spec
    [('N', 9, 0), ('N', 9, 0), ('N', 9, 0)]

    """
    FORMATS = ['dbf']
    MODES = ['r', 'w']

    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of the pysal's DBF handler.

        Arguments:
        dataPath -- str -- Path to file, including file.
        mode -- str -- 'r' or 'w'
        """
        pysal.core.Tables.DataTable.__init__(self, *args, **kwargs)
        if self.mode == 'r':
            self.f = f = open(self.dataPath, 'rb')
            numrec, lenheader = struct.unpack('<xxxxLH22x', f.read(32)) #from dbf file standards
            numfields = (lenheader - 33) // 32 #each field is 32 bytes
            self.n_records = numrec
            self.n_fields = numfields
            self.field_info = [('DeletionFlag', 'C', 1, 0)]
            record_size = 1
            fmt = 's' #each record is a string
            self._col_index = {}
            idx = 0
            for fieldno in xrange(numfields):
                name, typ, size, deci = struct.unpack(
                    '<11sc4xBB14x', f.read(32)) #again, check struct for fmt def.
                name = name.decode() #forces to unicode in 2, to str in 3
                typ = typ.decode() 
                name = name.replace('\0', '') #same as NULs, \x00
                     #eliminate NULs from string
                self._col_index[name] = (idx, record_size)
                idx += 1
                fmt += '%ds' % size #alt: str(size) + 's'
                record_size += size
                self.field_info.append((name, typ, size, deci))
            terminator = f.read(1).decode()
            assert terminator == '\r'
            self.header_size = self.f.tell()
            self.record_size = record_size
            self.record_fmt = fmt
            self.pos = 0
            self.header = [fInfo[0] for fInfo in self.field_info[1:]]
            field_spec = []
            for fname, ftype, flen, fpre in self.field_info[1:]:
                field_spec.append((ftype, flen, fpre))
            self.field_spec = field_spec

            #self.spec = [types[fInfo[0]] for fInfo in self.field_info]
        elif self.mode == 'w':
            self.f = open(self.dataPath, 'wb')
            self.header = None
            self.field_spec = None
            self.numrec = 0
            self.FIRST_WRITE = True

    def __len__(self):
        if self.mode != 'r':
            raise IOError("Invalid operation, Cannot read from a file opened in 'w' mode.")
        return self.n_records

    def seek(self, i):
        self.f.seek(self.header_size + (self.record_size * i))
        self.pos = i

    def _get_col(self, key):
        """return the column vector"""
        if key not in self._col_index:
            raise AttributeError('Field: % s does not exist in header' % key)
        prevPos = self.tell()
        idx, offset = self._col_index[key]
        typ, size, deci = self.field_spec[idx]
        gap = (self.record_size - size)
        f = self.f
        f.seek(self.header_size + offset)
        col = [0] * self.n_records
        for i in xrange(self.n_records):
            value = f.read(size)
            value = value.decode()
            f.seek(gap, 1)
            if typ == 'N':
                value = value.replace('\0', '').lstrip()
                if value == '':
                    value = pysal.MISSINGVALUE
                elif deci:
                    try:
                        value = float(value)
                    except ValueError:
                        value = pysal.MISSINGVALUE
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        value = pysal.MISSINGVALUE
            elif typ == 'D':
                try:
                    y, m, d = int(value[:4]), int(value[4:6]), int(value[6:8])
                    value = datetime.date(y, m, d)
                except ValueError:
                    value = pysal.MISSINGVALUE
            elif typ == 'L':
                value = (value in 'YyTt' and 'T') or (
                    value in 'NnFf' and 'F') or '?'
            elif typ == 'F':
                value = value.replace('\0', '').lstrip()
                if value == '':
                    value = pysal.MISSINGVALUE
                else:
                    value = float(value)
            if isinstance(value, str) or isinstance(value, unicode):
                value = value.rstrip()
            col[i] = value
        self.seek(prevPos)
        return col

    def read_record(self, i):
        self.seek(i)
        rec = list(struct.unpack(
            self.record_fmt, self.f.read(self.record_size)))
        rec = [entry.decode() for entry in rec]
        if rec[0] != ' ':
            return self.read_record(i + 1)
        result = []
        for (name, typ, size, deci), value in itertools.izip(self.field_info, rec):
            if name == 'DeletionFlag':
                continue
            if typ == 'N':
                value = value.replace('\0', '').lstrip()
                if value == '':
                    value = pysal.MISSINGVALUE
                elif deci:
                    try:
                        value = float(value)
                    except ValueError:
                        value = pysal.MISSINGVALUE
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        value = pysal.MISSINGVALUE
            elif typ == 'D':
                try:
                    y, m, d = int(value[:4]), int(value[4:6]), int(value[6:8])
                    value = datetime.date(y, m, d)
                except ValueError:
                    #value = datetime.date.min#NULL Date: See issue 114
                    value = pysal.MISSINGVALUE
            elif typ == 'L':
                value = (value in 'YyTt' and 'T') or (
                    value in 'NnFf' and 'F') or '?'
            elif typ == 'F':
                value = value.replace('\0', '').lstrip()
                if value == '':
                    value = pysal.MISSINGVALUE
                else:
                    value = float(value)
            if isinstance(value, str) or isinstance(value, unicode):
                value = value.rstrip()
            result.append(value)
        return result

    def _read(self):
        if self.mode != 'r':
            raise IOError("Invalid operation, Cannot read from a file opened in 'w' mode.")
        if self.pos < len(self):
            rec = self.read_record(self.pos)
            self.pos += 1
            return rec
        else:
            return None

    def write(self, obj):
        self._complain_ifclosed(self.closed)
        if self.mode != 'w':
            raise IOError("Invalid operation, Cannot write to a file opened in 'r' mode.")
        if self.FIRST_WRITE:
            self._firstWrite(obj)
        if len(obj) != len(self.header):
            raise TypeError("Rows must contains %d fields" % len(self.header))
        self.numrec += 1
        self.f.write(' '.encode())                        # deletion flag
        for (typ, size, deci), value in itertools.izip(self.field_spec, obj):
            if value is None:
                if typ == 'C':
                    value = ' ' * size
                else:
                    value = '\0' * size
            elif typ == "N" or typ == "F":
                v = str(value).rjust(size, ' ')
                #if len(v) == size:
                #    value = v
                #else:
                value = (("%" + "%d.%d" % (size, deci) + "f") % (value))[:size]
            elif typ == 'D':
                value = value.strftime('%Y%m%d')
            elif typ == 'L':
                value = str(value)[0].upper()
            else:
                value = str(value)[:size].ljust(size, ' ')
            try:
                assert len(value) == size
            except:
                print value, len(value), size
                raise
            self.f.write(value.encode())
            self.pos += 1

    def flush(self):
        self._complain_ifclosed(self.closed)
        self._writeHeader()
        self.f.flush()

    def close(self):
        if self.mode == 'w':
            self.flush()
            # End of file
            self.f.write('\x1A'.encode())
        self.f.close()
        pysal.core.Tables.DataTable.close(self)

    def _firstWrite(self, obj):
        if not self.header:
            raise IOError("No header, DBF files require a header.")
        if not self.field_spec:
            raise IOError("No field_spec, DBF files require a specification.")
        self._writeHeader()
        self.FIRST_WRITE = False

    def _writeHeader(self):
        """ Modified from: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/362715 """
        POS = self.f.tell()
        self.f.seek(0)
        ver = 3
        now = datetime.datetime.now()
        yr, mon, day = now.year - 1900, now.month, now.day
        numrec = self.numrec
        numfields = len(self.header)
        lenheader = numfields * 32 + 33
        lenrecord = sum(field[1] for field in self.field_spec) + 1
        hdr = struct.pack('<BBBBLHH20x', ver, yr, mon, day, numrec,
                          lenheader, lenrecord)
        self.f.write(hdr)
        # field specs
        for name, (typ, size, deci) in itertools.izip(self.header, self.field_spec):
            typ = typ.encode()
            name = name.ljust(11, '\x00')
            name = name.encode()
            fld = struct.pack('<11sc4xBB14x', name, typ, size, deci)
            self.f.write(fld)
        # terminator
        term = '\r'.encode()
        self.f.write(term)
        if self.f.tell() != POS and not self.FIRST_WRITE:
            self.f.seek(POS)

if __name__ == '__main__':
    import pysal
    file_name = pysal.examples.get_path("10740.dbf")
    f = pysal.open(file_name, 'r')
    newDB = pysal.open('copy.dbf', 'w')
    newDB.header = f.header
    newDB.field_spec = f.field_spec
    print f.header
    for row in f:
        print row
        newDB.write(row)
    newDB.close()
    copy = pysal.open('copy.dbf', 'r')
    f.seek(0)
    print "HEADER: ", copy.header == f.header
    print "SPEC: ", copy.field_spec == f.field_spec
    print "DATA: ", list(copy) == list(f)
