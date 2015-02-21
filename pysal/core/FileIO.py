"""
FileIO: Module for reading and writing various file types in a Pythonic way.
This module should not be used directly, instead...
import pysal.core.FileIO as FileIO
Readers and Writers will mimic python file objects.
.seek(n) seeks to the n'th object
.read(n) reads n objects, default == all
.next() reads the next object
"""

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"

__all__ = ['FileIO']
import os.path
import struct
from warnings import warn
import pysal


class FileIO_MetaCls(type):
    """
    This Meta Class is instantiated when the class is first defined.
    All subclasses of FileIO also inherit this meta class, which registers their abilities with the FileIO registry.
    Subclasses must contain FORMATS and MODES (both are type(list))
    """
    def __new__(mcs, name, bases, dict):
        cls = type.__new__(mcs, name, bases, dict)
        if name != 'FileIO' and name != 'DataTable':
            if "FORMATS" in dict and "MODES" in dict:
                #print "Registering %s with FileIO.\n\tFormats: %r\n\tModes: %r"%(name,dict['FORMATS'],dict['MODES'])
                FileIO._register(cls, dict['FORMATS'], dict['MODES'])
            else:
                raise TypeError("FileIO subclasses must have FORMATS and MODES defined")
        return cls


class FileIO(object):  # should be a type?
    """
    How this works:
    FileIO.open(\*args) == FileIO(\*args)
    When creating a new instance of FileIO the .__new__ method intercepts
    .__new__ parses the filename to determine the fileType
    next, .__registry and checked for that type.
    Each type supports one or more modes ['r','w','a',etc]
    If we support the type and mode, an instance of the appropriate handler
    is created and returned.
    All handlers must inherit from this class, and by doing so are automatically
    added to the .__registry and are forced to conform to the prescribed API.
    The metaclass takes cares of the registration by parsing the class definition.
    It doesn't make much sense to treat weights in the same way as shapefiles and dbfs,
    ....for now we'll just return an instance of W on mode='r'
    .... on mode='w', .write will expect an instance of W
    """
    __metaclass__ = FileIO_MetaCls
    __registry = {}  # {'shp':{'r':[OGRshpReader,pysalShpReader]}}

    def __new__(cls, dataPath='', mode='r', dataFormat=None):
        """
        Intercepts the instantiation of FileIO and dispatches to the correct handler
        If no suitable handler is found a python file object is returned.
        """
        if cls is FileIO:
            try:
                newCls = object.__new__(cls.__registry[cls.getType(dataPath,
                                                                   mode, dataFormat)][mode][0])
            except KeyError:
                return open(dataPath, mode)
            return newCls
        else:
            return object.__new__(cls)

    @staticmethod
    def getType(dataPath, mode, dataFormat=None):
        """Parse the dataPath and return the data type"""
        if dataFormat:
            ext = dataFormat
        else:
            ext = os.path.splitext(dataPath)[1]
            ext = ext.replace('.', '')
            ext = ext.lower()
        if ext == 'txt':
            f = open(dataPath, 'r')
            l1 = f.readline()
            l2 = f.readline()
            if ext == 'txt':
                try:
                    n, k = l1.split(',')
                    n, k = int(n), int(k)
                    fields = l2.split(',')
                    assert len(fields) == k
                    return 'geoda_txt'
                except:
                    return ext
        return ext

    @classmethod
    def _register(cls, parser, formats, modes):
        """ This method is called automatically via the MetaClass of FileIO subclasses
        This should be private, but that hides it from the MetaClass
        """
        assert cls is FileIO
        for format in formats:
            if not format in cls.__registry:
                cls.__registry[format] = {}
            for mode in modes:
                if not mode in cls.__registry[format]:
                    cls.__registry[format][mode] = []
                cls.__registry[format][mode].append(parser)
        #cls.check()

    @classmethod
    def check(cls):
        """ Prints the contents of the registry """
        print "PySAL File I/O understands the following file extensions:"
        for key, val in cls.__registry.iteritems():
            print "Ext: '.%s', Modes: %r" % (key, val.keys())

    @classmethod
    def open(cls, *args, **kwargs):
        """ Alias for FileIO() """
        return cls(*args, **kwargs)

    class _By_Row:
        def __init__(self, parent):
            self.p = parent

        def __repr__(self):
            if not self.p.ids:
                return "keys: range(0,n)"
            else:
                return "keys: " + self.p.ids.keys().__repr__()

        def __getitem__(self, key):
            if type(key) == list:
                r = []
                if self.p.ids:
                    for k in key:
                        r.append(self.p.get(self.p.ids[k]))
                else:
                    for k in key:
                        r.append(self.p.get(k))
                return r
            if self.p.ids:
                return self.p.get(self.p.ids[key])
            else:
                return self.p.get(key)
        __call__ = __getitem__

    def __init__(self, dataPath='', mode='r', dataFormat=None):
        self.dataPath = dataPath
        self.dataObj = ''
        self.mode = mode
        #pos Should ALWAYS be in the range 0,...,n
        #for custom IDs set the ids property.
        self.pos = 0
        self.__ids = None  # {'id':n}
        self.__rIds = None
        self.closed = False
        self._spec = []
        self.header = []

    def __getitem__(self, key):
        return self.by_row.__getitem__(key)

    @property
    def by_row(self):
        return self._By_Row(self)

    def __getIds(self):
        return self.__ids

    def __setIds(self, ids):
        """ Property Method for .ids
        Takes a list of ids and maps then to a 0 based index
        Need to provide a method to set ID's based on a fieldName
        preferably without reading the whole file.
        """
        if isinstance(ids, list):
            try:
                assert len(ids) == len(set(ids))
            except AssertionError:
                raise KeyError("IDs must be unique")
            # keys: ID values: i
            self.__ids = {}
            # keys: i values: ID
            self.__rIds = {}
            for i, id in enumerate(ids):
                self.__ids[id] = i
                self.__rIds[i] = id
        elif isinstance(ids, dict):
            self.__ids = ids
            self.__rIds = {}
            for id, n in ids.iteritems():
                self.__rIds[n] = id
        elif not ids:
            self.__ids = None
            self.__rIds = None
    ids = property(fget=__getIds, fset=__setIds)

    @property
    def rIds(self):
        return self.__rIds

    def __iter__(self):
        self.seek(0)
        return self

    @staticmethod
    def _complain_ifclosed(closed):
        """ from StringIO """
        if closed:
            raise ValueError("I/O operation on closed file")

    def cast(self, key, typ):
        """cast key as typ"""
        if key in self.header:
            if not self._spec:
                self._spec = [lambda x:x for k in self.header]
            if typ is None:
                self._spec[self.header.index(key)] = lambda x: x
            else:
                try:
                    assert hasattr(typ, '__call__')
                    self._spec[self.header.index(key)] = typ
                except AssertionError:
                    raise TypeError('Cast Objects must be callable')
        else:
            raise KeyError("%s" % key)

    def _cast(self, row):
        if self._spec and row:
            try:
                return [f(v) for f, v in zip(self._spec, row)]
            except ValueError:
                r = []
                for f, v in zip(self._spec, row):
                    try:
                        if not v and f != str:
                            raise ValueError
                        r.append(f(v))
                    except ValueError:
                        warn("Value '%r' could not be cast to %s, value set to pysal.MISSINGVALUE" % (v, str(f)), RuntimeWarning)
                        r.append(pysal.MISSINGVALUE)
                return r

        else:
            return row

    def next(self):
        """A FileIO object is its own iterator, see StringIO"""
        self._complain_ifclosed(self.closed)
        r = self.__read()
        if r is None:
            raise StopIteration
        return r

    def close(self):
        """ subclasses should clean themselves up and then call this method """
        if not self.closed:
            self.closed = True
            del self.dataObj, self.pos

    def get(self, n):
        """ Seeks the file to n and returns n
        If .ids is set n should be an id,
        else, n should be an offset
        """
        prevPos = self.tell()
        self.seek(n)
        obj = self.__read()
        self.seek(prevPos)
        return obj

    def seek(self, n):
        """ Seek the FileObj to the beginning of the n'th record,
            if ids are set, seeks to the beginning of the record at id, n"""
        self._complain_ifclosed(self.closed)
        self.pos = n

    def tell(self):
        """ Return id (or offset) of next object """
        self._complain_ifclosed(self.closed)
        return self.pos

    def read(self, n=-1):
        """ Read at most n objects, less if read hits EOF
        if size is negative or omitted read all objects until EOF
        returns None if EOF is reached before any objects.
        """
        self._complain_ifclosed(self.closed)
        if n < 0:
            #return list(self)
            result = []
            while 1:
                try:
                    result.append(self.__read())
                except StopIteration:
                    break
            return result
        elif n == 0:
            return None
        else:
            result = []
            for i in range(0, n):
                try:
                    result.append(self.__read())
                except StopIteration:
                    break
            return result

    def __read(self):
        """ Gets one row from the file handler, and if necessary casts it's objects """
        row = self._read()
        if row is None:
            raise StopIteration
        row = self._cast(row)
        return row

    def _read(self):
        """ Must be implemented by subclasses that support 'r'
        subclasses should increment .pos
        and redefine this doc string
        """
        self._complain_ifclosed(self.closed)
        raise NotImplementedError

    def truncate(self, size=None):
        """ Should be implemented by subclasses
        and redefine this doc string
        """
        self._complain_ifclosed(self.closed)
        raise NotImplementedError

    def write(self, obj):
        """ Must be implemented by subclasses that support 'w'
        subclasses should increment .pos
        subclasses should also check if obj is an instance of type(list)
        and redefine this doc string
        """
        self._complain_ifclosed(self.closed)
        "Write obj to dataObj"
        raise NotImplementedError

    def flush(self):
        self._complain_ifclosed(self.closed)
        raise NotImplementedError
