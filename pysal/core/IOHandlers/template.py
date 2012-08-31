""" Example Reader and Writer

    These are working readers/writers that parse '.foo' and '.bar' files

"""

import pysal
from pysal.core.FileIO import FileIO
__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ['TemplateWriter', 'TemplateReaderWriter']


# Always subclass FileIO
class TemplateWriter(FileIO):
    #REQUIRED, List the formats this class supports.
    FORMATS = ['foo']
    #REQUIRED, List the modes supported by this class.
    # One class can support both reading and writing.
    # For simplicity this class will only support one.
    # You could support custom modes, but these could be hard to document.
    MODES = ['w']

    # use .__init__ to open any need file handlers
    def __init__(self, *args, **kwargs):
        # initialize the parent class...
        FileIO.__init__(self, *args, **kwargs)
        # this gives you:
        # self.dataPath == the connection string or path to file
        # self.mode == the mode the file should be opened in

        self.fileObj = open(self.dataPath, self.mode)

    #writers must subclass .write
    def write(self, obj):
        """ .write method of the 'foobar' template, receives an obj """

        # GOOD TO HAVE, this will prevent invalid operations on closed files.
        self._complain_ifclosed(self.closed)

        # it's up to the writer to understand the object, you should check that object is of the type you expect and raise a TypeError is its now.
        # we will support writing string objects in this example, all string are derived from basestring...
        if issubclass(type(obj), basestring):

            #Non-essential ...
            def foobar(c):
                if c in 'foobar':
                    return True
                else:
                    return False
            result = filter(foobar, obj)  # e.g.   'foobara' == filter(foobar,'my little foobar example')

            #do the actual writing...
            self.fileObj.write(result + '\n')
            #REQUIRED, increment the internal pos pointer.
            self.pos += 1

        else:
            raise TypeError("Expected a string, got: %s" % (type(obj)))

    #default is to raise "NotImplementedError"
    def flush(self):
        self._complain_ifclosed(self.closed)
        self.fileObj.flush()

    #REQUIRED
    def close(self):
        self.fileObj.close()
        #clean up the parent class too....
        FileIO.close(self)


class TemplateReaderWriter(FileIO):
    FORMATS = ['bar']
    MODES = ['r', 'w']

    def __init__(self, *args, **kwargs):
        FileIO.__init__(self, *args, **kwargs)
        self.fileObj = open(self.dataPath, self.mode)
    #Notice reading is a bit different

    def _filter(self, st):
        def foobar(c):
            if c in 'foobar':
                return True
            else:
                return False
        return filter(foobar, st)  # e.g.   'foobara' == filter(foobar,'my little foobar example')

    def _read(self):
        """ the _read method should return only ONE object and raise StopIteration at the EOF."""
        line = self.fileObj.readline()
        obj = self._filter(line)
        self.pos += 1  # REQUIRED
        if line:
            return obj + '\n'
        else:
            raise StopIteration  # REQUIRED

    def write(self, obj):
        """ .write method of the 'foobar' template, receives an obj """
        self._complain_ifclosed(self.closed)
        if issubclass(type(obj), basestring):
            result = self._filter(obj)
            self.fileObj.write(result + '\n')
            self.pos += 1
        else:
            raise TypeError("Expected a string, got: %s" % (type(obj)))

    def flush(self):
        self._complain_ifclosed(self.closed)
        self.fileObj.flush()

    def close(self):
        self.fileObj.close()
        FileIO.close(self)


if __name__ == '__main__':
    "note, by running OR importing this module it's automatically added to the pysal fileIO registry."
    pysal.open.check()

    lines = ['This is an example of template FileIO classes',
             'Each call to write expects a string object',
             'that string is filtered and only letters "f,o,b,a,r" are kept',
             'these kept letters are written to the file and a new line char is appends to each line',
             'likewise the reader filters each line from a file']
    f = pysal.open('test.foo', 'w')
    for line in lines:
        f.write(line)
    f.close()

    f = pysal.open('test.bar', 'w')
    for line in lines:
        f.write(line)
    f.close()

    f = pysal.open('test.bar', 'r')
    s = ''.join(f.read())
    f.close()
    print s

    f = open('test.foo', 'r')
    s2 = f.read()
    f.close()
    print s == s2
