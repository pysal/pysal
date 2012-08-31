__all__ = ['DataTable', 'DataRow']
import FileIO

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"


class DataTable(FileIO.FileIO):
    """ DataTable provides additional functionality to FileIO for data table file tables
        FileIO Handlers that provide data tables should subclass this instead of FileIO """
    class _By_Col:
        def __init__(self, parent):
            self.p = parent

        def __repr__(self):
            return "keys: " + self.p.header.__repr__()

        def __getitem__(self, key):
            return self.p._get_col(key)

        def __setitem__(self, key, val):
            self.p.cast(key, val)

        def __call__(self, key):
            return self.p._get_col(key)

    def __init__(self, *args, **kwargs):
        FileIO.FileIO.__init__(self, *args, **kwargs)

    def __repr__(self):
        return 'DataTable: % s' % self.dataPath

    def __len__(self):
        """ __len__ should be implemented by DataTable Subclasses """
        raise NotImplementedError

    @property
    def by_col(self):
        return self._By_Col(self)

    def _get_col(self, key):
        """ returns the column vector
        """
        if not self.header:
            raise AttributeError('Please set the header')
        if key in self.header:
            return self[:, self.header.index(key)]
        else:
            raise AttributeError('Field: % s does not exist in header' % key)

    def __getitem__(self, key):
        """ DataTables fully support slicing in 2D,
            To provide slicing,  handlers must provide __len__
            Slicing accepts up to two arguments.
            Syntax,
            table[row]
            table[row, col]
            table[row_start:row_stop]
            table[row_start:row_stop:row_step]
            table[:, col]
            table[:, col_start:col_stop]
            etc.

            ALL indices are Zero-Offsets,
            i.e.
            #>>> assert index in range(0, len(table))
        """
        prevPos = self.tell()
        if issubclass(type(key), basestring):
            raise TypeError("index should be int or slice")
        if issubclass(type(key), int) or isinstance(key, slice):
            rows = key
            cols = None
        elif len(key) > 2:
            raise TypeError("DataTables support two dimmensional slicing,  % d slices provided" % len(key))
        elif len(key) == 2:
            rows, cols = key
        else:
            raise TypeError("Key: % r,  is confusing me.  I don't know what to do" % key)
        if isinstance(rows, slice):
            row_start, row_stop, row_step = rows.indices(len(self))
            self.seek(row_start)
            data = [self.next() for i in range(row_start, row_stop, row_step)]
        else:
            self.seek(slice(rows).indices(len(self))[1])
            data = [self.next()]
        if cols is not None:
            if isinstance(cols, slice):
                col_start, col_stop, col_step = cols.indices(len(data[0]))
                data = [r[col_start:col_stop:col_step] for r in data]
            else:
                #col_start, col_stop, col_step = cols, cols+1, 1
                data = [r[cols] for r in data]
        self.seek(prevPos)
        return data


def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == '__main__':
    _test()
