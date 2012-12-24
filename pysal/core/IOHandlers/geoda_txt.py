import pysal.core.Tables as Tables

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ['GeoDaTxtReader']


class GeoDaTxtReader(Tables.DataTable):
    """GeoDa Text File Export Format
    """
    __doc__ = Tables.DataTable.__doc__
    FORMATS = ['geoda_txt']
    MODES = ['r']

    def __init__(self, *args, **kwargs):
        """
        Examples
        --------
        >>> import pysal
        >>> f = pysal.open(pysal.examples.get_path('stl_hom.txt'),'r')
        >>> f.header
        ['FIPSNO', 'HR8488', 'HR8893', 'HC8488']
        >>> len(f)
        78
        >>> f.dat[0]
        ['17107', '1.290722', '1.624458', '2']
        >>> f.dat[-1]
        ['29223', '0', '8.451537', '0']
        >>> f._spec
        [<type 'int'>, <type 'float'>, <type 'float'>, <type 'int'>]

        """
        Tables.DataTable.__init__(self, *args, **kwargs)
        self.__idx = {}
        self.__len = None
        self.pos = 0
        self._open()

    def _open(self):
        if self.mode == 'r':
            self.fileObj = open(self.dataPath, 'r')
            n, k = self.fileObj.readline().strip().split(',')
            n, k = int(n), int(k)
            header = self.fileObj.readline().strip().split(',')
            self.header = [f.replace('"', '') for f in header]
            try:
                assert len(self.header) == k
            except AssertionError:
                raise TypeError("This is not a valid geoda_txt file.")
            dat = self.fileObj.readlines()
            self.dat = [line.strip().split(',') for line in dat]
            self._spec = self._determineSpec(self.dat)
            self.__len = len(dat)

    def __len__(self):
        return self.__len

    def _read(self):
        if self.pos < len(self):
            row = self.dat[self.pos]
            self.pos += 1
            return row
        else:
            raise None

    def close(self):
        self.fileObj.close()
        Tables.DataTable.close(self)

    @staticmethod
    def _determineSpec(data):
        cols = len(data[0])
        spec = []
        for j in range(cols):
            isInt = True
            isFloat = True
            for row in data:
                val = row[j]
                if not val.strip().replace('-', '').replace('.', '').isdigit():
                    isInt = False
                    isFloat = False
                    break
                else:
                    if isInt and '.' in val:
                        isInt = False
            if isInt:
                spec.append(int)
            elif isFloat:
                spec.append(float)
            else:
                spec.append(str)
        return spec
