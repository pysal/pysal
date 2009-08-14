import pysal.core.Tables as Tables
import csv

class csvWrapper(Tables.DataTable):

    __doc__ = Tables.DataTable.__doc__

    FORMATS = ['csv']
    MODES = ['r']

    def __init__(self,*args,**kwargs):
        Tables.DataTable.__init__(self,*args,**kwargs)
        self.__idx = {}
        self.__len = None
        self._open()
    def __len__(self):
        return self.__len
    def _open(self):
        self.fileObj = open(self.dataPath,self.mode)
        if self.mode == 'r':
            self.dataObj = csv.reader(self.fileObj)
            data = list(self.dataObj)
            if self._determineHeader(data):
                self._header = data.pop(0)
            else:
                self._header = ['field_%d'%i for i in range(len(data[0]))]
            self._spec = self._determineSpec(data)
            self.data = data
            self.fileObj.close()
            self.__len = len(data)
    def _determineHeader(self,data):
        #head = [val.strip().replace('-','').replace('.','').isdigit() for val in data[0]]
        #if True in head: #no numbers in header!
        #    HEADER = False
        #    return HEADER
        headSpec = self._determineSpec([data[0]])
        restSpec = self._determineSpec(data[1:])
        if headSpec == restSpec:
            HEADER = False
            return HEADER
        return True
    @staticmethod
    def _determineSpec(data):
        cols = len(data[0])
        spec = []
        for j in range(cols):
            isInt = True
            isFloat = True
            for row in data:
                val = row[j]
                if not val.strip().replace('-','').replace('.','').isdigit():
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
    def _read(self):
        if self.pos < len(self):
            row  = self.data[self.pos]
            self.pos+=1
            return row
        else:
            return None
if __name__ == '__main__':
    file_name = '../../data/stl_hom.csv'
    f = pysal.open(file_name,'r')
    print f._header
    print f._spec
    for row in f:
        print row
        
        
