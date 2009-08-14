import pysal.core.FileIO as FileIO

class GeoDaTxtReader(FileIO.FileIO):
    """GeoDa Text File Export Format"""

    FORMATS = ['geoda_txt']
    MODES = ['r']

    def __init__(self,*args,**kwargs):
        pysal.core.FileIO.__init__(self,*args,**kwargs)
        self.file = open(self.dataPath,self.mode)
        n,k=map(int, self.file.readline().strip().split(",")) # Get through header info

    def _read(self):
        line = self.file.readline()
        self.pos += 1
        if line:
            return map(float, line.strip().split(','))
        else:
            raise StopIteration

    def close(self):
        self.file.close()
        pysal.core.FileIO.close(self)


