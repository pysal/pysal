""" This module wraps Andrew's wrapper around Shapelib """
import pysal.core.FileIO as FileIO
import pyshp.shp as shp
import pysal.cg.shapes as cg

STRING_TO_TYPE = {'POLYGON':cg.Polygon,'POINT':cg.Point,'ARC':cg.LineSegment}
TYPE_TO_STRING = {} #build the reverse map
for key,value in STRING_TO_TYPE.iteritems():
    TYPE_TO_STRING[value] = key

class ShpLibWrapper(FileIO.FileIO):
    FORMATS = ['shp']
    MODES = ['w']
    def __init__(self,*args,**kwargs):
        pysal.core.FileIO.__init__(self,*args,**kwargs)
        if self.mode == 'r':
            self.__open()
        elif self.mode == 'w':
            self.__create()
    def __open(self):
        self.dataObj = shp.Shpfile(self.dataPath)
        try:
            self.type = STRING_TO_TYPE[self.dataObj.type()]
        except KeyError:
            raise TypeError,'%s does not support shapes of type: %s'\
                        %(self.__class__.__name__,self.dataObj.type())
    def __create(self):
        self.write = self.__firstWrite
    def __firstWrite(self,shape):
        self.type = TYPE_TO_STRING[type(shape)]
        self.dataObj = shp.create_shpshx_file(self.dataPath,self.type)
        self.write = self.__writer
        self.write(shape)
    def __writer(self,shape):
        if self.type == 'POINT':
            mins = [shape.x,shape.y,0,0]
            maxs = [shape.x,shape.y,0,0]
            shape = [shape]
        else:
            mins = [shape.bbox.minx,shape.bbox.miny,0,0]
            maxs = [shape.bbox.maxx,shape.bbox.maxy,0,0]
        verts = [(x,y) for x,y in shape]
        record = [self.type,self.pos+1,[],verts,mins,maxs] #+1 because shpfiles start at 1
        self.dataObj.write([record])
        self.pos+=1
