import unittest
import pysal
import tempfile
import os

class csv_Tester(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = '../../examples/stl_hom.csv'
        self.obj = pysal.core.IOHandlers.csvWrapper.csvWrapper(test_file,'r')
    def test_len(self):
        self.assertEquals(len(self.obj),78)
    def test_tell(self):
        self.assertEquals(self.obj.tell(),0)
        self.obj.read(1)
        self.assertEquals(self.obj.tell(),1)
        self.obj.read(50)
        self.assertEquals(self.obj.tell(),51)
        self.obj.read()
        self.assertEquals(self.obj.tell(),78)
    def test_seek(self):
        self.obj.seek(0)
        self.assertEquals(self.obj.tell(),0)
        self.obj.seek(55)
        self.assertEquals(self.obj.tell(),55)
        self.obj.read(1)
        self.assertEquals(self.obj.tell(),56)
    def test_read(self):
        self.obj.seek(0)
        objs = self.obj.read()
        self.assertEquals(len(objs),78)
        self.obj.seek(0)
        objsB = list(self.obj)
        self.assertEquals(len(objsB),78)
        for rowA,rowB in zip(objs,objsB):
            self.assertEquals(rowA,rowB)
    def test_casting(self):
        self.obj.cast('WKT',pysal.core.IOHandlers.wkt.WKTParser())
        verts = [(-89.585220336914062,39.978794097900391),(-89.581146240234375,40.094867706298828),(-89.603988647460938,40.095306396484375),(-89.60589599609375,40.136119842529297),(-89.6103515625,40.3251953125),(-89.269027709960938,40.329566955566406),(-89.268562316894531,40.285579681396484),(-89.154655456542969,40.285774230957031),(-89.152763366699219,40.054969787597656),(-89.151618957519531,39.919403076171875),(-89.224777221679688,39.918678283691406),(-89.411857604980469,39.918041229248047),(-89.412437438964844,39.931644439697266),(-89.495201110839844,39.933486938476562),(-89.4927978515625,39.980186462402344),(-89.585220336914062,39.978794097900391)]
        for i,pt in enumerate(self.obj.next()[0].vertices):
            self.assertEquals(pt[:],verts[i])
    def test_by_col(self):
        for field in self.obj.header:
            self.assertEquals(len(self.obj.by_col[field]),78)
    def test_slicing(self):
        chunk = self.obj[50:55,1:3]
        self.assertEquals(chunk[0],['Jefferson','Missouri'])
        self.assertEquals(chunk[1],['Jefferson', 'Illinois'])
        self.assertEquals(chunk[2],['Miller', 'Missouri'])
        self.assertEquals(chunk[3],['Maries', 'Missouri'])
        self.assertEquals(chunk[4],['White', 'Illinois'])
            

class PurePyDbf_Tester(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = '../../examples/10740.dbf'
        self.dbObj = pysal.core.IOHandlers.pyDbfIO.DBF(test_file,'r')
    def test_len(self):
        self.assertEquals(len(self.dbObj),195)
    def test_tell(self):
        self.assertEquals(self.dbObj.tell(),0)
        self.dbObj.read(1)
        self.assertEquals(self.dbObj.tell(),1)
        self.dbObj.read(50)
        self.assertEquals(self.dbObj.tell(),51)
        self.dbObj.read()
        self.assertEquals(self.dbObj.tell(),195)
    def test_seek(self):
        self.dbObj.seek(0)
        self.assertEquals(self.dbObj.tell(),0)
        self.dbObj.seek(55)
        self.assertEquals(self.dbObj.tell(),55)
        self.dbObj.read(1)
        self.assertEquals(self.dbObj.tell(),56)
    def test_read(self):
        self.dbObj.seek(0)
        objs = self.dbObj.read()
        self.assertEquals(len(objs),195)
        self.dbObj.seek(0)
        objsB = list(self.dbObj)
        self.assertEquals(len(objsB),195)
        for rowA,rowB in zip(objs,objsB):
            self.assertEquals(rowA,rowB)
    def test_random_access(self):
        self.dbObj.seek(0)
        db0 = self.dbObj.read(1)[0]
        self.assertEquals(db0,[1, '35001', '000107', '35001000107', '1.07'])
        self.dbObj.seek(57)
        db57 = self.dbObj.read(1)[0]
        self.assertEquals(db57,[58, '35001', '001900', '35001001900', '19'])
        self.dbObj.seek(32)
        db32 = self.dbObj.read(1)[0]
        self.assertEquals(db32,[33, '35001', '000500', '35001000500', '5'])
        self.dbObj.seek(0)
        self.assertEquals(self.dbObj.next(),db0)
        self.dbObj.seek(57)
        self.assertEquals(self.dbObj.next(),db57)
        self.dbObj.seek(32)
        self.assertEquals(self.dbObj.next(),db32)
    def test_write(self):
        f = tempfile.NamedTemporaryFile(suffix='.dbf'); fname = f.name; f.close()
        self.dbfcopy = fname
        self.out = pysal.core.IOHandlers.pyDbfIO.DBF(fname,'w')
        self.dbObj.seek(0)
        self.out.header  = self.dbObj.header
        self.out.field_spec  = self.dbObj.field_spec
        for row in self.dbObj:
            self.out.write(row)
        self.out.close()
        
        orig = open(self.test_file,'rb')
        copy = open(self.dbfcopy,'rb')
        orig.seek(32)#self.dbObj.header_size) #skip the header, file date has changed
        copy.seek(32)#self.dbObj.header_size) #skip the header, file date has changed
        
        #PySAL writes proper DBF files with a terminator at the end, not everyone does.
        n = self.dbObj.record_size*self.dbObj.n_records #bytes to read.
        self.assertEquals(orig.read(n), copy.read(n)) 
        #self.assertEquals(orig.read(1), copy.read(1)) # last byte may fail
        orig.close()
        copy.close()
        os.remove(self.dbfcopy)
    
class GalIO_Tester(unittest.TestCase):
    def setUp(self):
        self.test_file = '../../examples/10740.shp'
        self.w = pysal.queen_from_shapefile(self.test_file)
    def test_write(self):
        out = pysal.open('tst.gal','w')
        out.write(self.w)
        out.close()
        
class PurePyShp_Tester(unittest.TestCase):
    def setUp(self):
        test_file = '../../examples/10740.shp'
        self.test_file = test_file
        self.shpObj = pysal.core.IOHandlers.pyShpIO.PurePyShpWrapper(test_file,'r')
        f = tempfile.NamedTemporaryFile(suffix='.shp'); shpcopy = f.name; f.close()
        self.shpcopy = shpcopy
        self.shxcopy = shpcopy.replace('.shp','.shx')
    def test_len(self):
        self.assertEquals(len(self.shpObj),195)
    def test_tell(self):
        self.assertEquals(self.shpObj.tell(),0)
        self.shpObj.read(1)
        self.assertEquals(self.shpObj.tell(),1)
        self.shpObj.read(50)
        self.assertEquals(self.shpObj.tell(),51)
        self.shpObj.read()
        self.assertEquals(self.shpObj.tell(),195)
    def test_seek(self):
        self.shpObj.seek(0)
        self.assertEquals(self.shpObj.tell(),0)
        self.shpObj.seek(55)
        self.assertEquals(self.shpObj.tell(),55)
        self.shpObj.read(1)
        self.assertEquals(self.shpObj.tell(),56)
    def test_read(self):
        self.shpObj.seek(0)
        objs = self.shpObj.read()
        self.assertEquals(len(objs),195)

        self.shpObj.seek(0)
        objsB = list(self.shpObj)
        self.assertEquals(len(objsB),195)

        for shpA,shpB in zip(objs,objsB):
            self.assertEquals(shpA.vertices,shpB.vertices)
    def test_random_access(self):
        self.shpObj.seek(57)
        shp57 = self.shpObj.read(1)[0]
        self.shpObj.seek(32)
        shp32 = self.shpObj.read(1)[0]

        self.shpObj.seek(57)
        self.assertEquals(self.shpObj.read(1)[0].vertices,shp57.vertices)
        self.shpObj.seek(32)
        self.assertEquals(self.shpObj.read(1)[0].vertices,shp32.vertices)
    def test_write(self):
        out = pysal.core.IOHandlers.pyShpIO.PurePyShpWrapper(self.shpcopy,'w')
        self.shpObj.seek(0)
        for shp in self.shpObj:
            out.write(shp)
        out.close()

        orig = open(self.test_file,'rb')
        copy = open(self.shpcopy,'rb')
        self.assertEquals(orig.read(),copy.read())
        orig.close()
        copy.close()

        oshx = open(self.test_file.replace('.shp','.shx'),'rb')
        cshx = open(self.shxcopy,'rb')
        self.assertEquals(oshx.read(),cshx.read())
        oshx.close()
        cshx.close()

        os.remove(self.shpcopy)
        os.remove(self.shxcopy)


suite = unittest.TestSuite()
A = unittest.TestLoader().loadTestsFromTestCase(PurePyShp_Tester)
suite.addTest(A)
B = unittest.TestLoader().loadTestsFromTestCase(PurePyDbf_Tester)
suite.addTest(B)
C = unittest.TestLoader().loadTestsFromTestCase(csv_Tester)
suite.addTest(C)
D = unittest.TestLoader().loadTestsFromTestCase(GalIO_Tester)
suite.addTest(D)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
