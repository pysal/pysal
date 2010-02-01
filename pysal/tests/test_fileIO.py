import unittest
import pysal.core._FileIO.pyShpIO
import pysal.core._FileIO.pyDbfIO
import tempfile
import os

class PurePyDbf_Tester(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = '../examples/10740.dbf'
        self.dbObj = pysal.core._FileIO.pyDbfIO.DBF(test_file,'r')
        f = tempfile.NamedTemporaryFile(suffix='.dbf',delete=False); fname = f.name; f.close()
        self.fname = fname
        self.out = pysal.core._FileIO.pyDbfIO.DBF(fname,'w')
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
        self.dbObj.seek(57)
        db57 = self.dbObj.read(1)[0]
        self.dbObj.seek(32)
        db32 = self.dbObj.read(1)[0]

        self.dbObj.seek(57)
        self.assertEquals(self.dbObj.read(1)[0],db57)
        self.dbObj.seek(32)
        self.assertEquals(self.dbObj.read(1)[0],db32)
    def test_write(self):
        self.dbObj.seek(0)
        self.out.header  = self.dbObj.header
        self.out.field_spec  = self.dbObj.field_spec
        for row in self.dbObj:
            self.out.write(row)
        self.out.close()
        
        orig = open(self.test_file,'rb')
        copy = open(self.fname,'rb')
        orig.seek(self.dbObj.header_size) #skip the header, file date has changed
        copy.seek(self.dbObj.header_size) #skip the header, file date has changed
        
        #PySAL writes proper DBF files with a terminator at the end, not everyone does.
        n = self.dbObj.record_size*self.dbObj.n_records #bytes to read.
        self.assertEquals(orig.read(n), copy.read(n)) 
        #self.assertEquals(orig.read(1), copy.read(1)) # last byte may fail
        orig.close()
        copy.close()
        os.remove(self.fname)
    
class PurePyShp_Tester(unittest.TestCase):
    def setUp(self):
        test_file = '../examples/10740.shp'
        self.shpObj = pysal.core._FileIO.pyShpIO.PurePyShpWrapper(test_file,'r')
        f = tempfile.NamedTemporaryFile(suffix='.shp',delete=False); fname = f.name; f.close()
        self.fname = fname
        self.out = pysal.core._FileIO.pyShpIO.PurePyShpWrapper(fname,'w')
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
        

suite = unittest.TestSuite()
A = unittest.TestLoader().loadTestsFromTestCase(PurePyShp_Tester)
suite.addTest(A)
B = unittest.TestLoader().loadTestsFromTestCase(PurePyDbf_Tester)
suite.addTest(B)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
