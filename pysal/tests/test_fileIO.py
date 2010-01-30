import unittest
import pysal.core._FileIO.pyShpIO
import tempfile
import os

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
        

suite = unittest.TestLoader().loadTestsFromTestCase(PurePyShp_Tester)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
