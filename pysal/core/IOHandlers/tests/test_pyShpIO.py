import unittest
import pysal
import tempfile
import os


class test_PurePyShpWrapper(unittest.TestCase):
    def setUp(self):
        test_file = pysal.examples.get_path('10740.shp')
        self.test_file = test_file
        self.shpObj = pysal.core.IOHandlers.pyShpIO.PurePyShpWrapper(
            test_file, 'r')
        f = tempfile.NamedTemporaryFile(suffix='.shp')
        shpcopy = f.name
        f.close()
        self.shpcopy = shpcopy
        self.shxcopy = shpcopy.replace('.shp', '.shx')

    def test_len(self):
        self.assertEquals(len(self.shpObj), 195)

    def test_tell(self):
        self.assertEquals(self.shpObj.tell(), 0)
        self.shpObj.read(1)
        self.assertEquals(self.shpObj.tell(), 1)
        self.shpObj.read(50)
        self.assertEquals(self.shpObj.tell(), 51)
        self.shpObj.read()
        self.assertEquals(self.shpObj.tell(), 195)

    def test_seek(self):
        self.shpObj.seek(0)
        self.assertEquals(self.shpObj.tell(), 0)
        self.shpObj.seek(55)
        self.assertEquals(self.shpObj.tell(), 55)
        self.shpObj.read(1)
        self.assertEquals(self.shpObj.tell(), 56)

    def test_read(self):
        self.shpObj.seek(0)
        objs = self.shpObj.read()
        self.assertEquals(len(objs), 195)

        self.shpObj.seek(0)
        objsB = list(self.shpObj)
        self.assertEquals(len(objsB), 195)

        for shpA, shpB in zip(objs, objsB):
            self.assertEquals(shpA.vertices, shpB.vertices)

    def test_random_access(self):
        self.shpObj.seek(57)
        shp57 = self.shpObj.read(1)[0]
        self.shpObj.seek(32)
        shp32 = self.shpObj.read(1)[0]

        self.shpObj.seek(57)
        self.assertEquals(self.shpObj.read(1)[0].vertices, shp57.vertices)
        self.shpObj.seek(32)
        self.assertEquals(self.shpObj.read(1)[0].vertices, shp32.vertices)

    def test_write(self):
        out = pysal.core.IOHandlers.pyShpIO.PurePyShpWrapper(self.shpcopy, 'w')
        self.shpObj.seek(0)
        for shp in self.shpObj:
            out.write(shp)
        out.close()

        orig = open(self.test_file, 'rb')
        copy = open(self.shpcopy, 'rb')
        self.assertEquals(orig.read(), copy.read())
        orig.close()
        copy.close()

        oshx = open(self.test_file.replace('.shp', '.shx'), 'rb')
        cshx = open(self.shxcopy, 'rb')
        self.assertEquals(oshx.read(), cshx.read())
        oshx.close()
        cshx.close()

        os.remove(self.shpcopy)
        os.remove(self.shxcopy)

if __name__ == '__main__':
    unittest.main()
