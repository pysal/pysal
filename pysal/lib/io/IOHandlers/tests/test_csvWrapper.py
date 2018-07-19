import unittest
from .... import examples as pysal_examples
from .. import csvWrapper
from ...util import WKTParser
import tempfile
import os
from sys import version as V

PY3 = int(V[0]) > 2

class test_csvWrapper(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal_examples.get_path('stl_hom.csv')
        self.obj = csvWrapper.csvWrapper(test_file, 'r')

    def test_len(self):
        self.assertEqual(len(self.obj), 78)

    def test_tell(self):
        self.assertEqual(self.obj.tell(), 0)
        self.obj.read(1)
        self.assertEqual(self.obj.tell(), 1)
        self.obj.read(50)
        self.assertEqual(self.obj.tell(), 51)
        self.obj.read()
        self.assertEqual(self.obj.tell(), 78)

    def test_seek(self):
        self.obj.seek(0)
        self.assertEqual(self.obj.tell(), 0)
        self.obj.seek(55)
        self.assertEqual(self.obj.tell(), 55)
        self.obj.read(1)
        self.assertEqual(self.obj.tell(), 56)

    def test_read(self):
        self.obj.seek(0)
        objs = self.obj.read()
        self.assertEqual(len(objs), 78)
        self.obj.seek(0)
        objsB = list(self.obj)
        self.assertEqual(len(objsB), 78)
        for rowA, rowB in zip(objs, objsB):
            self.assertEqual(rowA, rowB)

    def test_casting(self):
        self.obj.cast('WKT', WKTParser())
        verts = [(-89.585220336914062, 39.978794097900391), (-89.581146240234375, 40.094867706298828), (-89.603988647460938, 40.095306396484375), (-89.60589599609375, 40.136119842529297), (-89.6103515625, 40.3251953125), (-89.269027709960938, 40.329566955566406), (-89.268562316894531, 40.285579681396484), (-89.154655456542969, 40.285774230957031), (-89.152763366699219, 40.054969787597656), (-89.151618957519531, 39.919403076171875), (-89.224777221679688, 39.918678283691406), (-89.411857604980469, 39.918041229248047), (-89.412437438964844, 39.931644439697266), (-89.495201110839844, 39.933486938476562), (-89.4927978515625, 39.980186462402344), (-89.585220336914062, 39.978794097900391)]
        if PY3:
            for i, pt in enumerate(self.obj.__next__()[0].vertices):
                self.assertEqual(pt[:], verts[i])
        else:
            for i, pt in enumerate(self.obj.next()[0].vertices):
                self.assertEqual(pt[:], verts[i])

    def test_by_col(self):
        for field in self.obj.header:
            self.assertEqual(len(self.obj.by_col[field]), 78)

    def test_slicing(self):
        chunk = self.obj[50:55, 1:3]
        self.assertEqual(chunk[0], ['Jefferson', 'Missouri'])
        self.assertEqual(chunk[1], ['Jefferson', 'Illinois'])
        self.assertEqual(chunk[2], ['Miller', 'Missouri'])
        self.assertEqual(chunk[3], ['Maries', 'Missouri'])
        self.assertEqual(chunk[4], ['White', 'Illinois'])

if __name__ == '__main__':
    unittest.main()
