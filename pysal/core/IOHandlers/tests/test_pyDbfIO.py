

import unittest
import pysal
import tempfile
import os

class test_DBF(unittest.TestCase):
    def setUp(self):
        self.test_file = test_file = pysal.examples.get_path('10740.dbf')
        self.dbObj = pysal.core.IOHandlers.pyDbfIO.DBF(test_file, 'r')

    def test_len(self):
        self.assertEqual(len(self.dbObj), 195)

    def test_tell(self):
        self.assertEqual(self.dbObj.tell(), 0)
        self.dbObj.read(1)
        self.assertEqual(self.dbObj.tell(), 1)
        self.dbObj.read(50)
        self.assertEqual(self.dbObj.tell(), 51)
        self.dbObj.read()
        self.assertEqual(self.dbObj.tell(), 195)
    
    def test_cast(self):
        self.assertEqual(self.dbObj._spec, [])
        self.dbObj.cast('FIPSSTCO', float)
        self.assertEqual(self.dbObj._spec[1], float)

    def test_seek(self):
        self.dbObj.seek(0)
        self.assertEqual(self.dbObj.tell(), 0)
        self.dbObj.seek(55)
        self.assertEqual(self.dbObj.tell(), 55)
        self.dbObj.read(1)
        self.assertEqual(self.dbObj.tell(), 56)

    def test_read(self):
        self.dbObj.seek(0)
        objs = self.dbObj.read()
        self.assertEqual(len(objs), 195)
        self.dbObj.seek(0)
        objsB = list(self.dbObj)
        self.assertEqual(len(objsB), 195)
        for rowA, rowB in zip(objs, objsB):
            self.assertEqual(rowA, rowB)

    def test_random_access(self):
        self.dbObj.seek(0)
        db0 = self.dbObj.read(1)[0]
        self.assertEqual(db0, [1, '35001', '000107', '35001000107', '1.07'])
        self.dbObj.seek(57)
        db57 = self.dbObj.read(1)[0]
        self.assertEqual(db57, [58, '35001', '001900', '35001001900', '19'])
        self.dbObj.seek(32)
        db32 = self.dbObj.read(1)[0]
        self.assertEqual(db32, [33, '35001', '000500', '35001000500', '5'])
        self.dbObj.seek(0)
        self.assertEqual(next(self.dbObj), db0)
        self.dbObj.seek(57)
        self.assertEqual(next(self.dbObj), db57)
        self.dbObj.seek(32)
        self.assertEqual(next(self.dbObj), db32)

    def test_write(self):
        f = tempfile.NamedTemporaryFile(suffix='.dbf')
        fname = f.name
        f.close()
        self.dbfcopy = fname
        self.out = pysal.core.IOHandlers.pyDbfIO.DBF(fname, 'w')
        self.dbObj.seek(0)
        self.out.header = self.dbObj.header
        self.out.field_spec = self.dbObj.field_spec
        for row in self.dbObj:
            self.out.write(row)
        self.out.close()

        orig = open(self.test_file, 'rb')
        copy = open(self.dbfcopy, 'rb')
        orig.seek(32)  # self.dbObj.header_size) #skip the header, file date has changed
        copy.seek(32)  # self.dbObj.header_size) #skip the header, file date has changed

        #PySAL writes proper DBF files with a terminator at the end, not everyone does.
        n = self.dbObj.record_size * self.dbObj.n_records  # bytes to read.
        self.assertEqual(orig.read(n), copy.read(n))
        #self.assertEquals(orig.read(1), copy.read(1)) # last byte may fail
        orig.close()
        copy.close()
        os.remove(self.dbfcopy)

    def test_writeNones(self):
        import datetime
        import time
        f = tempfile.NamedTemporaryFile(
            suffix='.dbf')
        fname = f.name
        f.close()
        db = pysal.core.IOHandlers.pyDbfIO.DBF(fname, 'w')
        db.header = ["recID", "date", "strID", "aFloat"]
        db.field_spec = [('N', 10, 0), ('D', 8, 0), ('C', 10, 0), ('N', 5, 5)]
        records = []
        for i in range(10):
            d = datetime.date(*time.localtime()[:3])
            rec = [i + 1, d, str(i + 1), (i + 1) / 2.0]
            records.append(rec)
        records.append([None, None, '', None])
        records.append(rec)
        for rec in records:
            db.write(rec)
        db.close()
        db2 = pysal.core.IOHandlers.pyDbfIO.DBF(fname, 'r')
        self.assertEqual(records, db2.read())

        os.remove(fname)

if __name__ == '__main__':
    unittest.main()
