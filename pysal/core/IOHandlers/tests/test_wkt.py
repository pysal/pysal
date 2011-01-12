"""Well-Known Text Unit Tests for PySAL"""
import unittest
import pysal
import numpy as np

class WKTReader_Tester(unittest.TestCase):

    def setUp(self):
        self.sample = '../../sample.wkt'


    def test_open(self):
        f = pysal.open(self.sample)
        

    def test_seek(self):
        pass
    def test_write(self):
        failUnlessRaises(f.write(), NotImplementedError)

class WKTParser_Tester(unittest.TestCase):

    def setUp(self):
        self.sample = '../../sample.wkt'

    def test_Point(self):
        pass
         
    def test_LineString(self):
        pass
    def test_Polygon(self):
        pass
    def test_fromWKT(self):
        pass
