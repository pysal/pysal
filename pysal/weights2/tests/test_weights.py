import unittest as ut
import pysal as ps
from pysal.weights2 import W

class Test_Weights(ut.TestCase):
    def setUp(self):
        raise NotImplementedError

    def test_init(self):
        raise NotImplementedError

    def test_from_file(self):
        raise NotImplementedError
    
    def test_transform(self):
        raise NotImplementedError

    def test_remap_ids(self):
        raise NotImplementedError

    def test_to_WSP(self):
        raise NotImplementedError

    def test_from_WSP(self):
        raise NotImplementedError

    def test_full(self):
        raise NotImplementedError

class Test_WSP(ut.TestCase):
    def setUp(self):
        raise NotImplementedError

    def test_init(self):
        raise NotImplementedError

    def test_from_W(self):
        raise NotImplementedError

    def test_to_W(self):
        raise NotImplementedError
