import unittest
import pysal
from pysal.spatial_dynamics import markov
import numpy as np

class test_Markov(unittest.TestCase):
    def test___init__(self):
        # markov = Markov(class_ids, classes)
        assert False # TODO: implement your test here

class test_Spatial_Markov(unittest.TestCase):
    def test___init__(self):
        # spatial__markov = Spatial_Markov(y, w, k, permutations, fixed)
        assert False # TODO: implement your test here

class test_chi2(unittest.TestCase):
    def test_chi2(self):
        # self.assertEqual(expected, chi2(T1, T2))
        assert False # TODO: implement your test here

class test_LISA_Markov(unittest.TestCase):
    def test___init__(self):
        # l_is_a__markov = LISA_Markov(y, w)
        assert False # TODO: implement your test here

class test_prais(unittest.TestCase):
    def test_prais(self):
        # self.assertEqual(expected, prais(pmat))
        assert False # TODO: implement your test here

class test_shorrock(unittest.TestCase):
    def test_shorrock(self):
        # self.assertEqual(expected, shorrock(pmat))
        assert False # TODO: implement your test here

class test_directional(unittest.TestCase):
    def test_directional(self):
        # self.assertEqual(expected, directional(pmat))
        assert False # TODO: implement your test here

class test_homogeneity(unittest.TestCase):
    def test_homogeneity(self):
        # self.assertEqual(expected, homogeneity(classids, colIds))
        assert False # TODO: implement your test here

class test_path_probabilities(unittest.TestCase):
    def test_path_probabilities(self):
        # self.assertEqual(expected, path_probabilities(class_ids, classes))
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()


