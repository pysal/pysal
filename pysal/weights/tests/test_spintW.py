import unittest
import pysal
import numpy as np
from pysal.weights.spintW import ODW

class TestODWeights(unittest.TestCase):
    def setUp(self):
        self.O = pysal.weights.lat2W(2,2)
        self.D = pysal.weights.lat2W(2,2)
        self.ODW =  np.array(
                [[ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.25,  0.  ,  0.  ,
                0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
                [ 0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.25,  0.25,
                0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ],
                [ 0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.25,  0.25,
                0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ],
                [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.25,  0.  ,  0.  ,
                0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
                [ 0.  ,  0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.25,  0.  ],
                [0.25,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.25],
                [ 0.25,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.25],
                [ 0.  ,  0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.25,  0.  ],
                [ 0.  ,  0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.25,  0.  ],
                [ 0.25,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.25],
                [ 0.25,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.25],
                [ 0.  ,  0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.25,  0.  ],
                [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.25,  0.  ,  0.  ,
                0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
                [ 0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.25,  0.25,
                0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ],
                [ 0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.25,  0.25,
                0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ],
                [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.25,  0.  ,  0.  ,
                0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])

    def test_ODW_full(self):
        W = ODW(self.O, self.D)
        np.testing.assert_allclose(self.ODW, W.full()[0])

class TestNetW(unittest.TestCase):
    def setUp(self):
        pass

    def test_netOD(self):
        pass

    def test_netO(self):
        pass

    def test_netD(self):
        pass

    def test_netC(self):
        pass

    def test_netA(self):
        pass

    def test_mat2L(self):
        pass

class TestVecW(unittest.TestCase):
    def setUp(self):
        pass

    def test_vecW(self):
        pass
