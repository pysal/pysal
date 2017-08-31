import unittest
import numpy as np
import pysal
from pysal.contrib.gwr.kernels import *

PEGP = pysal.examples.get_path

class TestKernels(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        x = np.arange(1,6)
        y = np.arange(5,0, -1)
        np.random.shuffle(x)
        np.random.shuffle(y)
        self.coords = np.array(zip(x, y))
        self.fix_gauss_kern = np.array([
        [ 1.        ,  0.38889556,  0.48567179,  0.48567179,  0.89483932],
        [ 0.38889556,  1.        ,  0.89483932,  0.64118039,  0.48567179],
        [ 0.48567179,  0.89483932,  1.        ,  0.89483932,  0.48567179],
        [ 0.48567179,  0.64118039,  0.89483932,  1.        ,  0.38889556],
        [ 0.89483932,  0.48567179,  0.48567179,  0.38889556,  1.        ]])
        self.adapt_gauss_kern = np.array([
        [ 1.        ,  0.52004183,  0.60653072,  0.60653072,  0.92596109],
        [ 0.34559083,  1.        ,  0.88249692,  0.60653072,  0.44374738],
        [ 0.03877423,  0.60653072,  1.        ,  0.60653072,  0.03877423],
        [ 0.44374738,  0.60653072,  0.88249692,  1.        ,  0.34559083],
        [ 0.92596109,  0.60653072,  0.60653072,  0.52004183,  1.        ]])
        self.fix_bisquare_kern = np.array([
        [ 1.        ,  0.        ,  0.        ,  0.        ,  0.60493827],
        [ 0.        ,  1.        ,  0.60493827,  0.01234568,  0.        ],
        [ 0.        ,  0.60493827,  1.        ,  0.60493827,  0.        ],
        [ 0.        ,  0.01234568,  0.60493827,  1.        ,  0.        ],
        [ 0.60493827,  0.        ,  0.        ,  0.        ,  1.        ]])
        self.adapt_bisquare_kern = np.array([
        [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
           3.99999881e-14,   7.15976383e-01],
        [  0.00000000e+00,   1.00000000e+00,   5.62500075e-01,
           3.99999881e-14,   0.00000000e+00],
        [  0.00000000e+00,   3.99999881e-14,   1.00000000e+00,
           3.99999881e-14,   0.00000000e+00],
        [  0.00000000e+00,   3.99999881e-14,   5.62500075e-01,
           1.00000000e+00,   0.00000000e+00],
        [  7.15976383e-01,   0.00000000e+00,   3.99999881e-14,
           0.00000000e+00,   1.00000000e+00]])
        self.fix_exp_kern = np.array([
        [ 1.        ,  0.2529993 ,  0.30063739,  0.30063739,  0.62412506],
        [ 0.2529993 ,  1.        ,  0.62412506,  0.38953209,  0.30063739],
        [ 0.30063739,  0.62412506,  1.        ,  0.62412506,  0.30063739],
        [ 0.30063739,  0.38953209,  0.62412506,  1.        ,  0.2529993 ],
        [ 0.62412506,  0.30063739,  0.30063739,  0.2529993 ,  1.        ]])
        self.adapt_exp_kern = np.array([
        [ 1.        ,  0.31868771,  0.36787948,  0.36787948,  0.67554721],
        [ 0.23276223,  1.        ,  0.60653069,  0.36787948,  0.27949951],
        [ 0.07811997,  0.36787948,  1.        ,  0.36787948,  0.07811997],
        [ 0.27949951,  0.36787948,  0.60653069,  1.        ,  0.23276223],
        [ 0.67554721,  0.36787948,  0.36787948,  0.31868771,  1.        ]])

    def test_fix_gauss(self):
        kern = fix_gauss(self.coords, 3)
        np.testing.assert_allclose(kern, self.fix_gauss_kern)

    def test_adapt_gauss(self):
        kern = adapt_gauss(self.coords, 3)
        np.testing.assert_allclose(kern, self.adapt_gauss_kern)

    def test_fix_biqsquare(self):
        kern = fix_bisquare(self.coords, 3)
        np.testing.assert_allclose(kern, self.fix_bisquare_kern,
                atol=1e-01)

    def test_adapt_bisqaure(self):
        kern = adapt_bisquare(self.coords, 3)
        np.testing.assert_allclose(kern, self.adapt_bisquare_kern, atol=1e-012)
    
    def test_fix_exp(self):
        kern = fix_exp(self.coords, 3)
        np.testing.assert_allclose(kern, self.fix_exp_kern)

    def test_adapt_exp(self):
        kern = adapt_exp(self.coords, 3)
        np.testing.assert_allclose(kern, self.adapt_exp_kern)

if __name__ == '__main__':
    unittest.main()
