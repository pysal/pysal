"""
Tests for CountModel class, which includes various linear models designed for
count data

Test data is the Columbus dataset after it has been rounded to integers to act
as count data. Results are verified using corresponding functions in R.

"""

__author__ = 'Taylor Oshan tayoshan@gmail.com'

import unittest
import numpy as np
import pysal
from pysal.contrib.spint.count_model import CountModel
from pysal.contrib.glm.family import Poisson

class TestCountModel(unittest.TestCase):
    """Tests CountModel class"""

    def setUp(self):
        db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
        y =  np.array(db.by_col("HOVAL"))
        y = np.reshape(y, (49,1))
        self.y = np.round(y).astype(int)
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T

    def test_PoissonGLM(self):
        model = CountModel(self.y, self.X, family=Poisson())
        results = model.fit('GLM')
        np.testing.assert_allclose(results.params, [3.92159085,  0.01183491,
            -0.01371397], atol=1.0e-8)
        self.assertIsInstance(results.family, Poisson)
        self.assertEqual(results.n, 49)
        self.assertEqual(results.k, 3)
        self.assertEqual(results.df_model, 2)
        self.assertEqual(results.df_resid, 46)
        np.testing.assert_allclose(results.yhat, 
                [ 51.26831574,  50.15022766,  40.06142973,  34.13799739,
                28.76119226,  42.6836241 ,  55.64593703,  34.08277997,
                40.90389582,  37.19727958,  23.47459217,  26.12384057,
                29.78303507,  25.96888223,  29.14073823,  26.04369592,
                34.18996367,  32.28924005,  27.42284396,  72.69207879,
                33.05316347,  36.52276972,  49.2551479 ,  35.33439632,
                24.07252457,  31.67153709,  27.81699478,  25.38021219,
                24.31759259,  23.13586161,  48.40724678,  48.57969818,
                31.92596006,  43.3679231 ,  34.32925819,  51.78908089,
                34.49778584,  27.56236198,  48.34273194,  57.50829097,
                50.66038226,  54.68701352,  35.77103116,  43.21886784,
                40.07615759,  49.98658004,  43.13352883,  40.28520774,
                46.28910294])
        np.testing.assert_allclose(results.cov_params, 
                [[  1.70280610e-02,  -6.18628383e-04,  -2.21386966e-04],
                [ -6.18628383e-04,   2.61733917e-05,   6.77496445e-06],
                [ -2.21386966e-04,   6.77496445e-06,   3.75463502e-06]])
        np.testing.assert_allclose(results.std_err, [ 0.13049161,  0.00511599,
            0.00193769], atol=1.0e-8)
        np.testing.assert_allclose(results.pvalues, [  2.02901657e-198,
            2.07052532e-002,   1.46788805e-012])
        np.testing.assert_allclose(results.tvalues, [ 30.0524361 ,   2.31331634,
            -7.07748998])
        np.testing.assert_allclose(results.resid,
                [ 28.73168426,  -5.15022766, -14.06142973,  -1.13799739,
                -5.76119226, -13.6836241 ,  19.35406297,   2.91722003,
                12.09610418,  58.80272042,  -3.47459217,  -6.12384057,
                12.21696493,  17.03111777, -11.14073823,  -7.04369592,
                7.81003633,  27.71075995,   3.57715604,   8.30792121,
                -13.05316347,  -6.52276972,  -1.2551479 ,  17.66560368,
                -6.07252457, -11.67153709,   6.18300522,  -2.38021219,
                7.68240741,  -1.13586161, -16.40724678,  -8.57969818,
                -7.92596006, -15.3679231 ,  -7.32925819, -15.78908089,
                8.50221416,  -4.56236198,  -8.34273194,   4.49170903,
                -8.66038226, -10.68701352,  -9.77103116,  -9.21886784,
                -12.07615759,  26.01341996,  -1.13352883, -13.28520774,
                -10.28910294])
        self.assertAlmostEqual(results.deviance, 230.46013824817649)
        self.assertAlmostEqual(results.llf, -247.42592089969378)
        self.assertAlmostEqual(results.AIC, 500.85184179938756)
        self.assertAlmostEqual(results.D2, 0.388656011675)
        self.assertAlmostEqual(results.adj_D2, 0.36207583826952761)

if __name__ == '__main__':
	    unittest.main()

