
"""
Tests for regressiom based dispersion tests (Cameron & Trivedi, 2013)

Cameron, Colin A. & Trivedi, Pravin K. (2013) Regression Analysis of Count Data.
    Camridge University Press: New York, New York. 

"""

__author__ = 'Taylor Oshan tayoshan@gmail.com'

import unittest
import numpy as np
import pysal
from pysal.contrib.spint.count_model import CountModel
from pysal.contrib.glm.family import Poisson
from pysal.contrib.spint.dispersion import phi_disp, alpha_disp

class TestDispersion(unittest.TestCase):

    def setUp(self):
        db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
        y =  np.array(db.by_col("HOVAL"))
        y = np.reshape(y, (49,1))
        self.y = np.round(y).astype(int) 
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T

    def test_Dispersion(self):
        model = CountModel(self.y, self.X, family=Poisson())
        results = model.fit('GLM')
        phi = phi_disp(results)
        alpha1 = alpha_disp(results)
        alpha2 = alpha_disp(results, lambda x:x**2)
        np.testing.assert_allclose(phi, [ 5.39968689,  2.3230411 ,  0.01008847],
                atol=1.0e-8)
        np.testing.assert_allclose(alpha1, [ 4.39968689,  2.3230411 ,
            0.01008847], atol=1.0e-8)
        np.testing.assert_allclose(alpha2, [ 0.10690133,  2.24709978,
            0.01231683], atol=1.0e-8)


if __name__ == '__main__':
	    unittest.main()

