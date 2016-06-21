
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
#from pysal.contrib.spint.count_model import CountModel
#from pysal.contrib.glm.family import Poisson
import sys
sys.path.append('/Users/toshan/dev/pysal/pysal/contrib/spint/')
from count_model import CountModel
from dispersion import phi_disp, alpha_disp
sys.path.append('/Users/toshan/dev/pysal/pysal/contrib/glm')
from family import Poisson

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

