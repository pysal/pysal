import iwls
import numpy as np
import pysal
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import families

db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
y = np.array(db.by_col("HOVAL"))
y = np.reshape(y, (49,1))
X = []
X.append(np.ones(len(y)))
X.append(db.by_col("INC"))
X.append(db.by_col("CRIME"))
X = np.array(X).T

#from pysal.spreg import ols
#OLS = ols.OLS(y, X)
#print OLS.betas


glm_g = iwls.iwls(X, y, families.Gaussian(), None, None)

glm_p = iwls.iwls(X, y, families.Poisson(), None, None)



londonhp = pd.read_csv('/Users/toshan/projects/londonhp.csv')
y = londonhp['BATH2'].values
y = np.reshape(y, (316,1))
X = []
X.append(np.ones(len(y)))
X.append(londonhp['FLOORSZ'].values)
X = np.array(X).T
glm_b = iwls.iwls(X, y, families.Binomial(), None, None)

print glm_p[0]
print glm_b[0]
print glm_b[-1]
print glm_g[0]
