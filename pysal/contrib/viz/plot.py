"""
Moran Plot using PySAL and Matplotlib

"""

__author__ = "Marynia Kolak <marynia.kolak@gmail.com>"

import pandas as pd
import numpy as np
import pysal as ps
import matplotlib.pyplot as plt


def Moran(var, w, **kwargs):
	'''
    Produce basic Moran Plot 
    ...

    Arguments
    ---------

    var	         	: array
                      values of variable
    w       	    : array
                      values of spatial weight
    '''
	w.transform = 'r'
	slag = ps.lag_spatial(w, var)

	 # Z-Score standardisation
	y_std   = (var - var.mean())/var.std()
	yl_std  = (slag - slag.mean())/slag.std()

	fig1 = plt.figure(figsize=(5,5))

	plt.scatter(y_std, yl_std, s=60, color='k', alpha=.6)

	plt.axvline(0, alpha=0.5)
	plt.axhline(0, alpha=0.5)

	plt.show()
	return None





