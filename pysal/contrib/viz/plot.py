"""
Moran Plot using PySAL and Matplotlib

"""

__author__ = "Marynia Kolak <marynia.kolak@gmail.com>"

import pandas as pd
import numpy as np
import pysal as ps
import matplotlib.pyplot as plt


def moran(var, w, xlabel='', ylabel='', title='', custom=(5,5)):
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

	
	## really shouldn't have w transformed here -- error if doesn't match size? (MK)
	w.transform = 'r' 
	slag = ps.lag_spatial(w, var)

	## Z-Score standardisation -- again, how to grab zx and zy from Moran call? (MK)
	y_std   = (var - var.mean())/var.std()
	yl_std  = (slag - slag.mean())/slag.std()

	## Customize plot
	fig1 = plt.figure(figsize=custom)
	plt.xlabel(xlabel, fontsize=20)
    	plt.ylabel(ylabel, fontsize=20)
    	plt.suptitle(title, fontsize=30)

	plt.scatter(y_std, yl_std, s=60, color='k', alpha=.6)
	
	## Add Moran line here -- how much do we assume has been done so far?

	plt.axvline(0, alpha=0.5)
	plt.axhline(0, alpha=0.5)

	plt.show()
	return None





