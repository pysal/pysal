"""
Canned Views using PySAL and Matplotlib
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
    var	            : array
                      values of variable
    w       	    : spatial weight
    '''

	w.transform = 'r'
	slag = ps.lag_spatial(w, var)

	zx   = (var - var.mean())/var.std()
	zy  = (slag - slag.mean())/slag.std()

	fit = ps.spreg.OLS(zx[:, None], zy[:,None])

	## Customize plot
	fig1 = plt.figure(figsize=custom)
	plt.xlabel(xlabel, fontsize=20)
	plt.ylabel(ylabel, fontsize=20)
	plt.suptitle(title, fontsize=30)

	plt.scatter(zx, zy, s=60, color='k', alpha=.6)
	plot(zy, fit.predy, color='r')

	plt.axvline(0, alpha=0.5)
	plt.axhline(0, alpha=0.5)

	plt.show()

	return None
