"""
Canned Views using PySAL and Matplotlib
"""

__author__ = "Marynia Kolak <marynia.kolak@gmail.com>"

import pandas as pd
import numpy as np
import pysal as ps
import matplotlib.pyplot as plt

def mplot(m, xlabel='', ylabel='', title='', custom=(7,7)):
    '''
    Produce basic Moran Plot 
    ...
    Parameters
    ---------
    m            : array
                   values of Moran's I 
    xlabel       : str
                   label for x axis
    ylabel       : str
                   label for y axis                
    title        : str
                   title of plot
    custom       : tuple
                   dimensions of figure size

    Returns
    ---------
    plot         : png 
                    image file showing plot
            
    '''
    
    lag = ps.lag_spatial(m.w, m.z)
    fit = ps.spreg.OLS(m.z[:, None], lag[:,None])

    ## Customize plot
    fig = plt.figure(figsize=custom)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(title)

    plt.scatter(m.z, lag, s=60, color='k', alpha=.6)
    plt.plot(lag, fit.predy, color='r')

    plt.axvline(0, alpha=0.5)
    plt.axhline(0, alpha=0.5)
    plt.show()

    return None
