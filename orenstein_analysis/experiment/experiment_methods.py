'''
experiment_methods.py

    Methods for processing data.

'''
import numpy as np
import pandas as pd
import xarray as xr
import scipy.optimize as opt
import matplotlib.pyplot as plt

def fit_birefringence(x, y, p0=None):
    '''
    fits a corotation scan to 2theta and 4theta components.

    y(x) = a1*cos(2*x - phi1) + a2*cos(4*x - phi2) + b

    args:
        - x(float)
        - y(float)

    returns:
        - popt(float): optimal fit parameters [a1, phi1, a2, phi2, b]
    '''

    f = lambda x, a1, phi1, a2, phi2, b: a1*np.cos(2*x - phi1) + a1*np.cos(4*x - phi2) + b
    if p0==None:
        a10 = (1/2)*np.max(y)
        a20 = (1/2)*np.max(y)
        phi10 = 0
        phi20 = 0
        b = (1/2)*np.max(y)
        p0 = [a10, phi10, a20, phi20, b]
    popt, pcov = opt.curve_fit(f, x, y, p0=p0)
    names = ['Birefringence Amplitude', 'Birefringence Angle', '4Theta Amplitude', '4Theta Angle', 'Birefringence Offset']

    return popt, names
