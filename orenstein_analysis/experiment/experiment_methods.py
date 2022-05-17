'''
experiment_methods.py

    Methods for processing data. Methods in this module should take a measurement as their first argumenet and return data_var and coord_var dictionaries with key:value pairs corresponding to variable names and data respectively, which are then eventually added to a measurement.

'''
import numpy as np
import pandas as pd
import xarray as xr
import scipy.optimize as opt
import matplotlib.pyplot as plt

def fit_birefringence(measurement, x_var, y_var, p0=None):
    '''
    fits a corotation scan to 2theta and 4theta components.

    y(x) = a1*cos(2*x - phi1) + a2*cos(4*x - phi2) + b

    args:
        - measurement(Dataset):
        - x(string):
        - y(string):

    returns:
        - data_vars:                dict of tuples (None, val) where the name is a fit parameter and val the corresponding value.
        - coord_vars:               set to None.

    *kwargs:
        - p0:                       list of arguments
    '''
    x = measurement[x_var].data
    y = measurement[y_var].data
    f = lambda var, a1, phi1, a2, phi2, b: a1*np.cos(2*var - phi1) + a1*np.cos(4*var - phi2) + b
    if p0==None:
        a10 = (1/2)*np.max(y)
        a20 = (1/2)*np.max(y)
        phi10 = 0
        phi20 = 0
        b = (1/2)*np.max(y)
        p0 = [a10, phi10, a20, phi20, b]
    popt, pcov = opt.curve_fit(f, x, y, p0=p0)
    names = ['Birefringence Amplitude', 'Birefringence Angle', '4Theta Amplitude', '4Theta Angle', 'Birefringence Offset']
    data_vars = {}
    for ii, name in enumerate(names):
        data_vars[name] = ((), popt[ii])
    return data_vars, None
