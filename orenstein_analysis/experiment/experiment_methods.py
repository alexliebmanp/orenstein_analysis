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

    y(x) = a1*sin(2*x + phi1) + a2*sin(4*x + phi2) + b

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
    f = lambda var, a2, phi2, a1, phi1, b: a1*np.sin(2*(var+phi1)/180*np.pi) + a2*np.sin(4*(var+phi2)/180*np.pi) + b
    if p0==None:
        a10 = (1/2)*np.max(y)
        a20 = (1/2)*np.max(y)
        phi10 = 0
        phi20 = 0
        b = (1/2)*np.max(y)
        p0 = [a20, phi20, a10, phi10, b]
    popt, pcov = opt.curve_fit(f, x, y, p0=p0)
    xfit = np.linspace(x[0], x[-1],1000)
    yfit = np.asarray([f(i, *popt) for i in xfit])
    popt = redefine_fit_angles(popt)
    names = ['4Theta Amplitude', '4Theta Angle', 'Birefringence Amplitude', 'Birefringence Angle', 'Birefringence Offset']
    data_vars = {}
    coord_vars= {}
    for ii, name in enumerate(names):
        data_vars[name] = ((), popt[ii])
    coord_vars[x_var+' (fit)'] = xfit
    data_vars[y_var+' (fit)'] = ((x_var+' (fit)'), yfit)
    return data_vars, coord_vars


def redefine_fit_angles(params):
    '''
    helper function for fit_birefringence(), which
    '''
    if params[0]<0:
            params[0]=-params[0]
            params[1]=params[1]+180/4
    #postive amplitude 2theta
    if params[2]<0:
        params[2]=-params[2]
        params[3]=params[3]+180/2
    while params[1]>90:
        params[1]=params[1]-90
    while params[1]<0:
        params[1]=params[1]+90
    while params[3]<0:
        params[3]=params[3]+180
    while params[3]>180:
        params[3]=params[3]-180
    return params
