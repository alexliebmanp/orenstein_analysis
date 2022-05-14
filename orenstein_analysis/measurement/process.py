'''
process.py

    General methods for processing data.

    TODO: make function(s) for stepping through a multidimensional array and carring out some operation or set of operations.

'''
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def add_data_to_measurement(measurement, var_name, var_data, dims, exclude=False):
    '''
    One of the most basic operations you can do to process a measurement is to add a layer of data over some set of dimensions. This function handles that in a general manner. This is basically a wrapper around the expression

                measurement[var_name] = (dims, var_data)

    For example, you might take data as a function of time, fit to a decaying exponential to get a time constant, and then you want to add the time constant as a function of all the other coordinate variables.

    TODO: are there ways to check that data can actually be added to measurement and give useful feedback if not? If it finds and issue can it try to fix it? In general, what are the requirements for this? Also, exclude feature is ambigious.

    args:
        - measurement(Dataset):     Dataset over which to add data.
        - var_name(string):         name for new data.
        - var_data(DataArray):      data to add.
        - dims(string):             tuple of dimensions over which to add data.

    returns:
        - modified_measurement(Dataset):

    kwargs**:
        - exclude(bool):            If exclude is True, will add data over all coordinates except for coords.
    '''
    if set(dims).issubset(set(list(measurement.dims)))==False:
        raise ValueError('dims '+str(dims)+' are not already dimensions of measurement object. Please add dimensions to measurement and try agian.')
    if exclude==True:
        dims = tuple([i for i in list(measurement.dims) if i not in dims])
    shape = tuple([measurement[i].shape[0] for i in dims])
    if shape!=var_data.shape:
        raise ValueError('Incompatible shapes. Shape of dimensions = '+str(shape)+', shape of data = '+str(var_data.shape))
    modified_measurement = measurement.copy()
    modified_measurement[var_name] = (dims, var_data)

    return modified_measurement

def add_1D_fit(measurement, x_var, y_var, f, p0=None):
    '''
    fits a 1D scan to a function and returns another measurement with added data variables corresponding to fit values.

    This function should be used as a lambda function with x_var, y_var, and func specified in a particular context.

    TODO: can this be made more general? Also can it handle simpler data?

    args:
        - measurement(Dataset):
        x_var(string):              name of x variable in measurement
        y_var(string):              name of y variable in measurement
    returns:
        - measurement(Dataset): another measurement with the fits
    '''
    x = measurement[x_var].data
    y = measurement[y_var].data
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise ValueError('Not 1D data. x and y data have dimensions '+str(len(x.shape))+' and '+str(len(y.shape))+' respectively.')
    popt, names = f(x, y, p0)
    dims = tuple([i for i in measurement.dims if i != x_var])
    nesting_depth = len(dims)
    for ii, p_data in enumerate(popt):
        var_data = p_data
        for i in range(nesting_depth):
            var_data = [var_data]
        var_data = np.array(var_data)
        measurement = add_data_to_measurement(measurement, names[ii], var_data, dims)
    return measurement

def define_coordinates(measurement, coordinate_function):
    '''
    Given a function that acts on a measurement, defines the coordinates and dimensions in the measurement based on output of the function. Also goes through and renames all other dimensions to match the coordinate dimension that match the default expression 'dim'.

    For example, for a corotation scan the function can calculate the corotation angle based on Angle 1 and then modifies the coordinates to reflect this.

    TODO: This seems like a poorly designed and syntaxed function.

    args:
        - measurement(Dataset):
        - coordinate_function:  a function that returns the coordinate name and data (in particular, this generates a dimensional coordinate)

    returns:
        - modified_measurement(Dataset):
    '''
    dataset = measurement.copy()
    coord_name, coord_data = coordinate_function(dataset)
    old_dims = list(dataset.dims)
    rename_dict = {}
    for i in old_dims:
        if 'dim' in i:
            rename_dict[i] = coord_name
    dataset = dataset.rename(rename_dict)
    dataset.coords[coord_name] = coord_data
    return dataset
