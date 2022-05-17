'''
process.py

    General methods for processing data.

    TODO: make function(s) for stepping through a multidimensional array and carring out some operation or set of operations.

'''
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def add_data_to_measurement(measurement, var_name, var_data, dims=None, exclude=False):
    '''
    One of the most basic operations you can do to process a measurement is to add a layer of data to the dataset. This function handles that in a general manner. This is basically a wrapper around the expression MAYBE THIS NEEDS TO BE MODELED AFTER THE BASIC INITIZLIAZATION CODE.

                measurement[var_name] = (dims, var_data)

    If no dimensions are given, adds data as a dimensionless data variable.

    For example, you might take data as a function of time, fit to a decaying exponential to get a time constant, and then you want to add the time constant as a function of all the other coordinate variables.

    TODO: are there ways to check that data can actually be added to measurement and give useful feedback if not? If it finds and issue can it try to fix it? In general, what are the requirements for this? Also, exclude feature is ambigious.

    args:
        - measurement(Dataset):     Dataset over which to add data.
        - var_name(string):         name for new data.
        - var_data(DataArray):      data to add.

    returns:
        - modified_measurement(Dataset):

    kwargs**:
        - dims(string):             tuple of dimensions over which to add data. Defaults to not specifying dimensions.
        - exclude(bool):            If exclude is True, will add data over all coordinates except for coords.
    '''
    modified_measurement = measurement.copy()
    if dims==None:
        modified_measurement[var_name] = var_data
    else:
        if set(dims).issubset(set(list(measurement.dims)))==False:
            raise ValueError('dims '+str(dims)+' are not already dimensions of measurement object. Please add dimensions to measurement and try agian.')
        if exclude==True:
            dims = tuple([i for i in list(measurement.dims) if i not in dims])
        shape = tuple([measurement[i].shape[0] for i in dims])
        if shape!=var_data.shape:
            raise ValueError('Incompatible shapes. Shape of dimensions = '+str(shape)+', shape of data = '+str(var_data.shape))
        modified_measurement[var_name] = (dims, var_data)
    return modified_measurement

def add_processed(measurement, instruction_set, kwrags=None):
    '''
    Sequentially operate on measurement with function in instruction set and add new measurement variables according to the output of each function, which have the form of a dictionary with key:value pairs indicating.

    The hope for this function is that it can handle all kinds of dimensions and in particular can be used to process specific functions in multiple dimensions after initial processing. For example, adding a layer of fourier transform.
    '''
    for f in instruction_set:
        data_vars, coord_vars = f(measurement)
        measurement = add_data_to_measurement(measurement, data_vars, coord_vars)
    return measurement


def add_1D_fit(measurement, x_var, y_var, f, p0=None):
    '''
    fits a 1D scan to a function and returns another measurement with added data variables corresponding to fit values.

    This function should be used as a lambda function with x_var, y_var, and func specified in a particular context.

    TODO: can this be made more general? Also can it handle simpler data?

    args:
        - measurement(Dataset):
        x_var(string):              name of x variable in measurement
        y_var(string):              name of y variable in measurement
        f(func):                    function that takes x and y coordinates, any other kwargs, and outputs a dictionary of parameters to add to measurement.
    returns:
        - measurement(Dataset): another measurement with the fits
    '''
    x = measurement[x_var].data
    y = measurement[y_var].data
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise ValueError('Not 1D data. x and y data have dimensions '+str(len(x.shape))+' and '+str(len(y.shape))+' respectively.')
    popt_dict = f(x, y, p0)
    for var_name in list(popt_dict):
        var_data = np.array(popt_dict[var_name])
        measurement = add_data_to_measurement(measurement, var_name, var_data)
    return measurement

def define_dimensional_coordinates(measurement, coordinates):
    '''
    Adds a dimensional coordinate variable to measurement and replaces default dimensions on all data variables in the data to match new dimension. Only intended to work for 1 dimensional Datasets.

    args:
        - measurement(Dataset):     input measurement
        - coordinates:              dictionary of key:value for name and coordinate data

    returns:
        - modified_measurement(Dataset):
    '''
    if len(list(coordinates))>1:
        raise ValueError('ambiguous input. Coordinates dictionary must contain only one new coordinate.')
    coordinate_name = list(coordinates)[0]
    coordinate_data = coordinates[coordinate_name]
    if type(coordinate_data) is xr.core.dataarray.DataArray:
        coordinate_data = coordinate_data.data
    modified_measurement = measurement.copy()
    old_dims = list(modified_measurement.dims)
    rename_dict = {}
    for i in old_dims:
        if 'dim' in i:
            rename_dict[i] = coordinate_name
    modified_measurement = modified_measurement.rename(rename_dict)
    modified_measurement.coords[coordinate_name] = coordinate_data
    return modified_measurement


def add_dimensional_coordinates(measurement, coordinates):
    '''
    Adds dimensional coordinate variables and modifies all data variables to be functions of these new dimension in order as they appear in the dictionary.

    args:
        - measurement(Dataset):     input measurement
        - coordinates:              dictionary of key:value for name and coordinate data

    returns:
        - modified_measurement(Dataset):
    '''
    coordinate_names = list(coordinates)
    modified_measurement = measurement.copy()
    for coord_name in coordinate_names:
        coord_data = coordinates[coord_name]
        if type(coord_data) is xr.core.dataarray.DataArray:
            coord_data = coord_data.data
        modified_measurement.coords[coord_name] = coord_data
    for coord_name in coordinate_names[::-1]:
        for data_var in list(measurement.data_vars):
            modified_measurement[data_var] = modified_measurement[data_var].expand_dims(dim=coord_name)
    return modified_measurement
