'''
process.py

    General methods for processing data.

    TODO: make function(s) for stepping through a multidimensional array and carring out some operation or set of operations.

'''
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def add_data_to_measurement(measurement, data_vars={}, coord_vars={}):
    '''
    This function handles one of the most basic operations you can do to a measurement: add a layer of data variables and coordinate variables to the dataset. This is essentially a wrapper around the expressions:

                        measurement[name] = (dims, data)

    Note that this differs in functionality from add_dimensional_coordinates() in that it does not modify any existing data or coordinate variables.

    args:
        - measurement(Dataset):

    returns:
        - modified_measurement(Dataset)

    **kwargs:
        - data_vars:                dictionary where keys are variable names and values are tuples of form (dims, data), where dims are tuples and data is a np.array. Alternatively, the value can be a DataArray.
        - coord_vars:               same form as data_vars, but for coordinate variables.
    '''
    modified_measurement = measurement.copy()
    if data_vars == {}:
        pass
    else:
        for name in list(data_vars):
            modified_measurement[name] = data_vars[name]
    if coord_vars == {}:
        pass
    else:
        for name in list(coord_vars):
            modified_measurement.coords[name] = coord_vars[name]
    return modified_measurement

def add_processed(measurement, function_set):
    '''
    Sequentially operate on measurement with function in function_set and add new measurement variables according to the output of each function, which have the form of data_vars and coord_vars dictionaries with keys as names for new variables. The function take measurement as first argument.

    Todo: figure out a good way to handle kwargs.

    args:
        - measurement(Dataset):
        - function_set:              list of tuples (function, arguments). each function must take measurement as first input and return data_vars and coord_vars.

    returns:
        - modified_measurement(Dataset)
    '''
    modified_measurement = measurement.copy()
    if type(function_set) is tuple:
        function_set = [function_set]
    for (f, args) in function_set:
        data_vars, coord_vars = f(measurement, *args)
        modified_measurement = add_data_to_measurement(modified_measurement, data_vars, coord_vars)
    return modified_measurement

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
    for coord_name in coordinate_names: # [::-1] to reverse order.
        for data_var in list(measurement.data_vars):
            modified_measurement[data_var] = modified_measurement[data_var].expand_dims(dim=coord_name)
    return modified_measurement

def reshape(measurement, coordinates):
    '''
        Takes a measurement with N-dimensional data stored as 1D and reshapes according to coordinates list. That, is for each label in coordinates, it treats that as a dimensional coordinate.

        args:
            - measurement:  1D measurement
            - coordinates:  list of data variables by which to reshape
    '''
    data_variables = list(measurement.data_vars.keys())
    for coord in coordinates:
        if coord not in data_variables:
            raise ValueError(f'Invalid coordinate {coord}. Please choose from list {data_variables}')
    coordinates_temp = []
    for var in data_variables:
        if var in coordinates:
            coordinates_temp.append(var)
    coordinates = coordinates_temp
    coordinates.reverse()
    coords_dict = {}
    coords_dims = []
    for coord in coordinates:
        coord_array = measurement[coord]
        coord_vect = np.unique(coord_array.values)
        coord_dim = len(coord_vect)
        coords_dict[coord] = coord_vect
        coords_dims.append(coord_dim)
        data_variables.remove(coord)
    reshape_tuple = tuple(coords_dims)

    data_vars_dict = {}
    for data_var in data_variables:
        data = measurement[data_var].values
        data = np.reshape(data, reshape_tuple)
        data_vars_dict[data_var] = (coordinates, data)

    reshaped_measurement = xr.Dataset(data_vars=data_vars_dict, coords=coords_dict)

    return reshaped_measurement
