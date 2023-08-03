'''
process.py

    General methods for processing data.

    TODO: make function(s) for stepping through a multidimensional array and carring out some operation or set of operations.

'''
import numpy as np
import pandas as pd
import xarray as xr
import itertools as iter
import matplotlib.pyplot as plt
import traceback

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
    if type(function_set) is tuple or hasattr(function_set, '__call__'):
        function_set = [function_set]
    for fs in function_set:
        if hasattr(fs, '__call__'):
            f = fs
            args=None
        else:
            f = fs[0]
            args=fs[1]
        if args==None:
            data_vars, coord_vars = f(measurement)
        else:
            data_vars, coord_vars = f(measurement, *args)
        modified_measurement = add_data_to_measurement(modified_measurement, data_vars, coord_vars)
    return modified_measurement

def add_processed_nd(measurement, function_set, coord_vars=[]):
    '''
    General utility for processing data after loading.

    Takes a measurement (Dataset) and for all values of data variables that are in coord_vars, acts the function_set and adds processed data to Dataset as function of all data variables not in data_vars. Coords must be dimensional coordinates!

    args:
        - measurement(Dataset):
        - function_set:              list of tuples (function, arguments). each function must take measurement as first input and return data_vars and coord_vars.

    returns:
        - modified_measurement(Dataset)
    '''

    if coord_vars==[]:
        coord_vars = list(measurement.coords)
    if type(function_set) is tuple:
        function_set = [function_set]
    other_coord_vars = list(measurement.coords)
    for coord in coord_vars:
        other_coord_vars.remove(coord)
    coord_data = []
    measurement_list = []
    for coord in other_coord_vars:
        coord_data.append(measurement[coord].data)
    coord_vals = gen_coordinates_recurse(coord_data, len(coord_data)-1)
    for vals in coord_vals:
            coords_dict = {}
            for ii, coord in enumerate(other_coord_vars):
                coords_dict[coord] = vals[ii]
            sub_measurement = measurement.sel(coords_dict)
            sub_measurement.drop_vars(coord_vars)
            modified_measurement = add_processed(sub_measurement, function_set)
            modified_measurement = add_dimensional_coordinates(modified_measurement, coords_dict)
            measurement_list.append(modified_measurement)
    try:
        return xr.combine_by_coords(measurement_list)
    except Exception:
        traceback.print_exc()
        return measurement_list

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
            - coordinates:  list of data or coord variables by which to reshape
    '''
    data_variables = list(measurement.data_vars.keys())+list(measurement.coords.keys())
    for coord in coordinates:
        if coord not in data_variables:
            raise ValueError(f'Invalid coordinate {coord}. Please choose from list {data_variables}')
    coordinates.reverse()
    coords_dict = {}
    coords_dims = []
    coords_vects = []
    coords_indx_vects = []
    for coord in coordinates:
        coord_array = measurement[coord]
        coord_vect = np.unique(coord_array.values)
        coord_dim = len(coord_vect)
        coords_dict[coord] = coord_vect
        coords_dims.append(coord_dim)
        coords_vects.append(coord_vect)
        coords_indx_vects.append(np.arange(coord_dim))
        data_variables.remove(coord)
    reshape_tuple = tuple(coords_dims)
    positions = list(iter.product(*coords_vects))
    indices = list(iter.product(*coords_indx_vects))

    data_vars_dict = {}
    for data_var in data_variables:
        #data = measurement[data_var].values
        #data = np.reshape(data, reshape_tuple)
        data = np.zeros(reshape_tuple)
        for ii, p in enumerate(positions):
            sel_dict = {}
            indx = indices[ii]
            for jj, coord in enumerate(coordinates):
                sel_dict[coord] = p[jj]
            try:
                data[indx] = measurement[data_var].sel(sel_dict)
        data_vars_dict[data_var] = (coordinates, data)

    reshaped_measurement = xr.Dataset(data_vars=data_vars_dict, coords=coords_dict)

    return reshaped_measurement

def gen_coordinates_recurse(range_list, n, pos_list=[], current_pos=None):
    '''    given an empty pos_list, and a range_list, recursively generates a list of positions that span the spacce in range_list. Note that positions are written from first entry in range_list to last.

    args:
        - range_list:       a list of np arrays, where each array is a range of interest.
        - n:                max index of arrays in range_list, needed for recursion
        - post_list:        should be an empty list which the function will append to
        - current_pos:      n+1 dim array that carries around the positions to append for each recursive iteration.

    returns:
        - post_list
    '''
    if n==len(range_list)-1:
        current_pos = np.asarray([i[0] for i in range_list])#np.asarray(range_list)[:,0]
        pos_list = []
    if n>=0:
        for i in range_list[n]:
            current_pos[n] = i
            pos_list = gen_coordinates_recurse(range_list, n-1, pos_list, current_pos)
    else:
        pos_list.append(np.copy(current_pos))

    return pos_list
