'''
processors.py

    General methods for processing data.
'''
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def define_coordinates(measurement, coordinate_function):
    '''
    Given a function that acts on a measurement, defines the coordinates and dimensions in the measurement based on output of the function. Also goes through and renames all other dimensions to match the coordinate dimension that match the default expression 'dim'.

    For example, for a corotation scan the function can calculate the corotation angle based on Angle 1 and then modifies the coordinates to reflect this.

    args:
        - measurement(Dataset):
        - coordinate_function:  a function that returns the coordinate data, name and dimension
    returns:
        - modified_measurement(Dataset):
    '''

    dataset = measurement.copy()
    coord_data, coord_name, coord_dim = coordinate_function(dataset)
    old_dims = list(dataset.dims)
    rename_dict = {}
    for i in old_dims:
        if 'dim' in i:
            rename_dict[i] = coord_dim
    dataset = dataset.rename(rename_dict)
    dataset.coords[coord_name] = (coord_dim, coord_data)
    return dataset

def corotation_coordinates(measurement):
    '''
    Returns a corotation angle
    '''

    return 2*measurement['Angle 1 (deg)'].data, 'Polarization Angle', 'Angle (theta)'
