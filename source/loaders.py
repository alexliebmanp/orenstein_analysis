'''
data_loader.py

        Methods for importing data. They rely on a standardized method of storing data from measurements and more complicated measurement routines. As a general rule, measurements refer to 1 dimension datasets which measure 1 or multiple observables as a function of an independent variable. More complex datasets should be stored as measurement files in a directory, with filenames according to a set convention.
'''
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import glob
import re

def load_measurement(filename, independent_variable=None):
    '''
    loads a textfile data into a Dataset based on the headers. By default, this import method treats all data columns as dependent variables and the coordinates are with respect to some arbitrary integer independent variable. Alternatively, an independent variable can be specified.

    TODO: for some reason still getting dimension without coordinates when I specify the indep_var.

    args:
        - filename(string):     full path to file.
    returns:
        - measurement(Dataset): dataset representation of measurement

    kwargs:
        - independent_variable(string): tuple of form (name, dim) for independent variable to be used as the coordinate axis. If set to None, returns dataset without specifying coordinates.
    '''
    data = []
    header = []
    metadata = [] # add in later
    metaheader = []
    with open(filename, 'r') as fromfile:
        for number, line in enumerate(fromfile):
            if number == 0:
                header = line.strip().split('\t')
            else:
                try:
                    data.append([float(li) for li in line.split()])
                except:
                    continue
    data_dictionary = data_to_dictionary(header, np.array(data))
    measurement = dictionary_to_dataset(data_dictionary, independent_variable)
    return measurement


def load_ndim_measurement(directory, dimensions, regexp_list, independent_variable=None, instruction_set=[]):
    '''
    parses a directory for textfiles and stores data as a multidimensional Dataset, where each dimension is assigned based on parsing filenames according to parse_list. This function will form the basis of more specific loading functions.

    Ideally, this method will first try to parse based on metadata in file, and otherwise based on filename

    For example, load_ndim_measurements(dir, ['x', 'y'], ['_x[0-9]+_', '_[0-9]+_']) can be used to load in a 2D map.

    TODO: come up with algorithm to reshape ds_grid and coord_grid, implement it, and then use xr.combine_nested() to create mutlidimensional Dataset. Lastly extract the axes from coord_grid.

    args:
        - directory(string):                full path to data directory
        - dimensions(string):               name of each dimension
        - regex_list(string):               list of regex patterns for extracting each coordinate value from the file names. If set to None, will try to extract from Dataset attributes.

    returns:
        - ndim_measurement(Dataset):        dataset

    kwargs:
        - independent_variable(string):     tuple of form (name, dim) for independent variable to be used as the coordinate axis. If set to None, returns dataset without specifying coordinates.
        - instruction_set(functions):       list of functions to sequentially operate on each imported Dataset from left to right. Functions must take a Dataset as the only argument and return another Dataset.
    '''
    if len(dimensions)!=len(regexp_list) and regexp_list!=None:
        raise ValueError('number of dimensions to search does not match number of regular expressions for parsing filenames.')
    data_files = glob.glob(directory+'*.dat')
    ds_grid = []
    coord_grid = []
    for filename in data_files:
        ds = load_measurement(filename, independent_variable)
        for operation in instruction_set:
            ds = operation(ds)
        coord = []
        for regexp in regexp_list:
            match_list = re.findall(regexp, filename)
            if match_list == []:
                raise ValueError('no match found for regexp '+regexp+' in filename '+filename)
            elif len(match_list) == 2:
                raise ValueError('multiple matches found for regexp '+regexp+' in filename '+filename)
            else:
                p = re.compile('[0-9]+')
                val = float(p.search(match_list[0]).group())
                coord.append(val)
        ds_grid.append(ds)
        coord_grid.append(coord)

    return ds_grid, coord_grid


def data_to_dictionary(header, data):
    '''
    Construct dictionary of data columns, referenced by data-file headers.
    Args:
        - header(list of string):   data file headers
        - data(array of float):     dataset -- columns correspond to different headers

    Return:
        - dataset(dict):            dictionary with key:value pairs corresponding to header string: array of float
    '''
    dataset = {}
    if len(header) != np.shape(data)[1]:
        print(len(header))
        print(np.shape(data))
        raise ValueError('Invalid combination of headers and data columns')
    else:
        for index, hi in enumerate(header):
            dataset[hi] = data[:,index]
    return dataset

def dictionary_to_dataset(dictionary, independent_variable):
    '''
    Constructs a Dataset from dictionary by passing arguments to Dataset of the form: {header: (dim, data)}.

    Todo: better handling of multidimensional data

    args:
        - dictionary(float):            a dictionary of form header:data
        - independent_variable(string): tuple of form (name, dim) for independent variable to be used as the coordinate axis. If set to None, returns dataset without specifying coordinates.
    returns:
        - dataset(Dataset):
    '''
    if independent_variable!=None:
        if independent_variable[0] not in dictionary:
            print(independent_variable)
            raise ValueError('independent_variable not in dictionary')
    data_vars = {}
    coords = {}
    if independent_variable == None:
        for header, data in dictionary.items():
                data_vars[header] = xr.DataArray(data)
        dataset = xr.Dataset(data_vars)
    else:
        coord_name = independent_variable[0]
        coord_dim = independent_variable[1]
        coord_data = dictionary[coord_name]
        coords = {coord_name:(coord_dim, coord_data)}
        reduced_dictionary = dictionary.copy()
        del reduced_dictionary[coord_name]
        for ii, (header, data) in enumerate(reduced_dictionary.items()):
            data_vars[header] = (coord_dim, data)
        dataset = xr.Dataset(data_vars, coords=coords)
    return dataset
