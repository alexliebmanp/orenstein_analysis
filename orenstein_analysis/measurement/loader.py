'''
loader.py

        Methods for importing data. They rely on a standardized method of storing data from measurements and more complicated measurement routines. As a general rule, measurements refer to 1 dimension datasets which measure 1 or multiple observables as a function of an independent variable. More complex datasets should be stored as measurement files in a directory, with filenames according to a set convention.
'''
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import glob
import re
from orenstein_analysis.measurement import process

def load_measurement(filename, independent_variable=None, instruction_set=[]):
    '''
    loads a textfile data into a Dataset based on the headers. By default, this import method treats all data columns as dependent variables and the coordinates are with respect to some arbitrary integer independent variable. Alternatively, an independent variable can be specified.

    TODO: handle possibility of metadata in file -> attributes. For example, might want to have a system for keeping track of units.

    args:
        - filename(string):     full path to file.

    returns:
        - measurement(Dataset): dataset representation of measurement

    **kwargs:
        - independent_variable(string): name for independent variable to be used as the coordinate axis. If set to None, returns dataset without specifying coordinates.
        - instruction_set(func): list of functions to sequentially operate on each imported Dataset from left to right. Functions must take a Dataset as the only argument and return another Dataset.
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
    for operation in instruction_set:
        measurement = operation(measurement)
    return measurement


def load_ndim_measurement(directory, dimensions_dict, independent_variable=None, instruction_set=[]):
    '''
    parses a directory for textfiles and stores data as a multidimensional Dataset, where each dimension is assigned based on parsing filenames according to regexp_list or via metadata contained in each data file. In addition, a list of functions can be passed to this method which act sequentially to raw data in order to process it as it gets loaded.

    Ideally, this method will first try to parse based on metadata in file, and otherwise based on filename

    For example, load_ndim_measurements(dir, ['x', 'y'], ['_x[0-9]+_', '_[0-9]+_']) can be used to load in a 2D map.

    TODO: incorporate possibility for metadata stored in file. I also need to start thinking more deeply about the data pipeline - when you should additional dimensions be added? How can process operations handle arbitrary dimensions so as to avoid annoying bugs? Ie, I don't like that I add some diemsnions, then do operations, and then go back and modify the old variables. Either get rid of instruction set or pass it into load_measurement directly to make it cleanest. Make sure to be separating tasks into very specific functions.


    args:
        - directory(string):                full path to data directory
        - dimensions_dict(string):          dict with keys are dimensions and values are regex patterns for extracting coordinate value from filename.
        - regex_list(string):               list of regex patterns for extracting each coordinate value from the file names.

    returns:
        - ndim_measurement(Dataset):        dataset

    **kwargs:
        - independent_variable(string):     name for independent variable to be used as the coordinate axis. If set to None, returns dataset without specifying coordinates.
        - instruction_set(functions):       list of functions to sequentially operate on each imported Dataset from left to right. Functions must take a Dataset as the only argument and return another Dataset.
    '''
    dimensions = list(dimensions_dict)
    regexp_list = list(dimensions_dict.values())
    data_files = glob.glob(directory+'*.dat')
    measurement_list = []
    #coords_list = []
    for filename in data_files:
        measurement = load_measurement(filename, independent_variable)
        for operation in instruction_set:
            measurement = operation(measurement)
        coords = {}
        for ii, regexp in enumerate(regexp_list):
            match_list = re.findall(regexp, filename)
            if match_list == []:
                raise ValueError('no match found for regexp '+regexp+' in filename '+filename)
            elif len(match_list) == 2:
                raise ValueError('multiple matches found for regexp '+regexp+' in filename '+filename)
            else:
                p = re.compile('[0-9]+')
                val = float(p.search(match_list[0]).group())
                coords[dimensions[ii]] = val
        measurement = process.add_dimensional_coordinates(measurement, coords)
        measurement_list.append(measurement)

        #coords_list.append(coords)
    return xr.combine_by_coords(measurement_list)


def data_to_dictionary(header, data):
    '''
    Construct dictionary of data columns, referenced by data-file headers.

    args:
        - header(list of string):   data file headers
        - data(array of float):     dataset -- columns correspond to different headers

    return:
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
        - independent_variable(string): name for independent variable to be used as the coordinate axis. If set to None, returns dataset without specifying coordinates.

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
        coord_name = independent_variable
        coord_data = dictionary[coord_name]
        coords = {coord_name:coord_data}
        reduced_dictionary = dictionary.copy()
        del reduced_dictionary[coord_name]
        for ii, (header, data) in enumerate(reduced_dictionary.items()):
            data_vars[header] = (coord_name, data)
        dataset = xr.Dataset(data_vars, coords=coords)
    return dataset

def reshape_from_coordinates(obj_list, coords_list):
    '''
    Takes a list and reshapes it based on n-dimensional coordinates into a multidimensional nested list. This function is useful for combining datasets and is used in load_ndim_measurements.

    For example, suppose you have the list [o3, o1, o4, o2] with associated [x,y] coordinates [[x2, y1], [x1, y1], [x2, y2], [x1, y2]] == [c3, c1, c4, c2]. This will get reshaped into:

                [[o2, o4],          and         [[c2, c4],
                 [o1, o3]]                       [c1, c3]]

    UNDER CONSTRUCTION: may not be necessary to make this function since a good solution to this problem is simply to use combine_by_coords().

    args:
        - obj_list:             any 1 dimensional list
        - coord_list(float):    list of coordinates associated with each object. Coordinates are specified as another list of length n (one for each dimension of the coordinate space).

    returns:
        -
        -
    '''
