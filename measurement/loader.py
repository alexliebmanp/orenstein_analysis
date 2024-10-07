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
import os
from orenstein_analysis.measurement import process
import time
import traceback
from tqdm.auto import tqdm

def load_measurement(filename, independent_variable=None, instruction_set=[], add_metadata=True):
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
    metadata = {}
    read_head = 'Data'
    data_start = 0
    metadata_start = 0
    with open(filename, 'r') as fromfile:
        for number, line in enumerate(fromfile):
            if '[Metadata]' in line:
                read_head = 'Metadata'
                metadata_start = number+1
            elif '[Data]' in line:
                read_head = 'Data'
                data_start = number+1
            else:
                if read_head == 'Metadata':
                    property, val = line.split(':\t')
                    if val=='None\n':
                        val = np.nan
                    try:
                        val = float(val)
                    except:
                        pass
                    metadata[property] = val
                elif read_head == 'Data' and number == data_start:
                    header = line.strip().split('\t')
                else:
                    try:
                        data.append([float(li) for li in line.split()])
                    except Exception:
                        print(f'Execption in file {filename} line {number}.')
                        traceback.print_exc()
                        break
    data_dictionary = data_to_dictionary(header, np.array(data))
    measurement = dictionary_to_dataset(data_dictionary, independent_variable)
    if add_metadata:
        measurement.attrs = metadata
    for operation in instruction_set:
        measurement = operation(measurement)
    return measurement

def load_ndim_measurement(directory, dimensions_dict, datavars_dict={}, search_string='', independent_variable=None, instruction_set=[], add_metadata=True, print_flag=False):
    '''
    parses a directory for textfiles and stores data as a multidimensional Dataset, where each dimension is assigned based on parsing filenames according to regexp_list or via metadata contained in each data file. In addition, a list of functions can be passed to this method which act sequentially to raw data in order to process it as it gets loaded.

    Ideally, this method will first try to parse based on metadata in file, and otherwise based on filename

    For example, load_ndim_measurements(dir, ['x', 'y'], ['_x[0-9]+_', '_[0-9]+_']) can be used to load in a 2D map.

    TODO: incorporate possibility for metadata stored in file. I also need to start thinking more deeply about the data pipeline - when you should additional dimensions be added? How can process operations handle arbitrary dimensions so as to avoid annoying bugs? Ie, I don't like that I add some diemsnions, then do operations, and then go back and modify the old variables. Either get rid of instruction set or pass it into load_measurement directly to make it cleanest. Make sure to be separating tasks into very specific functions.


    args:
        - directory(string):                full path to data directory
        - dimensions_dict(string):          dict with keys are dimensions and values are regex patterns for extracting coordinate value from filename.

    returns:
        - ndim_measurement(Dataset):        dataset

    **kwargs:
        - independent_variable(string):     name for independent variable to be used as the coordinate axis. If set to None, returns dataset without specifying coordinates.
        - search_string:                    additional string that filename must match
        - datavars_dict:                        a dictionary like dimensions_dict that parses the filename and extracts value, adding to dataset as a data variable
        - instruction_set(functions):       list of functions to sequentially operate on each imported Dataset from left to right. Functions must take a Dataset as the only argument and return another Dataset.
        - print_flag(bool):                      if True, prints filenames as they get processed. Mainly for troubleshooting.
    '''
    dimensions = list(dimensions_dict)
    regexp_list = list(dimensions_dict.values())
    datavars = list(datavars_dict)
    dregexp_list = list(datavars_dict.values())
    data_files = glob.glob(directory+'*.dat')
    if len(data_files)==0:
        print ('No files found, check path.')
        return 0
    measurement_list = []
    #coords_list = []
    lookat_files = []
    for filename in data_files:
        look_at_file = True
        for regexp in regexp_list:
            match = re.search(regexp, filename)
            if not bool(match):
                look_at_file = False
        for regexp in dregexp_list:
            match = re.search(regexp, filename)
            if not bool(match):
                look_at_file = False
        if search_string!='':
            match = re.search(search_string, filename)
            if not bool(match):
                look_at_file = False
        if look_at_file==True:
            lookat_files.append(filename)
    if len(lookat_files)==0:
        print('No files match search criteria.')
        return 0
    num_files = len(lookat_files)
    for ii in tqdm(range(num_files)):
        filename = lookat_files[ii]
        #os.system('clear')
        #print(f'Processing file {num_files} of {len(data_files)} files.')
        if print_flag==True:
            print(filename)
        measurement = load_measurement(filename, independent_variable, add_metadata=add_metadata)
        data_vars = {}
        for ii, dregexp in enumerate(dregexp_list):
            match_list = re.findall(dregexp, filename)
            if len(match_list)==2:
                raise ValueError('multiple matches found for regexp '+regexp+' in filename '+filename)
            else:
                p = re.compile('-?[0-9]+[\.]?[0-9]*') # matches arbitrary decimal value
                val = float(p.search(match_list[0]).group())
                data_vars[datavars[ii]] = val
        measurement = process.add_data_to_measurement(measurement, data_vars=data_vars)
        for operation in instruction_set:
            measurement = operation(measurement)
        coords = {}
        for ii, regexp in enumerate(regexp_list):
            match_list = re.findall(regexp, filename)
            if len(match_list)==2:
                raise ValueError('multiple matches found for regexp '+regexp+' in filename '+filename)
            else:
                p = re.compile('-?[0-9]+[\.]?[0-9]*') # matches arbitrary decimal value
                val = float(p.search(match_list[0]).group())
                coords[dimensions[ii]] = val
        measurement = process.add_dimensional_coordinates(measurement, coords)
        measurement_list.append(measurement)

    #coords_list.append(coords)
    #measurement_list = match_measurement_length_1D(measurement_list)
    try:
        return xr.combine_by_coords(measurement_list, combine_attrs='override')
    except Exception:
        traceback.print_exc()
        return measurement_list

def match_measurement_length_1D(measurement_list):
    '''
    pads data arrays in measurements from measurement_list to match longest data array and replaces coordinates with longest coordinate array. This allows a combine_by_coords operation for measurement sets with different lengths. Assumes data spacing in each measurement is equivalent, all keys are the same, and all coordinate arrays are the same (up to extrapolation)

    Under construction
    '''

    measurement_list_mod = []
    key = list(measurement_list[0].data_vars)[0]
    maxlen = 0
    maxind = 0
    lens = np.zeros(len(measurement_list))
    for ii, meas in enumerate(measurement_list):
        curlen = len(meas[key].data[-1])
        lens[ii] = curlen
        if curlen > maxlen:
            maxlen=curlen
            maxind=ii

    template_meas = measurement_list[ii]
    coords = list(template_meas.coords)
    data_vars = list(template_meas.data_vars)
    new_coords = [template_meas[coord].data for coord in coords]
    for ii, meas in enumerate(measurement_list):
        meas_mod = meas.copy()
        if lens[ii] < maxlen:
            for jj, coord in enumerate(coords):
                #meas_mod[coord].data = new_coords[jj]
                pass
            for data_var in data_vars:
                data = meas_mod[data_var].data
                data_mod = np.pad(data[-1], ((0,int(maxlen-lens[ii]))))
                meas_mod.assign(data_var=data_mod)
        measurement_list_mod.append(meas_mod)

    for meas in measurement_list_mod:
        print(meas[key].data.shape)

    return measurement_list_mod

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
        print('header: '+str(header))
        print('header length: '+str(len(header)))
        print('data shape = '+str(np.shape(data)))
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

def save_data_to_file(fname, data, header, metadata=None):
    '''
    utility function for saving data to a file, with optional metadata

    args:
        - fname(string):           full path to datafile
        - data(array):             (n,m) array containing data
        - header(array):           (m) array of strings labeling each column
        - metadata(dict):          a dictionary of parameters to store at the top of the datafile under [Metadata]

    returns: None
    '''
    if not(len(header) == len(data[0,:])):
        raise ValueError('number of header items does not match number of data columns.')
    with open(fname, 'w') as f:
        if not(metadata==None):
            f.write('[METADATA]\n')
            for key, value in list(metadata.keys()):
                f.write(f'{key}:\t{value}\n')
            f.write('[DATA]\n')
        for item in header:
            f.write(str(item)+'\t')
        f.write('\n')
        for line in data:
            for item in line:
                f.write(str(item)+'\t')
            f.write('\n')
