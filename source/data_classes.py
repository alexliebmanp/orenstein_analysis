#import my standard data modules.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate
from button_plotter import ButtonPlotter



class Measurement:
    '''
    bin for textfile data.
    '''

    def __init__(self, filename):

        self.filename = filename
        self.header, self.dataset = self.load_data()

    def load_data(self):
        '''
        loads data from a textfile, taking the first row as headers. Subsequent rows are data.

        args:
            - filename (string): location of data file

        returns:
            - header (list of string): data headers
            - dictionary of header/data
        '''

        data = []
        header = []
        with open(self.filename, 'r') as fromfile:
            for number, line in enumerate(fromfile):
                if number == 0:
                    header = line.strip().split('\t')
                else:
                    try:
                        data.append([float(li) for li in line.split()])
                    except:
                        continue
        return header, self.data_to_dictionary(header, np.array(data))

    def data_to_dictionary(self, header, data):
        '''
        Construct dictionary of data columns, referenced by data-file headers.
        Args:
            - header (list of string): data file headers
            - data (array of float): dataset -- columns correspond to different headers

        Return:
            - dataset: dictionary with key:value pairs corresponding to header string: array of float
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

    def data_interpolation(self, abscissa_data, ordinate_data, resample_abscissa):
        '''
        Interpolate data over a specified range and number of points. Our
        time series data often means that different experiments have different
        shapes. This method enables us to resample each set at the same spacing.
        Sampling region must be interpolative, not extrapolative.

        args:
            - abscissa_data (array of float): numeric data for domain
            - ordinate_data (array of float): numeric data
            - resample_abscissa (list of 3 numeric): start, stop and number of points

        return:
            - new_abscissa (array of float): new abscissa data points
            - interpolated (array of float): interpolated dataset
        '''
        if resample_abscissa[0]<abscissa_data.min() or resample_abscissa[1]>abscissa_data.max():
            raise ValueError('Invalid interpolation region. Exceeding data limits')
        if abscissa_data[0] > abscissa_data[-1]:
            abscissa_data = abscissa_data[::-1]
            ordinate_data = ordinate_data[::-1]
        new_abscissa = np.linspace(resample_abscissa[0], resample_abscissa[1], int(resample_abscissa[2]))
        interpolated = np.interp(new_abscissa, abscissa_data, ordinate_data)
        return new_abscissa, interpolated

    def explore(self):
        """
        Interactive matplotlib widget for quickly visualizing an experiment by
        selecting absicca and ordinate axes from headers for a plot.
        """
        plotter = ButtonPlotter(self.headers, self.dataset)
        plotter.execute()

    def print_fields(self):
        """
        Helper function. Print the available data fields to utilize.
        """
        for hi in self.headers:
            print(hi)


class Corotation(Measurement):
    '''
    In the Orenstein Lab, a very common modality for data acquisition is the corotation scan. This class is specifically tailored to such data.
    '''

    def __init__(self, filename):

        self.filename = filename
        self.header, self.dataset = self.load_data()



class Map:
    '''
    Abstractly, a map is an association between (x,y) points and a set of information (ie, values of a pixel). The information can be as simple as a a float, however it can also be more complex such as a Measurement (in the case of a Corotation map, for example). This class lays the foundation for handling such a data structure.
    '''
