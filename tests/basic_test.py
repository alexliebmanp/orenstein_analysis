from orenstein_analysis.measurement import loader
from orenstein_analysis.measurement import process
from orenstein_analysis.experiment import coordinates

path = './test_data/'
filename = 'EuIn2As2_S2_BR_13K_x2740_y2340.dat'

measurement = loader.load_measurement(path+filename)
measurement_mod = process.define_coordinates(measurement, coordinates.corotation_coordinates)

func = lambda meas: process.define_coordinates(meas, coordinates.corotation_coordinates)

#ndim_meas = loaders.load_ndim_measurement(path, ['x', 'y'], ['_x[0-9]+', '_y[0-9]+'], independent_variable='Angle 1 (deg)')
ndim_meas = loader.load_ndim_measurement(path, ['x (um)', 'y (um)'], ['_x[0-9]+', '_y[0-9]+'], instruction_set=[func])
