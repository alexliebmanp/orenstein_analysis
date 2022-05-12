from orenstein_analysis.source import loaders, processors

path = '/Users/oxide/orenstein/code/orenstein_analysis/test_data/'
filename = 'EuIn2As2_S2_BR_11K_x2940_y2460.dat'

path = '/Volumes/GoogleDrive-108258609193715139680/Shared drives/Orenstein Lab/Data/Elizabeth/EuIn2As2/220509/'

measurement = loaders.load_measurement(path+filename)
measurement_mod = processors.define_coordinates(measurement, processors.corotation_coordinates)

func = lambda meas: processors.define_coordinates(meas, processors.corotation_coordinates)

#ds_grid, coord_grid = loaders.load_ndim_measurement(path, ['x', 'y'], ['_x[0-9]+', '_y[0-9]+'], independent_variable=('Angle 1 (deg)', 'Angle (deg)'))
ds_grid, coord_grid = loaders.load_ndim_measurement(path, ['x', 'y'], ['_x[0-9]+', '_y[0-9]+'], instruction_set=[func])
#ds_grid, coord_grid = loaders.load_ndim_measurement(path, ['x', 'y'], ['bol', 'bleh'])
