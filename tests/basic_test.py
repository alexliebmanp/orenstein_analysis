# %% codecell
import os
from orenstein_analysis.measurement import loader
from orenstein_analysis.measurement import process
from orenstein_analysis.experiment import experiment_methods
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% codecell
#path = os.path.dirname(os.path.realpath(__file__))+'/test_data/'
path = '/Users/oxide/orenstein/code/orenstein-analysis/tests/test_data/'
filename = 'EuIn2As2_S2_BR_9K_x2740_y2540.dat'

set_coord = lambda meas: process.define_dimensional_coordinates(meas, {'Polarization Angle (deg)':2*meas['Angle 1 (deg)']})

fit_bf = lambda meas: process.add_processed(meas, (experiment_methods.fit_birefringence, ['Polarization Angle (deg)', 'Demod x']))

ndim_meas = loader.load_ndim_measurement(path, {'x ($\mu$m)':'_x[0-9]+', 'y ($\mu$m)':'_y[0-9]+'}, instruction_set=[set_coord, fit_bf])
ndim_meas['Birefringence Angle'].data

# %% codecell
ndim_meas['Birefringence Angle'].plot(cmap='twilight')
plt.title(r'EuIn$_2$As$_2$')

# %% codecell
xr.plot.hist(ndim_meas['Birefringence Angle'])
