import os
from orenstein_analysis.measurement import loader
from orenstein_analysis.measurement import process
from orenstein_analysis.experiment import experiment_methods
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/Users/oxide/Documents/research/orenstein/code/orenstein-analysis/tests/test_data_2/'

set_coord = lambda meas: process.define_dimensional_coordinates(meas, {'Polarization Angle (deg)':2*meas['Angle 1 (deg)']})

fit_bf = lambda meas: process.add_processed(meas, (experiment_methods.fit_birefringence, ['Polarization Angle (deg)', 'Demod x']))

ndim_meas = loader.load_ndim_measurement(path, {'V (V)':'_V[0-9]', 'x ($\mu$m)':'_x[0-9]+', 'y ($\mu$m)':'_y[0-9]+'}, datavars_dict={'Capacitance (pF)':'[0-9]+.[0-9]+pF'}, instruction_set=[set_coord, fit_bf])

ndim_meas


sel = ndim_meas.sel({'x ($\mu$m)':1, 'y ($\mu$m)':1})

sel['Capacitance (pF)'].plot()
