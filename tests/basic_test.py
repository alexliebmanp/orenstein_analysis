import os
from orenstein_analysis.measurement import loader
from orenstein_analysis.measurement import process
from orenstein_analysis.experiment import experiment_methods
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))+'/test_data/'

set_coord = lambda meas: process.define_dimensional_coordinates(meas, {'Polarization Angle (deg)':2*meas['Angle 1 (deg)']})

fit_bf = lambda meas: process.add_processed(meas, (experiment_methods.fit_birefringence, ['Polarization Angle (deg)', 'Demod x']))

ndim_meas = loader.load_ndim_measurement(path, {'x ($\mu$m)':'_x[0-9]+', 'y ($\mu$m)':'_y[0-9]+'}, instruction_set=[set_coord, fit_bf])
print(ndim_meas)

fig, [ax1, ax2, ax3]  = plt.subplots(1,3)

ndim_meas['Demod x (fit)'].sel({'x ($\mu$m)':2800, 'y ($\mu$m)':2475}, method='nearest').plot(ax=ax1, label='fit')
ndim_meas['Demod x'].sel({'x ($\mu$m)':2800, 'y ($\mu$m)':2475}, method='nearest').plot(ax=ax1, marker='o', ms=8, linestyle='None', label='data')
ax1.legend()
ax1.set_title('Birefringence fit at x = 2800 and y = 2475')

ndim_meas['Birefringence Angle'].plot(ax=ax2, cmap='twilight')
ax2.set_title(r'EuIn$_2$As$_2$')

xr.plot.hist(ndim_meas['Birefringence Angle'], ax=ax3)

fig.set_size_inches(12, 4, forward=True)
plt.tight_layout()
plt.show()
