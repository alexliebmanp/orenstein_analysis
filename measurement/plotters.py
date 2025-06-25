
%matplotlib widget
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from IPython.display import display


def interactive_plot(ds, processed_variables, raw_variables, processed_coords, raw_coord, figsize=(12,5))

# define coordinate data
pc_data = [ds[coord].data for coord in processed_coords]
rc_data = ds[raw_coord].data

# define variable data
pv_data = [ds[var].data for var in processed_variables]
rv_data = [ds[var].data for var in raw_variables]

# Initial plot
fig, axs = plt.subplots(1, 2, figsize=figsize)
im = ax1.imshow(ds[var_xy].T, origin='lower', aspect=1,
                extent=[x.min(), x.max(), y.min(), y.max()])
ax1.set_title('var_xy (2D)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
line, = ax2.plot([], [])
ax2.set_title('var_angle (vs angle)')
ax2.set_xlabel('angle')
ax2.set_ylabel('value')

# Interactive click event
def on_click(event):
    if event.inaxes == ax1:
        # Convert x/y coords from plot space to data space
        x_click, y_click = event.xdata, event.ydata

        # Find nearest x/y index
        x_idx = np.argmin(np.abs(ds['x (um)'].values - x_click))
        y_idx = np.argmin(np.abs(ds['y (um)'].values - y_click))

        # Extract data
        angle_vals = ds['Polarization Angle (deg)'].values
        b_vals = ds[var_angle].isel({'x (um)':x_idx, 'y (um)':y_idx}).values

        # Update plot
        line.set_data(angle_vals, b_vals)
        ax2.set_xlim(angle_vals.min(), angle_vals.max())
        ax2.set_ylim(np.nanmin(b_vals), np.nanmax(b_vals))
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.tight_layout()
plt.show()
