'''
experiment_methods.py

    Methods for processing data. Methods in this module should take a measurement as their first argumenet and return data_var and coord_var dictionaries with key:value pairs corresponding to variable names and data respectively, which are then eventually added to a measurement.

'''
import numpy as np
import pandas as pd
import xarray as xr
import scipy.optimize as opt
import scipy.fft as fft
import scipy.signal as sig
import scipy as sp
import matplotlib.pyplot as plt
import orenstein_analysis.experiment.fit_functions as ff

def fit_birefringence(measurement, x_var, y_var, p0=None):
    '''
    fits a corotation scan to 2theta and 4theta components.

    y(x) = a1*sin(2*x + phi1) + a2*sin(4*x + phi2) + b

    args:
        - measurement(Dataset):
        - x(string):
        - y(string):

    returns:
        - data_vars:                dict of tuples (None, val) where the name is a fit parameter and val the corresponding value.
        - coord_vars:               set to None.

    *kwargs:
        - p0:                       list of arguments
    '''
    x = measurement[x_var].data.flatten()
    y = measurement[y_var].data.flatten()
    f = lambda var, a2, phi2, a1, phi1, b: a1*np.sin(2*(var+phi1)/180*np.pi) + a2*np.sin(4*(var+phi2)/180*np.pi) + b
    if p0==None:
        a10 = (1/2)*np.max(y)
        a20 = (1/2)*np.max(y)
        phi10 = 0
        phi20 = 0
        b = (1/2)*np.max(y)
        p0 = [a20, phi20, a10, phi10, b]
    popt, pcov = opt.curve_fit(f, x, y, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    xfit = np.linspace(x[0], x[-1],1000)
    yfit = np.asarray([f(i, *popt) for i in xfit])
    popt = redefine_fit_angles(popt)
    names = [f'4Theta Amplitude ({y_var})', f'4Theta Angle ({y_var})', f'Birefringence Amplitude ({y_var})', f'Birefringence Angle ({y_var})', f'Birefringence Offset ({y_var})']
    data_vars = {}
    coord_vars= {}
    attrs = {}
    for ii, name in enumerate(names):
        data_vars[name] = ((), popt[ii])
        data_vars[f'{name} Variance'] = ((), perr[ii])
    diagonal = popt[2]*np.cos(2*popt[3]*np.pi/180)
    off_diagonal = popt[2]*np.sin(2*popt[3]*np.pi/180)
    diagonal_var = np.sqrt((np.cos(2*popt[3]*np.pi/180)**2)*perr[2]**2 + ((2*popt[2]*np.sin(2*popt[3]*np.pi/180))**2)*(perr[3]*np.pi/180)**2) # error propagation formula
    off_diagonal_var = np.sqrt((np.sin(2*popt[3]*np.pi/180)**2)*perr[2]**2 + ((2*popt[2]*np.cos(2*popt[3]*np.pi/180))**2)*(perr[3]*np.pi/180)**2) # error propagation formula
    data_vars[f'Diagonal Birefringence ({y_var})'] = ((), diagonal)
    data_vars[f'Off-diagonal Birefringence ({y_var})'] = ((), off_diagonal)
    data_vars[f'Diagonal Birefringence ({y_var}) Variance'] = ((), diagonal_var)
    data_vars[f'Off-diagonal Birefringence ({y_var}) Variance'] = ((), off_diagonal_var)
    coord_vars[x_var+f' ({y_var} fit)'] = xfit
    data_vars[y_var+' (fit)'] = ((x_var+f' ({y_var} fit)'), yfit)
    return data_vars, coord_vars, attrs

def meas_fft(measurement, x_var, y_var, padding_fraction=1):

    xdata = measurement[x_var].data
    ydata = measurement[y_var].data
    dx = xdata[1]-xdata[0]
    if padding_fraction!=0:
        #dx = xdata[1]-xdata[1] # is this a bug?
        l = len(xdata)
        xdata = np.concatenate([xdata,xdata[-1]+dx*(1+np.arange(int(padding_fraction*l)))])
        ydata = np.concatenate([ydata,np.zeros(int(padding_fraction*l))])
    yfft = fft.fft(ydata)*dx
    freqs = fft.fftfreq(len(xdata), dx)
    sortindices = np.argsort(freqs)
    freqs = freqs[sortindices]
    yfft = yfft[sortindices]
    data_vars = {}
    coord_vars= {}
    attrs = {}
    freqname = f'{x_var} freqs'
    if x_var == 'Time Delay (ps)':
        freqname = 'Frequency (THz)'
        freqs = freqs
    coord_vars[freqname] = freqs
    data_vars[f'FFT[{y_var}]'] = ((freqname), yfft)
    data_vars[f'P[{y_var}]'] = ((freqname), np.real(yfft*np.conjugate(yfft))) # used to be np.abs(yfft)**2, which is equivalent
    data_vars[f'Re(FFT[{y_var}])'] = ((freqname), np.real(yfft))
    data_vars[f'Im(FFT[{y_var}])'] = ((freqname), np.imag(yfft))
    return data_vars, coord_vars, attrs

def savgol_smooth(measurement, x_var, y_var, window_length=None, window_frac=0.1, polyorder=4):
    if window_length is None:
        window_length=int(len(ydata)*window_frac)
    ydata = measurement[y_var].data
    ydata_smoothed = sig.savgol_filter(ydata, window_length=window_length, polyorder=polyorder)
    data_vars = {f'{y_var} smoothed':((x_var), ydata_smoothed)}
    coord_vars= {}
    attrs = {}
    return data_vars, coord_vars, attrs

def savgol_subtract(measurement, x_var, y_var, window_frac=1, polyorder=2):

    ydata = measurement[y_var].data
    background = sig.savgol_filter(ydata, window_length=int(len(ydata)*window_frac), polyorder=polyorder)
    ydata_subtract = ydata - background
    data_vars = {}
    coord_vars= {}
    data_vars[f'{y_var} background'] = ((x_var), background)
    data_vars[f'{y_var} BS'] = ((x_var), ydata_subtract)
    attrs = {}
    return data_vars, coord_vars, attrs

def fit_ndamped_meas(measurement, x_var, y_vars, n_damped, n_over_damped, p0=None, freq_bounds=None, bounds=None, normalize=False, kwargs_dict={}, method='least_squares'):

    if type(y_vars) is not list:
        y_vars = [y_vars]
    x = measurement[x_var].data.flatten()
    ys = [measurement[y_var].data.flatten() for y_var in y_vars]
    opt_params, var, x_fit, ys_fit, ys_guess = fit_ndamped(x, ys, n_damped, n_over_damped, p0, bounds=bounds, freq_bounds=freq_bounds, normalize=normalize, kwargs_dict=kwargs_dict, method=method)
    freqs, damps, cs, d_amps, d_phis, od_amps1, od_amps2 = opt_params
    var_freqs, var_damps, var_cs, var_d_amps, var_d_phis, var_od_amps1, var_od_amps2 = var
    data_vars = {}
    coord_vars= {}
    attrs = {}
    #coord_vars['Mode Params'] = ['Frequency (THz)', 'Damping (THz)', 'Amplitude', 'Phase', 'Type']
    for ii, y_var in enumerate(y_vars):
        coord_vars[x_var+f' ({y_var} fit)'] = x_fit
        data_vars[f'{y_var} (fit)'] = ((x_var+f' ({y_var} fit)'), ys_fit[ii])
        data_vars[f'{y_var} (fit guess)'] = ((x_var+f' ({y_var} fit)'), ys_guess[ii])
        for jj, f in enumerate(freqs):
            data_vars[f'Fit Offset ({y_var})'] = ((), cs[ii])
            data_vars[f'Fit Offset Variance ({y_var})'] = ((), var_cs[ii])
            if jj<n_damped:
                data_vars[f'Mode {jj+1} Amplitude ({y_var})'] = ((), d_amps[ii][jj])
                data_vars[f'Mode {jj+1} Amplitude Variance ({y_var})'] = ((), var_d_amps[ii][jj])
                data_vars[f'Mode {jj+1} Phase ({y_var})'] = ((), d_phis[ii][jj])
                data_vars[f'Mode {jj+1} Phase Variance ({y_var})'] = ((), var_d_phis[ii,jj])
                data_vars[f'Mode {jj+1} Type ({y_var})'] = ((), 'under damped')
            else:
                data_vars[f'Mode {jj+1} Amplitude 1 ({y_var})'] = ((), od_amps1[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} Amplitude Variance ({y_var})'] = ((), var_od_amps1[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} Amplitude 2 ({y_var})'] = ((), od_amps2[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} Amplitude 2 Variance ({y_var})'] = ((), var_od_amps2[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} Type ({y_var})'] = ((), 'over damped')      
    for ii, f in enumerate(freqs):
            data_vars[f'Mode {ii+1} Frequency (THz)'] = ((), f)
            data_vars[f'Mode {ii+1} Frequency Variance (THz)'] = ((), var_freqs[ii])
            data_vars[f'Mode {ii+1} Damping (THz)'] = ((), damps[ii])
            data_vars[f'Mode {ii+1} Damping Variance (THz)'] = ((), var_damps[ii])
            if ii<n_damped:
                data_vars[f'Mode {ii+1} Type'] = ((), 'under damped')
            else:
                data_vars[f'Mode {ii+1} Type'] = ((), 'over damped')
    attrs['ndamped fit params'] = opt_params

    return data_vars, coord_vars, attrs

def fit_fft_ndamped_meas(measurement, x_var, y_vars, n_damped, n_over_damped, p0=None, freq_bounds=None, bounds=None, normalize=False, kwargs_dict={}, method='least_squares'):

    if type(y_vars) is not list:
        y_vars = [y_vars]
    x = measurement[x_var].data.flatten()
    ys = [measurement[y_var].data.flatten() for y_var in y_vars]
    opt_params, var, x_fit, ys_fit, ys_guess = fit_fft_ndamped(x, ys, n_damped, n_over_damped, p0, bounds=bounds, freq_bounds=freq_bounds, normalize=normalize, kwargs_dict=kwargs_dict, method=method)
    freqs, damps, d_amps, d_phis, od_amps1, od_amps2 = opt_params
    var_freqs, var_damps, var_d_amps, var_d_phis, var_od_amps1, var_od_amps2 = var
    data_vars = {}
    coord_vars= {}
    attrs = {}
    #coord_vars['Mode Params'] = ['Frequency (THz)', 'Damping (THz)', 'Amplitude', 'Phase', 'Type']
    for ii, y_var in enumerate(y_vars):
        coord_vars[x_var+f' ({y_var} fit)'] = x_fit
        data_vars[f'{y_var} (fit)'] = ((x_var+f' ({y_var} fit)'), ys_fit[ii])
        data_vars[f'{y_var} (fit guess)'] = ((x_var+f' ({y_var} fit)'), ys_guess[ii])
        for jj, f in enumerate(freqs):
            if jj<n_damped:
                data_vars[f'Mode {jj+1} FFT Amplitude ({y_var})'] = ((), d_amps[ii][jj])
                data_vars[f'Mode {jj+1} FFT Amplitude Variance ({y_var})'] = ((), var_d_amps[ii][jj])
                data_vars[f'Mode {jj+1} FFT Phase ({y_var})'] = ((), d_phis[ii][jj])
                data_vars[f'Mode {jj+1} FFT Phase Variance ({y_var})'] = ((), var_d_phis[ii,jj])
                data_vars[f'Mode {jj+1} FFT Type ({y_var})'] = ((), 'under damped')
            else:
                data_vars[f'Mode {jj+1} FFT Amplitude 1 ({y_var})'] = ((), od_amps1[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} FFT Amplitude Variance ({y_var})'] = ((), var_od_amps1[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} FFT Amplitude 2 ({y_var})'] = ((), od_amps2[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} FFT Amplitude 2 Variance ({y_var})'] = ((), var_od_amps2[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} FFT Type ({y_var})'] = ((), 'over damped')      
    for ii, f in enumerate(freqs):
            data_vars[f'Mode {ii+1} FFT Frequency (THz)'] = ((), f)
            data_vars[f'Mode {ii+1} FFT Frequency Variance (THz)'] = ((), var_freqs[ii])
            data_vars[f'Mode {ii+1} FFT Damping (THz)'] = ((), damps[ii])
            data_vars[f'Mode {ii+1} FFT Damping Variance (THz)'] = ((), var_damps[ii])
            if ii<n_damped:
                data_vars[f'Mode {ii+1} FFT Type'] = ((), 'under damped')
            else:
                data_vars[f'Mode {ii+1} FFT Type'] = ((), 'over damped')
    attrs['FFT ndamped fit params'] = opt_params

    return data_vars, coord_vars, attrs

def fit_pow_ndamped_meas(measurement, x_var, y_vars, n_damped, n_over_damped, p0=None, freq_bounds=None, bounds=None, normalize=False, kwargs_dict={}, method='least_squares'):

    if type(y_vars) is not list:
        y_vars = [y_vars]
    x = measurement[x_var].data.flatten()
    ys = [measurement[y_var].data.flatten() for y_var in y_vars]
    opt_params, var, x_fit, ys_fit, ys_guess = fit_pow_ndamped(x, ys, n_damped, n_over_damped, p0, bounds=bounds, freq_bounds=freq_bounds, normalize=normalize, kwargs_dict=kwargs_dict, method=method)
    freqs, damps, d_amps, od_amps, od_phis = opt_params
    var_freqs, var_damps, var_d_amps, var_od_amps, var_od_phis = var
    data_vars = {}
    coord_vars= {}
    attrs = {}
    #coord_vars['Mode Params'] = ['Frequency (THz)', 'Damping (THz)', 'Amplitude', 'Phase', 'Type']
    for ii, y_var in enumerate(y_vars):
        coord_vars[x_var+f' ({y_var} fit)'] = x_fit
        data_vars[f'{y_var} (fit)'] = ((x_var+f' ({y_var} fit)'), ys_fit[ii])
        data_vars[f'{y_var} (fit guess)'] = ((x_var+f' ({y_var} fit)'), ys_guess[ii])
        for jj, f in enumerate(freqs):
            if jj<n_damped:
                data_vars[f'Mode {jj+1} Pow Amplitude ({y_var})'] = ((), d_amps[ii][jj])
                data_vars[f'Mode {jj+1} Pow Amplitude Variance ({y_var})'] = ((), var_d_amps[ii][jj])
                data_vars[f'Mode {jj+1} Pow Phase ({y_var})'] = ((), 0)
                data_vars[f'Mode {jj+1} Pow Phase Variance ({y_var})'] = ((), 0)
                data_vars[f'Mode {jj+1} Pow Type ({y_var})'] = ((), 'under damped')
            else:
                data_vars[f'Mode {jj+1} Pow Amplitude 1 ({y_var})'] = ((), od_amps[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} Pow Amplitude Variance ({y_var})'] = ((), var_od_amps[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} Pow Amplitude 2 ({y_var})'] = ((), od_phis[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} Pow Amplitude 2 Variance ({y_var})'] = ((), var_od_phis[ii][jj-n_damped])
                data_vars[f'Mode {jj+1} Pow Type ({y_var})'] = ((), 'over damped')      
    for ii, f in enumerate(freqs):
            data_vars[f'Mode {ii+1} Pow Frequency (THz)'] = ((), f)
            data_vars[f'Mode {ii+1} Pow Frequency Variance (THz)'] = ((), var_freqs[ii])
            data_vars[f'Mode {ii+1} Pow Damping (THz)'] = ((), damps[ii])
            data_vars[f'Mode {ii+1} Pow Damping Variance (THz)'] = ((), var_damps[ii])
            if ii<n_damped:
                data_vars[f'Mode {ii+1} Pow Type'] = ((), 'under damped')
            else:
                data_vars[f'Mode {ii+1} Pow Type'] = ((), 'over damped')
    attrs['Pow ndamped fit params'] = opt_params

    return data_vars, coord_vars, attrs

def fit_npoly_npk_meas(measurement, x_var, y_vars, npoly, npks, p0=None,  peak_type=0, freq_bounds=None, bounds=None):
    '''
    Under construction
    '''

    if type(y_vars) is not list:
        y_vars = [y_vars]
    x = measurement[x_var].data.flatten()
    ys = [measurement[y_var].data.flatten() for y_var in y_vars]
    opt_params, var, x_fit, ys_fit, ys_guess = fit_npoly_npks(x, ys, npoly, npks, p0, peak_type, bounds=bounds, freq_bounds=freq_bounds)
    polyamps, freqs, widths, amps = opt_params
    var_polyamps, var_freqs, var_widths, var_amps = var
    data_vars = {}
    coord_vars= {}
    attrs = {}
    coord_vars['Peak Fit Params'] = ['Frequency (THz)', 'Width (THz)', 'Amplitude']
    for ii, y_var in enumerate(y_vars):
        coord_vars[x_var+f' ({y_var} fit)'] = x_fit
        data_vars[y_var+' (fit)'] = ((x_var+f' ({y_var} fit)'), ys_fit[ii])
        data_vars[y_var+' (fit guess)'] = ((x_var+f' ({y_var} fit)'), ys_guess[ii])
        for jj, f in enumerate(freqs):
            peak_params = [f, widths[jj], amps[ii][[jj]]]
            peak_var_params = [var_freqs[jj], var_widths[jj], var_amps[ii][[jj]]]
            #data_vars[f'Peak {jj+1} Params ({y_var} fit)'] = (('Peak Fit Params'), peak_params)
            #data_vars[f'Peak {jj+1} Params Variance ({y_var} fit)'] = (('Peak Fit Params'), peak_var_params)
    for ii, f in enumerate(freqs):
            data_vars[f'Peak {ii+1} Frequency (THz)'] = ((), f)
            data_vars[f'Peak {ii+1} Frequency (THz) Variance'] = ((), var_freqs[ii])
            data_vars[f'Peak {ii+1} Width (THz)'] = ((), widths[ii])
            data_vars[f'Peak {ii+1} Widths (THz) Variance'] = ((), var_widths[ii])
    attrs['peak fit params'] = opt_params

    return data_vars, coord_vars, attrs

def force_angles_continuous(measurement, x_var, angles_var='Birefringence Angle (Demod x)'):

    angles = measurement[angles_var].data
    new_angles = make_angles_continuous(angles)
    data_vars = {f'{angles_var} (continuous)':((x_var), new_angles)}
    coord_vars = {}
    attrs = {}
    return data_vars, coord_vars, attrs

######################################
### Tempearture Integration ##########
######################################

def temp_integrate(meas, var, temp_var='Temperature (K)'):
    '''
    Given a measurement, cumulatively integrate var along T.
    '''

    mod_meas = meas.copy()

    temps = mod_meas[temp_var].data
    x = mod_meas[var].data
    #offset = np.mean(mod_meas[var].sel({temp_var:slice(temps[-3],temps[-1])}).data)
    #x = x - offset
    x_int = -sp.integrate.cumulative_trapezoid(x, temps, initial=0)
    x_int = x_int - x_int[-1]

    mod_meas[f'{var} T integrated'] = ((temp_var), x_int)
        
    return mod_meas

def temp_integrate_polarization(meas, var, temp_var='Temperature (K)'):
    '''
    Same as above but handles variables that depends on Polarization. 
    '''

    mod_meas = meas.copy()
    polarizations = mod_meas['Polarization Angle (deg)'].data
    var_dims = mod_meas[var].dims
    pol_idx = list(var_dims).index('Polarization Angle (deg)')

    x_int2d = np.zeros_like(mod_meas[var])
    for ii, pol in enumerate(polarizations):

        temps = mod_meas[temp_var].data
        x = mod_meas[var].sel({'Polarization Angle (deg)':pol}).data
        #offset = np.mean(mod_meas[var].sel({'Polarization Angle (deg)':pol, temp_var:slice(temps[-3],temps[-1])}).data)
        #x = x - offset
        x_int = -sp.integrate.cumulative_trapezoid(x, temps, initial=0)
        x_int = x_int - x_int[-1]
        if pol_idx==0:
            x_int2d[ii,:] = x_int
        else:
            x_int2d[:,ii] = x_int

    mod_meas[f'{var} T integrated'] = (var_dims, x_int2d)

    return mod_meas

###############
### Helpers ###
###############

def redefine_fit_angles(params):
    '''
    helper function for fit_birefringence(), which
    '''
    if params[0]<0:
            params[0]=-params[0]
            params[1]=params[1]+180/4
    #postive amplitude 2theta
    if params[2]<0:
        params[2]=-params[2]
        params[3]=params[3]+180/2
    while params[1]>90:
        params[1]=params[1]-90
    while params[1]<0:
        params[1]=params[1]+90
    while params[3]<0:
        params[3]=params[3]+180
    while params[3]>180:
        params[3]=params[3]-180
    return params

def make_angles_continuous(angles, direction=1):
    '''
    smooths out angles such that they are more continuous by picking equivalent angles that are nearest to last point
    '''
    if direction==-1:
        angles = np.flip(angles)
    curr_ang = angles[0]
    new_angles = np.zeros_like(angles)
    for ii, ang in enumerate(angles):
        new_angs = np.array([ang, ang+180, ang-180])
        keep_idx = np.argmin(np.abs(new_angs-curr_ang))
        new_angles[ii] = new_angs[keep_idx]
        curr_ang = new_angs[keep_idx]
    if direction==-1:
        new_angles = np.flip(new_angles)
    return new_angles

##########################################
### Time domain damped oscillator fits ###
###############s##########################

def fit_ndamped(x, ys, n_damped, n_over_damped, params0=None, bounds=None, freq_bounds=None, normalize=False, kwargs_dict={}, method='least_squares'):
    '''
    given xdata and a set of y datas, simulatneously fits the ys to same set of frequencies with other paramters variable.
    params are in the form [freqs, damps, c1, amps1, phis1 ..., cn, ampsn, phisn]
    '''
    # setup 
    if type(ys) is not list:
        ys = [ys]
    n_sets = len(ys)
    n_freqs = n_damped + n_over_damped

    if params0==None:
        freqs0 = np.array([100/np.max(x) for i in range(n_freqs)])
        damps0 = np.array([2/np.max(x) for i in range(n_freqs)])
        cs0 = np.array([0]*n_sets)
        d_amps0 = np.array([[np.max(ys[j]) for i in range(n_damped)] for j in range(n_sets)])
        d_phis0 = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        od_amps1 = np.array([[np.max(ys[j]) for i in range(n_over_damped)] for j in range(n_sets)])
        od_amps2 = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        params0=[freqs0, damps0, cs0, d_amps0, d_phis0, od_amps1, od_amps2]
    if bounds==None:
        if freq_bounds==None:
            upper_freqs = np.array([np.inf for i in range(n_freqs)])
            lower_freqs = np.array([0 for i in range(n_freqs)])
        else:
            lower_freqs = freq_bounds[0]
            upper_freqs = freq_bounds[1]
        upper_damps = np.array([np.inf for i in range(n_freqs)])
        upper_cs = np.array([np.inf]*n_sets)
        upper_d_amps = np.array([[np.inf for i in range(n_damped)] for j in range(n_sets)])
        upper_d_phis = np.array([[2*np.pi for i in range(n_damped)] for j in range(n_sets)])
        upper_od_amps1 = np.array([[np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        upper_od_amps2 = np.array([[np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        upper_bounds=flatten_params([upper_freqs, upper_damps, upper_cs, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2])
        lower_damps = np.array([0 for i in range(n_freqs)])
        lower_cs = np.array([-np.inf]*n_sets)
        lower_d_amps = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        lower_d_phis = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        lower_od_amps1 = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        lower_od_amps2 = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        lower_bounds=flatten_params([lower_freqs, lower_damps, lower_cs, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2])
        bounds = (lower_bounds, upper_bounds)

    # package bounds
    lower_freqs, lower_damps, lower_cs, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2 = bounds[0]
    upper_freqs, upper_damps, upper_cs, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2 = bounds[1]
    upper_bounds=flatten_params([upper_freqs, upper_damps, upper_cs, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2])
    lower_bounds=flatten_params([lower_freqs, lower_damps, lower_cs, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2])
    bounds = (lower_bounds, upper_bounds)
    bounds_minimize = [(lower_bounds[ii], upper_bounds[ii]) for ii in range(len(lower_bounds))]

    # normalize ys to fit on equal footing:
    if normalize==True:
        norm_ys = np.array([np.max(np.abs(y)) for y in ys])
        ys = [y/np.max(np.abs(y)) for y in ys]
        norm_ys_d = np.array([norm_ys for i in range(n_damped)]).transpose()
        norm_ys_od = np.array([norm_ys for i in range(n_over_damped)]).transpose()

        params0_nonorm = params0.copy()
        freqs0, damps0, cs0, d_amps0, d_phis0, od_amps01, od_amps02 = params0
        cs0 = cs0/norm_ys_d
        d_amps0 = d_amps0/norm_ys_d
        od_amps01 = od_amps01/norm_ys_od
        od_amps02 = od_amps02/norm_ys_od
        params0=[freqs0, damps0, cs0, d_amps0, d_phis0, od_amps01, od_amps02]

        lower_bounds, upper_bounds = bounds
        lower_freqs, lower_damps, lower_cs, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2 = unflatten_params(lower_bounds, n_damped, n_over_damped, n_sets)
        upper_freqs, upper_damps, upper_cs, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2 = unflatten_params(upper_bounds, n_damped, n_over_damped, n_sets)
        lower_cs = lower_cs/norm_ys_d
        lower_d_amps = lower_d_amps/norm_ys_d
        lower_od_amps1 = lower_od_amps1/norm_ys_od
        lower_od_amps2 = lower_od_amps2/norm_ys_od
        upper_cs = upper_cs/norm_ys_d
        upper_d_amps = upper_d_amps/norm_ys_d
        upper_od_amps1 = upper_od_amps1/norm_ys_od
        upper_od_amps2 = upper_od_amps2/norm_ys_od
        lower_bounds = flatten_params([lower_freqs, lower_damps, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2])
        upper_bounds=flatten_params([upper_freqs, upper_damps, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2])
        bounds = (lower_bounds, upper_bounds)

    
    # flatten params and execute fit
    params0_flattened = flatten_params(params0)
    params0_packed = pack_params(params0_flattened, n_freqs, n_sets)
    n_params = len(params0_packed)
    fosc = ff.ndamped_osc(n_damped, n_over_damped)
    if method=='least_squares':
        res_lsq = opt.least_squares(residual_ndamped, params0_flattened, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), **kwargs_dict)
        opt_params_flattened = res_lsq.x
        try:
            J = res_lsq.jac
            cov = np.linalg.inv(J.T.dot(J))*(res_lsq.fun.T @ res_lsq.fun)/ (res_lsq.fun.size - res_lsq.x.size)
            var_flattened = np.sqrt(np.diagonal(cov))
        except:
            print('singular matrix found')
            var_flattened = opt_params_flattened
    elif method=='dual_annealing':
        res = opt.dual_annealing(cost_ndamped, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), x0=params0_flattened, **kwargs_dict)
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
    elif method=='differential_evolution':
        res = opt.differential_evolution(cost_ndamped, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), x0=params0_flattened, **kwargs_dict)
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
    elif method=='basinhopping':
        res = opt.basinhopping(cost_ndamped, params0_flattened, minimizer_kwargs=dict(bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params)))
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
    elif method=='shgo':
        res = opt.shgo(cost_ndamped, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), **kwargs_dict)
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened

    # renormalize ys
    if normalize==True:
        opt_params = unflatten_params(opt_params_flattened, n_damped, n_over_damped, n_sets)
        var = unflatten_params(var_flattened, n_damped, n_over_damped, n_sets)
        freqs, damps, cs, d_amps, d_phis, od_amps1, od_amps2 = opt_params
        var_freqs, var_damps, var_cs, var_d_amps, var_d_phis, var_od_amps1, var_od_amps2 = var
        cs = np.array([cs[i]*norm_ys[i] for i in range(n_sets)])
        d_amps = np.array([d_amps[i]*norm_ys[i] for i in range(n_sets)])
        od_amps1 = np.array([od_amps1[i]*norm_ys[i] for i in range(n_sets)])
        od_amps2 = np.array([od_amps2[i]*norm_ys[i] for i in range(n_sets)])
        var_cs = np.array([var_cs[i]*norm_ys[i] for i in range(n_sets)])
        var_d_amps = np.array([var_d_amps[i]*norm_ys[i] for i in range(n_sets)])
        var_od_amps1 = np.array([var_od_amps1[i]*norm_ys[i] for i in range(n_sets)])
        var_od_amps2 = np.array([var_od_amps2[i]*norm_ys[i] for i in range(n_sets)])
        opt_params = [freqs, damps, cs, d_amps, d_phis, od_amps1, od_amps2]
        var = [var_freqs, var_damps, var_cs, var_d_amps, var_d_phis, var_od_amps1, var_od_amps2]
        opt_params_flattened = flatten_params(opt_params)
        var_flattened = flatten_params(var)

    # evaluate fit and guess over x
    opt_params_packed = pack_params(opt_params_flattened, n_freqs, n_sets)
    x_fit = np.linspace(x[0], x[-1], 1000)
    ys_fit = [fosc(x_fit, opt_params_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    ys_guess = [fosc(x_fit, params0_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]

    # unflatten params for return
    opt_params = unflatten_params(opt_params_flattened, n_damped, n_over_damped, n_sets)
    var = unflatten_params(var_flattened, n_damped, n_over_damped, n_sets)

    return opt_params, var, x_fit, ys_fit, ys_guess

def flatten_params(params):
    '''
    take parameters made for fit and pack them for residual function. params are in the form

    params = [[freqs], [damps], [cs], [[d_amps]], [[d_phis], [[od_amps]], [[od_phis]]
    
    and outputs (given n frequencies and m sets)

    params = [f1,..,fn,damps1,...,dampsn,c11,d_amp11,d_phi11,...,od_amp11, od_phi11,c12,...,d_phimn]

    as a numpy array
    '''

    freqs, damps, cs, d_amps, d_phis, od_amps, od_phis = params
    n_sets = len(cs)
    n_damped = len(d_amps[0])
    n_over_damped = len(od_amps[0])
    flattened_params = [f for f in freqs]+[d for d in damps]
    for i in range(n_sets):
        set_params = [cs[i]]
        for j in range(n_damped):
            set_params += [d_amps[i][j], d_phis[i][j]]
        for j in range(n_over_damped):
            set_params += [od_amps[i][j], od_phis[i][j]]
        flattened_params+=set_params
    return np.array(flattened_params)

def pack_params(params, n_freqs, n_sets):
    '''
    take flattened parameters made for fit and pack them for residual function. params are in the form

    params = [f1,..,fn,damps1,...,dampsn,c11,d_amp11,d_phi11,...,od_amp11, od_phi11,c12,...,d_phimn]
    
    and outputs

    params = [c1,f1,damps1,amp11,phi11,...,fn,dampsn,amps1n, phis1n, c2, f1, damps1, amp21, phis21,....,fn, dampsn, ampsdn, phisdn]

    '''
    nsetparams = 4*n_freqs+1
    packed_params = np.zeros(n_sets*nsetparams)
    for i in range(n_sets):
        packed_params[i*nsetparams] = params[2*n_freqs+i*(1+2*n_freqs)] # c
        for j in range(n_freqs):
            packed_params[i*nsetparams+4*j+1] = params[j]
            packed_params[i*nsetparams+4*j+2] = params[n_freqs+j]
            packed_params[i*nsetparams+4*j+3] = params[2*n_freqs+i*(1+2*n_freqs)+2*j+1]
            packed_params[i*nsetparams+4*j+4] = params[2*n_freqs+i*(1+2*n_freqs)+2*j+2]
    return np.array(packed_params)

def unflatten_params(params, n_damped, n_over_damped, n_sets):
    '''
    unpacks parameters back into original form, ie from 
    
    params = [f1,..,fn,damps1,...,dampsn,c11,d_amp11,d_phi11,...,od_amp11, od_phi11,c12,...,d_phimn]

    to 

    params = [[freqs], [damps], [cs], [[d_amps]], [[d_phis], [[od_amps]], [[od_phis]]
    '''

    n_freqs = n_damped + n_over_damped
    freqs = params[:n_freqs]
    damps = params[n_freqs:2*n_freqs]
    cs = [params[2*n_freqs+i*(1+2*n_freqs)] for i in range(n_sets)]
    d_amps = [[params[2*n_freqs+i*(1+2*n_freqs)+1+2*j] for j in range(n_damped)] for i in range(n_sets)]
    d_phis = [[params[2*n_freqs+i*(1+2*n_freqs)+1+2*j+1] for j in range(n_damped)] for i in range(n_sets)]
    od_amps = [[params[2*n_freqs+i*(1+2*n_freqs)+1+2*n_damped+2*j] for j in range(n_over_damped)] for i in range(n_sets)]
    od_phis = [[params[2*n_freqs+i*(1+2*n_freqs)+1+2*n_damped+2*j+1] for j in range(n_over_damped)] for i in range(n_sets)]
    unflattened_params = [np.array(freqs), np.array(damps), np.array(cs), np.array(d_amps), np.array(d_phis), np.array(od_amps), np.array(od_phis)]
    return unflattened_params

def residual_ndamped(params, x, ys, fosc, n_freqs, n_sets, n_params):
    '''
    computes residual for ndamped oscillators and 
    '''
    params = pack_params(params, n_freqs, n_sets)
    ys_computed = [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    y = np.concatenate(ys)
    y_computed = np.concatenate(ys_computed)
    return np.abs(y - y_computed)

def cost_ndamped(params, x, ys, fosc, n_freqs, n_sets, n_params):
    '''
    computes residual for ndamped oscillators and 
    '''
    params = pack_params(params, n_freqs, n_sets)
    ys_computed = [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    y = np.concatenate(ys)
    y_computed = np.concatenate(ys_computed)
    residual = y - y_computed
    return np.sum(residual**2)

def compute_ndamped_nsets(fosc, x, params, n_damped, n_over_damped, n_sets):
    n_freqs = n_damped+n_over_damped
    params = flatten_params(params)
    params = pack_params(params, n_freqs, n_sets)
    n_params = len(params)
    return [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]

##################################
### FFT damped oscillator fits ###
###############s##################

def fit_fft_ndamped(x, ys, n_damped, n_over_damped, params0=None, bounds=None, freq_bounds=None, normalize=False, kwargs_dict={}, method='least_squares'):
    '''
    given xdata and a set of y datas, simulatneously fits the ys to same set of frequencies with other paramters variable.
    params are in the form [freqs, c1, other_params1, ..., cn, other_params2]
    '''
    # setup 
    if type(ys) is not list:
        ys = [ys]
    n_sets = len(ys)
    n_freqs = n_damped + n_over_damped

    if params0==None:
        freqs0 = np.array([100/np.max(x) for i in range(n_freqs)])
        damps0 = np.array([2/np.max(x) for i in range(n_freqs)])
        d_amps0 = np.array([[np.real(np.max(ys[j])) for i in range(n_damped)] for j in range(n_sets)])
        d_phis0 = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        od_amps01 = np.array([[np.real(np.max(ys[j])) for i in range(n_over_damped)] for j in range(n_sets)])
        od_amps02 = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        params0=[freqs0, damps0, d_amps0, d_phis0, od_amps01, od_amps02]
    if bounds==None:
        if freq_bounds==None:
            upper_freqs = np.array([np.inf for i in range(n_freqs)])
            lower_freqs = np.array([0 for i in range(n_freqs)])
        else:
            lower_freqs = freq_bounds[0]
            upper_freqs = freq_bounds[1]
        upper_damps = np.array([np.inf for i in range(n_freqs)])
        upper_d_amps = np.array([[np.inf for i in range(n_damped)] for j in range(n_sets)])
        upper_d_phis = np.array([[2*np.pi for i in range(n_damped)] for j in range(n_sets)])
        upper_od_amps1 = np.array([[np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        upper_od_amps2 = np.array([[np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        upper_bounds=flatten_params_fft([upper_freqs, upper_damps, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2])
        lower_damps = np.array([0 for i in range(n_freqs)])
        lower_d_amps = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        lower_d_phis = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        lower_od_amps1 = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        lower_od_amps2 = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        lower_bounds=flatten_params_fft([lower_freqs, lower_damps, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2])
        bounds = (lower_bounds, upper_bounds)

    # package bounds
    lower_freqs, lower_damps, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2 = bounds[0]
    upper_freqs, upper_damps, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2 = bounds[1]
    upper_bounds=flatten_params_fft([upper_freqs, upper_damps, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2])
    lower_bounds=flatten_params_fft([lower_freqs, lower_damps, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2])
    bounds = (lower_bounds, upper_bounds)
    bounds_minimize = [(lower_bounds[ii], upper_bounds[ii]) for ii in range(len(lower_bounds))]

    # normalize ys to fit on equal footing:
    if normalize==True:
        norm_ys = np.array([np.max(np.abs(y)) for y in ys])
        ys = [y/np.max(np.abs(y)) for y in ys]
        norm_ys_d = np.array([norm_ys for i in range(n_damped)]).transpose()
        norm_ys_od = np.array([norm_ys for i in range(n_over_damped)]).transpose()

        params0_nonorm = params0.copy()
        freqs0, damps0, d_amps0, d_phis0, od_amps01, od_amps02 = params0
        d_amps0 = d_amps0/norm_ys_d
        od_amps01 = od_amps01/norm_ys_od
        od_amps02 = od_amps02/norm_ys_od
        params0=[freqs0, damps0, d_amps0, d_phis0, od_amps01, od_amps02]

        lower_bounds, upper_bounds = bounds
        lower_freqs, lower_damps, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2 = unflatten_params_fft(lower_bounds, n_damped, n_over_damped, n_sets)
        upper_freqs, upper_damps, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2 = unflatten_params_fft(upper_bounds, n_damped, n_over_damped, n_sets)
        lower_d_amps = lower_d_amps/norm_ys_d
        lower_od_amps1 = lower_od_amps1/norm_ys_od
        lower_od_amps2 = lower_od_amps2/norm_ys_od
        upper_d_amps = upper_d_amps/norm_ys_d
        upper_od_amps1 = upper_od_amps1/norm_ys_od
        upper_od_amps2 = upper_od_amps2/norm_ys_od
        lower_bounds = flatten_params_fft([lower_freqs, lower_damps, lower_d_amps, lower_d_phis, lower_od_amps1, lower_od_amps2])
        upper_bounds=flatten_params_fft([upper_freqs, upper_damps, upper_d_amps, upper_d_phis, upper_od_amps1, upper_od_amps2])
        bounds = (lower_bounds, upper_bounds)

    # flatten params and execute fit
    params0_flattened = flatten_params_fft(params0)
    params0_packed = pack_params_fft(params0_flattened, n_freqs, n_sets)
    n_params = len(params0_packed)
    fosc = ff.FFT_ndamped_osc(n_damped, n_over_damped)
    if method=='least_squares':
        res_lsq = opt.least_squares(residual_fft_ndamped, params0_flattened, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), **kwargs_dict)
        opt_params_flattened = res_lsq.x
        try:
            J = res_lsq.jac
            cov = np.linalg.inv(J.T.dot(J))*(res_lsq.fun.T @ res_lsq.fun)/ (res_lsq.fun.size - res_lsq.x.size)
            var_flattened = np.sqrt(np.diagonal(cov))
        except:
            print('singular matrix found')
            var_flattened = opt_params_flattened
    elif method=='dual_annealing':
        res = opt.dual_annealing(cost_fft_ndamped, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), x0=params0_flattened, **kwargs_dict)
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
    elif method=='differential_evolution':
        res = opt.differential_evolution(cost_fft_ndamped, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), x0=params0_flattened, **kwargs_dict)
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
    elif method=='basinhopping':
        res = opt.basinhopping(cost_fft_ndamped, params0_flattened, minimizer_kwargs=dict(bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params)))
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
    elif method=='shgo':
        res = opt.shgo(cost_fft_ndamped, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), **kwargs_dict)
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened

    # renormalize ys
    if normalize==True:
        opt_params = unflatten_params_fft(opt_params_flattened, n_damped, n_over_damped, n_sets)
        var = unflatten_params_fft(var_flattened, n_damped, n_over_damped, n_sets)
        freqs, damps, d_amps, d_phis, od_amps1, od_amps2 = opt_params
        var_freqs, var_damps, var_d_amps, var_d_phis, var_od_amps1, var_od_amps2 = var
        d_amps = np.array([d_amps[i]*norm_ys[i] for i in range(n_sets)])
        od_amps1 = np.array([od_amps1[i]*norm_ys[i] for i in range(n_sets)])
        od_amps2 = np.array([od_amps2[i]*norm_ys[i] for i in range(n_sets)])
        var_d_amps = np.array([var_d_amps[i]*norm_ys[i] for i in range(n_sets)])
        var_od_amps1 = np.array([var_od_amps1[i]*norm_ys[i] for i in range(n_sets)])
        var_od_amps2 = np.array([var_od_amps2[i]*norm_ys[i] for i in range(n_sets)])
        opt_params = [freqs, damps, d_amps, d_phis, od_amps1, od_amps2]
        var = [var_freqs, var_damps, var_d_amps, var_d_phis, var_od_amps1, var_od_amps2]
        opt_params_flattened = flatten_params_fft(opt_params)
        var_flattened = flatten_params_fft(var)

    # evaluate fit and guess over x
    opt_params_packed = pack_params_fft(opt_params_flattened, n_freqs, n_sets)
    n_params = len(opt_params_packed)
    x_fit = np.linspace(x[0], x[-1], 1000)
    ys_fit = [fosc(x_fit, opt_params_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    ys_guess = [fosc(x_fit, params0_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]

    # unflatten params for return
    opt_params = unflatten_params_fft(opt_params_flattened, n_damped, n_over_damped, n_sets)
    var = unflatten_params_fft(var_flattened, n_damped, n_over_damped, n_sets)

    return opt_params, var, x_fit, ys_fit, ys_guess

def flatten_params_fft(params):
    '''
    take parameters made for fit and pack them for residual function. params are in the form

    params = [[freqs], [damps], [cs], [[d_amps]], [[d_phis], [[od_amps]], [[od_phis]]
    
    and outputs (given n frequencies and m sets)

    params = [f1,..,fn,damps1,...,dampsn,c11,d_amp11,d_phi11,...,od_amp11, od_phi11,c12,...,d_phimn]

    as a numpy array
    '''

    freqs, damps, d_amps, d_phis, od_amps1, od_amps2 = params
    n_sets = len(d_amps)
    n_damped = len(d_amps[0])
    n_over_damped = len(od_amps1[0])
    flattened_params = [f for f in freqs]+[d for d in damps]
    for i in range(n_sets):
        set_params = []
        for j in range(n_damped):
            set_params += [d_amps[i][j], d_phis[i][j]]
        for j in range(n_over_damped):
            set_params += [od_amps1[i][j], od_amps2[i][j]]
        flattened_params+=set_params
    return np.array(flattened_params)

def pack_params_fft(params, n_freqs, n_sets):
    '''
    take flattened parameters made for fit and pack them for residual function. params are in the form

    params = [f1,..,fn,damps1,...,dampsn,c11,d_amp11,d_phi11,...,od_amp11, od_phi11,c12,...,d_phimn]
    
    and outputs

    params = [c1,f1,damps1,amp11,phi11,...,fn,dampsn,amps1n, phis1n, c2, f1, damps1, amp21, phis21,....,fn, dampsn, ampsdn, phisdn]

    '''
    nsetparams = 4*n_freqs
    packed_params = np.zeros(n_sets*nsetparams)
    for i in range(n_sets):
        for j in range(n_freqs):
            packed_params[i*nsetparams+4*j] = params[j]
            packed_params[i*nsetparams+4*j+1] = params[n_freqs+j]
            packed_params[i*nsetparams+4*j+2] = params[2*n_freqs+i*(2*n_freqs)+2*j]
            packed_params[i*nsetparams+4*j+3] = params[2*n_freqs+i*(2*n_freqs)+2*j+1]
    return np.array(packed_params)

def unflatten_params_fft(params, n_damped, n_over_damped, n_sets):
    '''
    unpacks parameters back into original form, ie from 
    
    params = [f1,..,fn,damps1,...,dampsn,c11,d_amp11,d_phi11,...,od_amp11, od_phi11,c12,...,d_phimn]

    to 

    params = [[freqs], [damps], [cs], [[d_amps]], [[d_phis], [[od_amps]], [[od_phis]]
    '''

    n_freqs = n_damped + n_over_damped
    freqs = params[:n_freqs]
    damps = params[n_freqs:2*n_freqs]
    d_amps = [[params[2*n_freqs+i*(2*n_freqs)+2*j] for j in range(n_damped)] for i in range(n_sets)]
    d_phis = [[params[2*n_freqs+i*(2*n_freqs)+2*j+1] for j in range(n_damped)] for i in range(n_sets)]
    od_amps1 = [[params[2*n_freqs+i*(2*n_freqs)+2*n_damped+2*j] for j in range(n_over_damped)] for i in range(n_sets)]
    od_amps2 = [[params[2*n_freqs+i*(2*n_freqs)+2*n_damped+2*j+1] for j in range(n_over_damped)] for i in range(n_sets)]
    unflattened_params = [np.array(freqs), np.array(damps), np.array(d_amps), np.array(d_phis), np.array(od_amps1), np.array(od_amps2)]
    return unflattened_params

def residual_fft_ndamped(params, x, ys, fosc, n_freqs, n_sets, n_params):
    '''
    computes residual for ndamped oscillators and 
    '''
    params = pack_params_fft(params, n_freqs, n_sets)
    ys_computed = [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    y = np.concatenate(ys)
    y_real, y_imag = np.real(y), np.imag(y)
    y_computed = np.concatenate(ys_computed)
    y_computed_real, y_computed_imag = np.real(y_computed), np.imag(y_computed)
    y_total = np.concatenate([y_real, y_imag])
    y_computed_total = np.concatenate([y_computed_real, y_computed_imag])
    return np.abs(y_total - y_computed_total)

def cost_fft_ndamped(params, x, ys, fosc, n_freqs, n_sets, n_params):
    '''
    computes residual for ndamped oscillators and 
    '''
    params = pack_params_fft(params, n_freqs, n_sets)
    ys_computed = [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    y = np.concatenate(ys)
    y_real, y_imag = np.real(y), np.imag(y)
    y_computed = np.concatenate(ys_computed)
    y_computed_real, y_computed_imag = np.real(y_computed), np.imag(y_computed)
    y_total = np.concatenate([y_real, y_imag])
    y_computed_total = np.concatenate([y_computed_real, y_computed_imag])
    residual = (y_total - y_computed_total)
    return np.sum(residual**2)

def compute_fft_ndamped_nsets(fosc, x, params, n_damped, n_over_damped, n_sets):
    n_freqs = n_damped+n_over_damped
    params = flatten_params_fft(params)
    params = pack_params_fft(params, n_freqs, n_sets)
    n_params = len(params)
    return [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]

############################################
### Power spectra damped oscillator fits ###
############################################

def fit_pow_ndamped(x, ys, n_damped, n_over_damped, params0=None, bounds=None, freq_bounds=None, normalize=False, kwargs_dict={}, method='least_squares'):
    '''
    fits for power spectrum (ie, ignores phase information)
    given xdata and a set of y datas, simulatneously fits the ys to same set of frequencies with other paramters variable.
    params are in the form [freqs, c1, other_params1, ..., cn, other_params2]
    '''
    # setup 
    if type(ys) is not list:
        ys = [ys]
    n_sets = len(ys)
    n_freqs = n_damped + n_over_damped

    if params0==None:
        freqs0 = np.array([100/np.max(x) for i in range(n_freqs)])
        damps0 = np.array([2/np.max(x) for i in range(n_freqs)])
        d_amps0 = np.array([[np.real(np.max(ys[j])) for i in range(n_damped)] for j in range(n_sets)])
        od_amps0 = np.array([[np.real(np.max(ys[j])) for i in range(n_over_damped)] for j in range(n_sets)])
        od_amps20 = np.array([[np.real(np.max(ys[j])) for i in range(n_over_damped)] for j in range(n_sets)])
        params0=[freqs0, damps0, d_amps0, od_amps0, od_amps20]
    if bounds==None:
        if freq_bounds==None:
            upper_freqs = np.array([np.inf for i in range(n_freqs)])
            lower_freqs = np.array([0 for i in range(n_freqs)])
        else:
            lower_freqs = freq_bounds[0]
            upper_freqs = freq_bounds[1]
        upper_damps = np.array([np.inf for i in range(n_freqs)])
        upper_d_amps = np.array([[np.inf for i in range(n_damped)] for j in range(n_sets)])
        upper_od_amps = np.array([[np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        upper_od_amps2 = np.array([[np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        upper_bounds=[upper_freqs, upper_damps, upper_d_amps, upper_od_amps, upper_od_amps2]
        lower_damps = np.array([0 for i in range(n_freqs)])
        lower_d_amps = np.array([[-np.inf for i in range(n_damped)] for j in range(n_sets)])
        lower_od_amps = np.array([[-np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        lower_od_amps2 = np.array([[-np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        lower_bounds=[lower_freqs, lower_damps, lower_d_amps, lower_od_amps, lower_od_amps2]
        bounds = (lower_bounds, upper_bounds)

    # package bounds
    lower_freqs, lower_damps, lower_d_amps, lower_od_amps, lower_od_amps2 = bounds[0]
    upper_freqs, upper_damps, upper_d_amps, upper_od_amps, upper_od_amps2 = bounds[1]
    upper_bounds=flatten_params_pow([upper_freqs, upper_damps, upper_d_amps, upper_od_amps, upper_od_amps2])
    lower_bounds=flatten_params_pow([lower_freqs, lower_damps, lower_d_amps, lower_od_amps, lower_od_amps2])
    bounds = (lower_bounds, upper_bounds)
    bounds_minimize = [(lower_bounds[ii], upper_bounds[ii]) for ii in range(len(lower_bounds))]

    # normalize ys to fit on equal footing:
    if normalize==True:
        norm_ys = np.array([np.max(np.abs(y)) for y in ys])
        ys = [y/np.max(np.abs(y)) for y in ys]
        norm_ys_d = np.array([norm_ys for i in range(n_damped)]).transpose()
        norm_ys_od = np.array([norm_ys for i in range(n_over_damped)]).transpose()

        params0_nonorm = params0.copy()
        freqs0, damps0, d_amps0, od_amps0, od_amps20 = params0
        d_amps0 = d_amps0/np.sqrt(norm_ys_d)
        od_amps0 = od_amps0/norm_ys_od
        od_amps20 = od_amps20/norm_ys_od
        params0=[freqs0, damps0, d_amps0, od_amps0, od_amps20]

        lower_bounds, upper_bounds = bounds
        lower_freqs, lower_damps, lower_d_amps, lower_od_amps, lower_od_amps2 = unflatten_params_pow(lower_bounds, n_damped, n_over_damped, n_sets)
        upper_freqs, upper_damps, upper_d_amps, upper_od_amps, upper_od_amps2 = unflatten_params_pow(upper_bounds, n_damped, n_over_damped, n_sets)
        lower_d_amps = lower_d_amps/np.sqrt(norm_ys_d)
        lower_od_amps = lower_od_amps/norm_ys_od
        lower_od_amps2 = lower_od_amps2/norm_ys_od
        upper_d_amps = upper_d_amps/np.sqrt(norm_ys_d)
        upper_od_amps = upper_od_amps/norm_ys_od
        upper_od_amps2 = upper_od_amps2/norm_ys_od
        lower_bounds = flatten_params_pow([lower_freqs, lower_damps, lower_d_amps, lower_od_amps, lower_od_amps2])
        upper_bounds=flatten_params_pow([upper_freqs, upper_damps, upper_d_amps, upper_od_amps, upper_od_amps2])
        bounds = (lower_bounds, upper_bounds)
            
    # flatten params and execute fit
    params0_flattened = flatten_params_pow(params0)
    params0_packed = pack_params_pow(params0_flattened, n_damped, n_over_damped, n_sets)
    n_params = len(params0_packed)
    fosc = ff.pow_ndamped_osc(n_damped, n_over_damped)
    if method=='least_squares':
        res_lsq = opt.least_squares(residualpowt_ndamped, params0_flattened, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), **kwargs_dict)
        opt_params_flattened = res_lsq.x
        try:
            J = res_lsq.jac
            cov = np.linalg.inv(J.T.dot(J))*(res_lsq.fun.T @ res_lsq.fun)/ (res_lsq.fun.size - res_lsq.x.size)
            var_flattened = np.sqrt(np.diagonal(cov))
        except:
            print('singular matrix found')
            var_flattened = opt_params_flattened
    elif method=='dual_annealing':
        res = opt.dual_annealing(cost_pow_ndamped, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), x0=params0_flattened, **kwargs_dict)
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
    elif method=='differential_evolution':
        res = opt.differential_evolution(cost_pow_ndamped, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), x0=params0_flattened, **kwargs_dict)
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
    elif method=='basinhopping':
        res = opt.basinhopping(cost_pow_ndamped, params0_flattened, minimizer_kwargs=dict(bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params)))
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
    elif method=='shgo':
        res = opt.shgo(cost_pow_ndamped, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params), **kwargs_dict)
        opt_params_flattened = res.x
        var_flattened = opt_params_flattened
        print('singular matrix found')
        var_flattened = opt_params_flattened

    # renormalize ys
    if normalize==True:
        params0 = params0_nonorm
        params0_flattened = flatten_params_pow(params0)
        params0_packed = pack_params_pow(params0_flattened, n_damped, n_over_damped, n_sets)
        opt_params = unflatten_params_pow(opt_params_flattened, n_damped, n_over_damped, n_sets)
        var = unflatten_params_pow(var_flattened, n_damped, n_over_damped, n_sets)
        freqs, damps, d_amps, od_amps, od_phis = opt_params
        var_freqs, var_damps, var_d_amps, var_od_amps, var_od_phis = var
        d_amps = np.array([d_amps[i]*np.sqrt(norm_ys[i]) for i in range(n_sets)])
        od_amps = np.array([od_amps[i]*norm_ys[i] for i in range(n_sets)])
        var_d_amps = np.array([var_d_amps[i]*np.sqrt(norm_ys[i]) for i in range(n_sets)])
        var_od_amps = np.array([var_od_amps[i]*norm_ys[i] for i in range(n_sets)])
        opt_params = [freqs, damps, d_amps, od_amps, od_phis]
        var = [var_freqs, var_damps, var_d_amps, var_od_amps, var_od_phis]
        opt_params_flattened = flatten_params_pow(opt_params)
        var_flattened = flatten_params_pow(var)

    # evaluate fit and guess over x
    opt_params_packed = pack_params_pow(opt_params_flattened, n_damped, n_over_damped, n_sets)
    n_params = len(opt_params_packed)
    x_fit = np.linspace(x[0], x[-1], 1000)
    ys_fit = [fosc(x_fit, opt_params_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    ys_guess = [fosc(x_fit, params0_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]

    # unflatten params for return
    opt_params = unflatten_params_pow(opt_params_flattened, n_damped, n_over_damped, n_sets)
    var = unflatten_params_pow(var_flattened, n_damped, n_over_damped, n_sets)

    return opt_params, var, x_fit, ys_fit, ys_guess

def flatten_params_pow(params):
    '''
    take parameters made for fit and pack them for residual function. params are in the form

    params = [[freqs], [damps], [cs], [[d_amps]], [[d_phis], [[od_amps]], [[od_phis]]
    
    and outputs (given n frequencies and m sets)

    params = [f1,..,fn,damps1,...,dampsn,c11,d_amp11,d_phi11,...,od_amp11, od_phi11,c12,...,d_phimn]

    as a numpy array
    '''

    freqs, damps, d_amps, od_amps, od_phis = params
    n_sets = len(d_amps)
    n_damped = len(d_amps[0])
    n_over_damped = len(od_amps[0])
    flattened_params = [f for f in freqs]+[d for d in damps]
    for i in range(n_sets):
        set_params = []
        for j in range(n_damped):
            set_params += [d_amps[i][j]]
        for j in range(n_over_damped):
            set_params += [od_amps[i][j], od_phis[i][j]]
        flattened_params+=set_params
    return np.array(flattened_params)

def pack_params_pow(params, n_d, n_od, n_sets):
    '''
    take flattened parameters made for fit and pack them for residual function. params are in the form

    params = [f1,..,fn,damps1,...,dampsn,c11,d_amp11,d_phi11,...,od_amp11, od_phi11,c12,...,d_phimn]
    
    and outputs

    params = [c1,f1,damps1,amp11,phi11,...,fn,dampsn,amps1n, phis1n, c2, f1, damps1, amp21, phis21,....,fn, dampsn, ampsdn, phisdn]

    '''
    n_freqs = n_d + n_od
    nsetparams = 3*n_d + 4*n_od + 1
    nparams = nsetparams*n_sets
    packed_params = np.zeros(nparams)
    for i in range(n_sets):
        packed_params[i*nsetparams] = 0 # c
        for j in range(n_d):
            packed_params[i*nsetparams+3*j+1] = params[j]
            packed_params[i*nsetparams+3*j+2] = params[n_freqs+j]
            packed_params[i*nsetparams+3*j+3] = params[2*n_freqs+i*(n_d+2*n_od)+j]
        for j in range(n_od):
            packed_params[i*(nsetparams+3*n_d)+4*j+1] = params[j]
            packed_params[i*(nsetparams+3*n_d)+4*j+2] = params[n_freqs+j]
            packed_params[i*(nsetparams+3*n_d)+4*j+3] = params[2*n_freqs+i*(n_d+2*n_od)+2*j]
            packed_params[i*(nsetparams+3*n_d)+4*j+4] = params[2*n_freqs+i*(n_d+2*n_od)+2*j+1]
    return np.array(packed_params)

def unflatten_params_pow(params, n_damped, n_over_damped, n_sets):
    '''
    unpacks parameters back into original form, ie from 
    
    params = [f1,..,fn,damps1,...,dampsn,c11,d_amp11,d_phi11,...,od_amp11, od_phi11,c12,...,d_phimn]

    to 

    params = [[freqs], [damps], [cs], [[d_amps]], [[d_phis], [[od_amps]], [[od_phis]]
    '''

    n_freqs = n_damped + n_over_damped
    freqs = params[:n_freqs]
    damps = params[n_freqs:2*n_freqs]
    d_amps = [[params[2*n_freqs+i*(n_damped + 2*n_over_damped)+j] for j in range(n_damped)] for i in range(n_sets)]
    od_amps = [[params[2*n_freqs+i*(n_damped + 2*n_over_damped)+n_damped+2*j] for j in range(n_over_damped)] for i in range(n_sets)]
    od_phis = [[params[2*n_freqs+i*(n_damped + 2*n_over_damped)+n_damped+2*j+1] for j in range(n_over_damped)] for i in range(n_sets)]
    unflattened_params = [np.array(freqs), np.array(damps), np.array(d_amps), np.array(od_amps), np.array(od_phis)]
    return unflattened_params

def residual_pow_ndamped(params, x, ys, fosc, n_damped, n_over_damped, n_sets, n_params):
    '''
    computes residual for ndamped oscillators and 
    '''
    params = pack_params_pow(params, n_damped, n_over_damped, n_sets)
    ys_computed = [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    y = np.concatenate(ys)
    y_computed = np.concatenate(ys_computed)
    return np.abs(y - y_computed)

def cost_pow_ndamped(params, x, ys, fosc, n_damped, n_over_damped, n_sets, n_params):
    '''
    computes residual for ndamped oscillators and 
    '''
    params = pack_params_pow(params, n_damped, n_over_damped, n_sets)
    ys_computed = [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    y = np.concatenate(ys)
    y_computed = np.concatenate(ys_computed)
    residual = y - y_computed
    return np.sum(residual**2)

def compute_pow_ndamped_nsets(fosc, x, params, n_damped, n_over_damped, n_sets):
    params = flatten_params_pow(params)
    params = pack_params_pow(params, n_damped, n_over_damped, n_sets)
    n_params = len(params)
    return [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]

########################################
### Gaussian/Lorentzian peak fitting ### # hasn't been updated in a while, currently deprecated
########################################

def fit_npoly_npks(x, ys, npoly, npks, params0=None, peak_type=0, bounds=None, freq_bounds=None):
    '''
    given xdata and a set of y datas, simulatneously fits the ys to same set of frequencies with other paramters variable. polynomials can also be added to handle backgrounds.
    params are in the form [polyamps, freqs, widths, a1, a2, ..., an]
    '''
    # setup 
    if type(ys) is not list:
        ys = [ys]
    n_sets = len(ys)
    if params0==None:
        coeffs0 = np.array([[0 for i in range(npoly)] for j in range(n_sets)])
        freqs0 = np.array([100/np.max(x) for i in range(npks)])
        widths0 = np.array([2/np.max(x) for i in range(npks)])
        d_amps0 = np.array([[np.max(ys[j]) for i in range(npks)] for j in range(n_sets)])
        params0=[coeffs0, freqs0, widths0, d_amps0]
    if bounds==None:
        if freq_bounds==None:
            upper_freqs = np.array([np.inf for i in range(npks)])
            lower_freqs = np.array([0 for i in range(npks)])
        else:
            lower_freqs = freq_bounds[0]
            upper_freqs = freq_bounds[1]
        upper_coeffs = np.array([[np.inf for i in range(npoly)] for j in range(n_sets)])
        upper_widths = np.array([np.inf for i in range(npks)])
        upper_amps = np.array([[np.inf for i in range(npks)] for j in range(n_sets)])
        upper_bounds = flatten_params_npoly_npks([upper_coeffs, upper_freqs, upper_widths, upper_amps])
        lower_coeffs = np.array([[-np.inf for i in range(npoly)] for j in range(n_sets)])
        lower_widths = np.array([0 for i in range(npks)])
        lower_amps = np.array([[-np.inf for i in range(npks)] for j in range(n_sets)])
        lower_bounds = flatten_params_npoly_npks([lower_coeffs, lower_freqs, lower_widths, lower_amps])
        bounds = (lower_bounds, upper_bounds)

    # flatten params and execute fit
    params0_flattened = flatten_params_npoly_npks(params0)
    params0_packed = pack_params_npoly_npks(params0_flattened, npoly, npks, n_sets)
    n_params = len(params0_packed)
    f = ff.npoly_npk(npoly, npks, peak_type)
    res_lsq = opt.least_squares(residual_npoly_npks, params0_flattened, bounds=bounds, args=(x, ys, f, npoly, npks, n_sets, n_params))
    opt_params_flattened = res_lsq.x
    J = res_lsq.jac
    cov = np.linalg.inv(J.T.dot(J))
    var_flattened = np.sqrt(np.diagonal(cov))

    # evaluate fit and guess over x
    opt_params_packed = pack_params_npoly_npks(opt_params_flattened, npoly, npks, n_sets)
    x_fit = np.linspace(x[0], x[-1], 1000)
    ys_fit = [f(x_fit, *opt_params_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    ys_guess = [f(x_fit, *params0_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]

    # unflatten params for return
    opt_params = unflatten_params_npoly_npks(opt_params_flattened, npoly, npks, n_sets)
    var = unflatten_params_npoly_npks(var_flattened, npoly, npks, n_sets)

    return opt_params, var, x_fit, ys_fit, ys_guess

def flatten_params_npoly_npks(params):
    '''
    take parameters made for fit and pack them for residual function. params are in the form

    params = [[[polyamps]], [freqs], [widths]], [[amps]]]
    
    and outputs (given l polynomials n frequencies and m sets)

    params = [polyamp11,...,polyamp1l,...,polyampml,f1,..,fn,width1,...,widthn,amp11,...,amp1n,....ampmn]

    as a numpy array
    '''

    polyamps, freqs, widths, amps = params
    n_sets = len(amps)
    npoly = len(polyamps)
    npks = len(freqs)
    flattened_params = []
    for i in range(n_sets):
        flattened_params += [j for j in polyamps[i]]
    flattened_params+=[f for f in freqs]+[w for w in widths]
    for i in range(n_sets):
        flattened_params += [j for j in amps[i]]
    return np.array(flattened_params)

def pack_params_npoly_npks(params, npoly, npks, n_sets):
    '''
    take flattened parameters made for fit and pack them for residual function. params are in the form

    params = [polyamp11,...,polyamp1l,...,polyampml,f1,..,fn,width1,...,widthn,amp11,...,amp1n,....ampmn]
    
    and outputs

    params = [polyamp11,...polyamp1l, amp11, f1, width1,...,amp1n,fn,widthn,...,polyampm1,...polyampml,...ampmn, fn, widthn]

    '''
    nsetparams = npoly+3*npks
    packed_params = np.zeros(n_sets*nsetparams)
    for i in range(n_sets):
        packed_params[i*nsetparams:i*nsetparams+npoly] = params[i*npoly:(i+1)*npoly] # polyamps
        for j in range(npks):
            packed_params[i*nsetparams+npoly+3*j] = params[npoly*n_sets+j]
            packed_params[i*nsetparams+npoly+3*j+1] = params[npoly*n_sets+npks+j]
            packed_params[i*nsetparams+npoly+3*j+2] = params[npoly*n_sets+2*npks+i*npks+j]
    return np.array(packed_params)

def unflatten_params_npoly_npks(params, npoly, npks, n_sets):
    '''
    unpacks parameters back into original form, ie from 
    
    params = [polyamp11,...,polyamp1l,...,polyampml,f1,..,fn,width1,...,widthn,amp11,...,amp1n,....ampmn]

    to 

    params = [[[polyamps]], [freqs], [widths]], [[amps]]]
    '''
    polyamps = amps = np.array(params[:npoly*n_sets]).reshape((n_sets, npoly))
    freqs = np.array(params[npoly*n_sets:npoly*n_sets+npks])
    widths = np.array(params[npoly*n_sets+npks:npoly*n_sets+2*npks])
    amps = np.array(params[npoly*n_sets+2*npks:]).reshape((n_sets, npks))
    unflattened_params = [polyamps, freqs, widths, amps]
    return unflattened_params

def residual_npoly_npks(params, x, ys, f, npoly, npks, n_sets, n_params):
    '''
    computes residual for polynomials + peak
    '''
    params = pack_params_npoly_npks(params, npoly, npks, n_sets)
    ys_computed = [f(x, *params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    y = np.concatenate(ys)
    y_computed = np.concatenate(ys_computed)
    return np.abs(y - y_computed)