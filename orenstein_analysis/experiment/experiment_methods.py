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
    coord_vars[x_var+f' ({y_var} fit)'] = xfit
    data_vars[y_var+' (fit)'] = ((x_var+f' ({y_var} fit)'), yfit)
    return data_vars, coord_vars, attrs

def meas_fft(measurement, x_var, y_var, padding_fraction=1):

    xdata = measurement[x_var].data
    ydata = measurement[y_var].data
    if padding_fraction!=0:
        dx = xdata[1]-xdata[1]
        l = len(xdata)
        xdata = np.concatenate([xdata,xdata[-1]+dx*(1+np.arange(int(padding_fraction*l)))])
        ydata = np.concatenate([ydata,np.zeros(int(padding_fraction*l))])
    yfft = fft.fft(ydata)
    freqs = fft.fftfreq(len(xdata), xdata[1]-xdata[0])
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
    data_vars[f'Re(FFT[{y_var}])'] = ((freqname), np.real(yfft))
    data_vars[f'Im(FFT[{y_var}])'] = ((freqname), np.imag(yfft))
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

def fit_ndamped_meas(measurement, x_var, y_vars, n_damped, n_over_damped, p0=None, freq_bounds=None, bounds=None):

    if type(y_vars) is not list:
        y_vars = [y_vars]
    x = measurement[x_var].data.flatten()
    ys = [measurement[y_var].data.flatten() for y_var in y_vars]
    opt_params, var, x_fit, ys_fit, ys_guess = fit_ndamped(x, ys, n_damped, n_over_damped, p0, bounds=bounds, freq_bounds=freq_bounds)
    freqs, damps, cs, d_amps, d_phis, od_amps, od_phis = opt_params
    var_freqs, var_damps, var_cs, var_d_amps, var_d_phis, var_od_amps, var_od_phis = var
    data_vars = {}
    coord_vars= {}
    attrs = {}
    coord_vars['Mode Params'] = ['C', 'Frequency (THz)', 'Damping (THz)', 'Amplitude', 'Phase', 'Type']
    for ii, y_var in enumerate(y_vars):
        coord_vars[x_var+f' ({y_var} fit)'] = x_fit
        data_vars[y_var+' (fit)'] = ((x_var+f' ({y_var} fit)'), ys_fit[ii])
        data_vars[y_var+' (fit guess)'] = ((x_var+f' ({y_var} fit)'), ys_guess[ii])
        for jj, f in enumerate(freqs):
            if jj<n_damped:
                mode_params = [cs[ii], f, damps[jj], d_amps[ii][jj], d_phis[ii][jj], 'under damped']
                mode_var_params = [var_cs[ii], var_freqs[jj], var_damps[jj], var_d_amps[ii][jj], var_d_phis[ii][jj], 'under damped']
                data_vars[f'Mode {jj+1} Params ({y_var} fit)'] = (('Mode Params'), mode_params)
                data_vars[f'Mode {jj+1} Params Variance ({y_var} fit)'] = (('Mode Params'), mode_var_params)
            else:
                mode_params = [cs[ii], f, damps[jj-n_damped], od_amps[ii][jj-n_damped], od_phis[ii][jj-n_damped], 'under damped']
                mode_var_params = [var_cs[ii], var_freqs[jj-n_damped], var_damps[jj-n_damped], var_od_amps[ii][jj-n_damped], var_od_phis[ii][jj-n_damped], 'over damped']
                data_vars[f'Mode {jj+1} Params ({y_var} fit)'] = (('Mode Params'), mode_params)
                data_vars[f'Mode {jj+1} Params Variance ({y_var} fit)'] = (('Mode Params'), mode_var_params)            
    for ii, f in enumerate(freqs):
            data_vars[f'Mode {ii+1} Frequency (THz)'] = ((), f)
            data_vars[f'Mode {ii+1} Frequency (THz) Variance'] = ((), var_freqs[ii])
            data_vars[f'Mode {ii+1} Damping (THz)'] = ((), damps[ii])
            data_vars[f'Mode {ii+1} Damping (THz) Variance'] = ((), var_damps[ii])
            if ii<n_damped:
                data_vars[f'Mode {ii+1} Type'] = ((), 'under damped')
            else:
                data_vars[f'Mode {ii+1} Type'] = ((), 'over damped')
    attrs['ndamped fit params'] = opt_params

    return data_vars, coord_vars, attrs

def fit_fft_ndamped_meas(measurement, x_var, y_vars, n_damped, n_over_damped, p0=None, freq_bounds=None, bounds=None):

    if type(y_vars) is not list:
        y_vars = [y_vars]
    x = measurement[x_var].data.flatten()
    ys = [measurement[y_var].data.flatten() for y_var in y_vars]
    opt_params, var, x_fit, ys_fit, ys_guess = fit_fft_ndamped(x, ys, n_damped, n_over_damped, p0, bounds=bounds, freq_bounds=freq_bounds)
    freqs, damps, cs, d_amps, d_phis, od_amps, od_phis = opt_params
    var_freqs, var_damps, var_cs, var_d_amps, var_d_phis, var_od_amps, var_od_phis = var
    data_vars = {}
    coord_vars= {}
    attrs = {}
    coord_vars['Mode Params'] = ['C', 'Frequency (THz)', 'Damping (Tz)', 'Amplitude', 'Phase', 'Type']
    for ii, y_var in enumerate(y_vars):
        coord_vars[x_var+f' ({y_var} fit)'] = x_fit
        data_vars[y_var+' (fit)'] = ((x_var+f' ({y_var} fit)'), ys_fit[ii])
        data_vars[y_var+' (fit guess)'] = ((x_var+f' ({y_var} fit)'), ys_guess[ii])
        data_vars[f'Re({y_var})'+' (fit)'] = ((x_var+f' ({y_var} fit)'), np.real(ys_fit[ii]))
        data_vars[f'Re({y_var})'+' (fit guess)'] = ((x_var+f' ({y_var} fit)'), np.real(ys_guess[ii]))
        data_vars[f'Im({y_var})'+' (fit)'] = ((x_var+f' ({y_var} fit)'), np.imag(ys_fit[ii]))
        data_vars[f'Im({y_var})'+' (fit guess)'] = ((x_var+f' ({y_var} fit)'), np.imag(ys_guess[ii]))
        for jj, f in enumerate(freqs):
            if jj<n_damped:
                mode_params = [cs[ii], f, damps[jj], d_amps[ii][jj], d_phis[ii][jj], 'under damped']
                mode_var_params = [var_cs[ii], var_freqs[jj], var_damps[jj], var_d_amps[ii][jj], var_d_phis[ii][jj], 'under damped']
                data_vars[f'Mode {jj+1} Params ({y_var} fit)'] = (('Mode Params'), mode_params)
                data_vars[f'Mode {jj+1} Params Variance ({y_var} fit)'] = (('Mode Params'), mode_var_params)
            else:
                mode_params = [cs[ii], f, damps[jj-n_damped], od_amps[ii][jj-n_damped], od_phis[ii][jj-n_damped], 'under damped']
                mode_var_params = [var_cs[ii], var_freqs[jj-n_damped], var_damps[jj-n_damped], var_od_amps[ii][jj-n_damped], var_od_phis[ii][jj-n_damped], 'over damped']
                data_vars[f'Mode {jj+1} Params ({y_var} fit)'] = (('Mode Params'), mode_params)
                data_vars[f'Mode {jj+1} Params Variance ({y_var} fit)'] = (('Mode Params'), mode_var_params)            
    for ii, f in enumerate(freqs):
            data_vars[f'Mode {ii+1} FFT Frequency (THz)'] = ((), f)
            data_vars[f'Mode {ii+1} FFT Frequency (THz) Variance'] = ((), var_freqs[ii])
            data_vars[f'Mode {ii+1} FFT Damping (THz)'] = ((), damps[ii])
            data_vars[f'Mode {ii+1} FFT Damping (THz) Variance'] = ((), var_damps[ii])
            if ii<n_damped:
                data_vars[f'Mode {ii+1} FFT Type'] = ((), 'under damped')
            else:
                data_vars[f'Mode {ii+1} FFT Type'] = ((), 'over damped')
    attrs['FFT ndamped fit params'] = opt_params

    return data_vars, coord_vars, attrs

###############
### Helpers ###
###############s

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

def fit_ndamped(x, ys, n_damped, n_over_damped, params0=None, bounds=None, freq_bounds=None):
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
        od_amps0 = np.array([[np.max(ys[j]) for i in range(n_over_damped)] for j in range(n_sets)])
        od_phis0 = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        params0=[freqs0, damps0, cs0, d_amps0, d_phis0, od_amps0, od_phis0]
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
        upper_od_amps = np.array([[np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        upper_od_phis = np.array([[2*np.pi for i in range(n_over_damped)] for j in range(n_sets)])
        upper_bounds=flatten_params([upper_freqs, upper_damps, upper_cs, upper_d_amps, upper_d_phis, upper_od_amps, upper_od_phis])
        lower_damps = np.array([0 for i in range(n_freqs)])
        lower_cs = np.array([-np.inf]*n_sets)
        lower_d_amps = np.array([[-np.inf for i in range(n_damped)] for j in range(n_sets)])
        lower_d_phis = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        lower_od_amps = np.array([[-np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        lower_od_phis = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        lower_bounds=flatten_params([lower_freqs, lower_damps, lower_cs, lower_d_amps, lower_d_phis, lower_od_amps, lower_od_phis])
        bounds = (lower_bounds, upper_bounds)



    
    # flatten params and execute fit
    params0_flattened = flatten_params(params0)
    params0_packed = pack_params(params0_flattened, n_freqs, n_sets)
    n_params = len(params0_packed)
    fosc = ff.ndamped_osc(n_damped, n_over_damped)
    res_lsq = opt.least_squares(residual_fft_ndamped, params0_flattened, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params))
    opt_params_flattened = res_lsq.x
    J = res_lsq.jac
    cov = np.linalg.inv(J.T.dot(J))
    var_flattened = np.sqrt(np.diagonal(cov))

    # evaluate fit and guess over x
    opt_params_packed = pack_params(opt_params_flattened, n_freqs, n_sets)
    x_fit = np.linspace(x[0], x[-1], 1000)
    ys_fit = [fosc(x_fit, opt_params_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    ys_guess = [fosc(x_fit, params0_packed[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]

    # unflatten params for return
    opt_params = unflatten_params(opt_params_flattened, n_damped, n_over_damped, n_sets)
    var = unflatten_params(var_flattened, n_damped, n_over_damped, n_sets)

    return opt_params, var, x_fit, ys_fit, ys_guess

def fit_fft_ndamped(x, ys, n_damped, n_over_damped, params0=None, bounds=None, freq_bounds=None):
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
        cs0 = np.array([0]*n_sets)
        d_amps0 = np.array([[np.real(np.max(ys[j])) for i in range(n_damped)] for j in range(n_sets)])
        d_phis0 = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        od_amps0 = np.array([[np.real(np.max(ys[j])) for i in range(n_over_damped)] for j in range(n_sets)])
        od_phis0 = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        params0=[freqs0, damps0, cs0, d_amps0, d_phis0, od_amps0, od_phis0]
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
        upper_od_amps = np.array([[np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        upper_od_phis = np.array([[2*np.pi for i in range(n_over_damped)] for j in range(n_sets)])
        upper_bounds=flatten_params([upper_freqs, upper_damps, upper_cs, upper_d_amps, upper_d_phis, upper_od_amps, upper_od_phis])
        lower_damps = np.array([0 for i in range(n_freqs)])
        lower_cs = np.array([-np.inf]*n_sets)
        lower_d_amps = np.array([[-np.inf for i in range(n_damped)] for j in range(n_sets)])
        lower_d_phis = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        lower_od_amps = np.array([[-np.inf for i in range(n_over_damped)] for j in range(n_sets)])
        lower_od_phis = np.array([[0 for i in range(n_over_damped)] for j in range(n_sets)])
        lower_bounds=flatten_params([lower_freqs, lower_damps, lower_cs, lower_d_amps, lower_d_phis, lower_od_amps, lower_od_phis])
        bounds = (lower_bounds, upper_bounds)
    # flatten params and execute fit
    params0_flattened = flatten_params(params0)
    params0_packed = pack_params(params0_flattened, n_freqs, n_sets)
    n_params = len(params0_packed)
    fosc = ff.FFT_ndamped_osc(n_damped, n_over_damped)
    res_lsq = opt.least_squares(residual_fft_ndamped, params0_flattened, bounds=bounds, args=(x, ys, fosc, n_freqs, n_sets, n_params))
    opt_params_flattened = res_lsq.x
    J = res_lsq.jac
    #cov = np.linalg.inv(J.T.dot(J))
    #var_flattened = np.sqrt(np.diagonal(cov))
    var_flattened = opt_params_flattened

    # evaluate fit and guess over x
    opt_params_packed = pack_params(opt_params_flattened, n_freqs, n_sets)
    n_params = len(opt_params_packed)
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

def residual_fft_ndamped(params, x, ys, fosc, n_freqs, n_sets, n_params):
    '''
    computes residual for ndamped oscillators and 
    '''
    params = pack_params(params, n_freqs, n_sets)
    ys_computed = [fosc(x, params[i*int(n_params/n_sets):(i+1)*int(n_params/n_sets)]) for i in range(n_sets)]
    y = np.concatenate(ys)
    y_real, y_imag = np.real(y), np.imag(y)
    y_computed = np.concatenate(ys_computed)
    y_computed_real, y_computed_imag = np.real(y_computed), np.imag(y_computed)
    y_total = np.concatenate([y_real, y_imag])
    y_computed_total = np.concatenate([y_computed_real, y_computed_imag])
    return np.abs(y_total - y_computed_total)