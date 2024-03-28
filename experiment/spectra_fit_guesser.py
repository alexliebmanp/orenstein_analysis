
import numpy as np
import matplotlib.pyplot as plt
import orenstein_analysis.experiment.fit_functions as ff
from orenstein_analysis.experiment import experiment_methods
from matplotlib.widgets import Slider, Button, TextBox
import pickle
import matplotlib
#matplotlib.use('QtAgg')
#from PyQt6.QtWidgets import QApplication

'''
Tool for fitting power spectra of a measurement, with possibility to do so simultaneously over numerous variables. 

Currently only designed to be used for under damped oscillator models (ie, n_over_damped=0)

'''

slider_label_size = 5

def ndamped_fit_guesser(meas, n_damped, n_over_damped, maxtime=None, vars=[r'$\Delta$ BS', r'$\delta r$ BS', r'$\gamma$ BS'], p0=None, length=20, height=18, bounds=None, filename='guess'):

    # setup fit functions
    n_sets = len(vars)
    n_freqs = n_damped+n_over_damped
    fosc = ff.ndamped_osc(n_damped, n_over_damped)

    # come up with appropriate starting point
    if p0==None:
        p0, bounds = p0_generate(meas, n_damped, n_over_damped, vars, bounds)
    elif bounds==None:
        p0_hold, bounds = p0_generate(meas, n_damped, n_over_damped, vars, bounds)

    # setup figures
    n_grid = 1000
    buffer=50
    spacing=50
    fig = plt.figure(figsize=(length, height)) # figsize=(length, height)
    plot_height_fraction = 0.3
    gs = fig.add_gridspec(n_grid, n_grid)
    plot_height = int(n_grid*plot_height_fraction)
    plot_horiz = int((n_grid-(n_sets-1)*buffer)/n_sets)
    gs_sliders = gs[plot_height+spacing:,:].subgridspec(n_grid, n_grid)
    ax=[]
    for i in range(n_sets):
        subax = fig.add_subplot(gs[:plot_height, (plot_horiz+buffer)*i:(plot_horiz+buffer)*(i+1)-buffer])
        ax.append(subax)
    ax = np.array(ax)
    fig.tight_layout()

    # obtain domain for guess and initial spectra of guess
    times = meas[f'Time Delay (ps)'].sel({f'Time Delay (ps)':slice(0,maxtime)}).data
    times_guess = np.linspace(times[0], times[-1],10000)
    guesses = experiment_methods.compute_ndamped_nsets(fosc, times_guess, p0, n_damped, n_over_damped, n_sets)

    # initialize plots
    plots = {}
    for ii, var in enumerate(vars):
        
        # plot data
        y = meas[var].sel({f'Time Delay (ps)':slice(0,maxtime)}).data
        meas[var].sel({f'Time Delay (ps)':slice(0,maxtime)}).plot(ax=ax[ii], marker='o', ms=4)
        meas[var].sel({f'Time Delay (ps)':slice(0,maxtime)}).plot(ax=ax[ii], marker='o', ms=4)
        ymin, ymax = np.min(y), np.max(y)
        upper_margin = 1.4
        lower_margin = 0.6
        ax[ii].set(ylim=(np.min([ymin*upper_margin, ymin*lower_margin]), np.max([ymax*upper_margin, ymax*lower_margin])))

        # initialize guess plots
        l1, = ax[ii].plot(times_guess, np.real(guesses[ii]), '--', color='red')
        plots[var] = l1

    # initialize sliders
    sliders, button = generate_sliders_from_p0(p0, vars, n_damped, n_over_damped, n_sets, gs_sliders, fig, bounds)

    # update plots function and saves p0 it goes
    update_func = lambda val: update_from_sliders(sliders, times_guess, fosc, vars, n_damped, n_over_damped, n_sets, fig, plots, filename)

    # save p0 func
    #save_func = lambda val: save_p0_from_sliders(sliders, n_sets, filename)
        

    # initiate updating
    flattened_sliders = experiment_methods.flatten_params(sliders)
    for s in flattened_sliders:
        s.on_changed(update_func)
    #button.on_clicked(save_func)

    #gs.tight_layout(fig)
    fig.tight_layout()
    fig.canvas.updateGeometry()
    plt.show()

def generate_sliders_from_p0(p0, vars, n_damped, n_over_damped, n_sets, gs_sliders, fig, bounds):
    '''
    Adds sliders to plot in following format:

            Background      Mode 1      Mode 2      Mode 3  ...

    Freqs:                  
    
    Damps:

    Amp 1

    Amp 2

    ....
  

    return slider objects in same format as p0.

    '''
    # unpack params
    freqs, damps, cs, d_amps, d_phis, od_amps1, od_amps2 = p0
    freqs_lower, damps_lower, cs_lower, d_amps_lower, d_phis_lower, od_amps1_lower, od_amps2_lower = bounds[0]
    freqs_upper, damps_upper, cs_upper, d_amps_upper, d_phis_upper, od_amps1_upper, od_amps2_upper = bounds[1]
    n_freqs = n_damped + n_over_damped

    # setup grid
    gs = gs_sliders
    n_height, n_width = gs.get_geometry()
    buffer_horiz = 60
    buffer_vert = 20
    numcols = n_freqs+2
    numrows = n_sets+3
    horiz_spacing = int((n_width - (numcols-1)*buffer_horiz)/numcols)
    vert_spacing = int((n_height - (numrows-1)*buffer_vert)/numrows)

    # setup labels on grid
    amp_phi_labels = []
    for var in vars:
        amp_phi_labels.append(f'Amplitude/Phase {var}')
    rowlabels = ['Frequency (THz)', 'Damping (THz)']+amp_phi_labels
    collabels = ['Offset']+[f'Mode {ii+1}' for ii in range(n_freqs)]
    for row, label in enumerate(rowlabels):
        ax = fig.add_subplot(gs[(row+1)*(vert_spacing+buffer_vert):(row+2)*(vert_spacing+buffer_vert)-buffer_vert,0:(horiz_spacing+buffer_horiz)-buffer_horiz])
        ax.text(0.5, 0.5, label, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    for col, label in enumerate(collabels):
        ax = fig.add_subplot(gs[0:(vert_spacing+buffer_vert)-buffer_vert,(col+1)*(horiz_spacing+buffer_horiz):(col+2)*(horiz_spacing+buffer_horiz)-buffer_horiz])
        ax.text(0.5, 0.5, label, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    fmt = "%.3e"

    freqs_sliders = []
    for ii, f in enumerate(freqs):
        gs_slider = gs[(vert_spacing+buffer_vert):2*(vert_spacing+buffer_vert)-buffer_vert,(ii+2)*(horiz_spacing+buffer_horiz):(ii+3)*(horiz_spacing+buffer_horiz)-buffer_horiz].subgridspec(2, 10)
        slider_ax = fig.add_subplot(gs_slider[:,:9])
        lower = freqs_lower[ii]
        upper = freqs_upper[ii]
        slider = Slider(slider_ax, '', lower, upper, valinit=f, valfmt=fmt)
        freqs_sliders.append(slider)
    damps_sliders = []
    for ii, d in enumerate(damps):
        gs_slider = gs[2*(vert_spacing+buffer_vert):3*(vert_spacing+buffer_vert)-buffer_vert,(ii+2)*(horiz_spacing+buffer_horiz):(ii+3)*(horiz_spacing+buffer_horiz)-buffer_horiz].subgridspec(2, 10)
        slider_ax = fig.add_subplot(gs_slider[:,:9])
        lower = damps_lower[ii]
        upper = damps_upper[ii]
        slider = Slider(slider_ax, '', lower, upper, valinit=d, valfmt=fmt)
        damps_sliders.append(slider)
    d_amps_sliders = []
    d_phis_sliders = []
    for ii in range(n_sets):
        amp_set_sliders = []
        phi_set_sliders = []
        for jj in range(n_damped):
            gs_slider = gs[(3+ii)*(vert_spacing+buffer_vert):(4+ii)*(vert_spacing+buffer_vert)-buffer_vert,(jj+2)*(horiz_spacing+buffer_horiz):(jj+3)*(horiz_spacing+buffer_horiz)-buffer_horiz].subgridspec(3, 10)
            amp_slider_ax = fig.add_subplot(gs_slider[0,:9])
            phi_slider_ax = fig.add_subplot(gs_slider[1,:9])
            amp_lower = d_amps_lower[ii][jj]
            amp_upper = d_amps_upper[ii][jj]
            phi_lower = d_phis_lower[ii][jj]
            phi_upper = d_phis_upper[ii][jj]
            amp_slider = Slider(amp_slider_ax, '', amp_lower, amp_upper, valinit=d_amps[ii][jj], valfmt=fmt)
            phi_slider = Slider(phi_slider_ax, '', phi_lower, phi_upper, valinit=d_phis[ii][jj], valfmt=fmt)
            amp_set_sliders.append(amp_slider)
            phi_set_sliders.append(phi_slider)
        d_amps_sliders.append(amp_set_sliders)
        d_phis_sliders.append(phi_set_sliders)
    cs_sliders = []
    for ii, c in enumerate(cs):
        gs_slider = gs[(3+ii)*(vert_spacing+buffer_vert):(4+ii)*(vert_spacing+buffer_vert)-buffer_vert,(horiz_spacing+buffer_horiz):2*(horiz_spacing+buffer_horiz)-buffer_horiz].subgridspec(2, 10)
        slider_ax = fig.add_subplot(gs_slider[:,:9])
        lower = cs_lower[ii]
        upper = cs_upper[ii]
        slider = Slider(slider_ax, '', lower, upper, valinit=c, valfmt=fmt)
        cs_sliders.append(slider)


    ## add other sliders if desired here

    sliders = [np.array(freqs_sliders), np.array(damps_sliders), np.array(cs_sliders), np.array(d_amps_sliders), np.array(d_phis_sliders), np.array([[]]), np.array([[]])]
    #sliders_min = [np.array(freqs_sliders_min), np.array(damps_sliders_min), np.array(d_amps_sliders_min), np.array([[]]), np.array([[]])]
    #sliders_max = [np.array(freqs_sliders_max), np.array(damps_sliders_max), np.array(d_amps_sliders_max), np.array([[]]), np.array([[]])]
    #sliders_bounds = (sliders_min, sliders_max)
    #gs.tight_layout(fig)

    #button_ax = fig.add_subplot(gs[-1*vert_spacing:,:])
    #button = Button(button_ax, 'Save guess')
    button=0

    return sliders, button

def update_from_sliders(sliders, times_guess, fosc, vars, n_damped, n_over_damped, n_sets, fig, plots, filename):
    p0 = read_sliders(sliders, n_sets)
    update_plots_from_p0(p0, times_guess, fosc, n_damped, n_over_damped, n_sets, vars, plots)
    save_p0(p0, filename)
    update_sliders_range(sliders, fig)

def update_plots_from_p0(p0, times_guess, fosc, n_damped, n_over_damped, n_sets, vars, plots):
    guesses = experiment_methods.compute_ndamped_nsets(fosc, times_guess, p0, n_damped, n_over_damped, n_sets)
    for ii, var in enumerate(vars):
        l1 = plots[var]
        l1.set_ydata(guesses[ii])

def p0_generate(meas, n_damped, n_over_damped, vars, bounds):

    vars_pow = [f'P[{var}]' for var in vars]
    n_freqs = n_damped + n_over_damped
    x = meas['Time Delay (ps)'].data*1e-9
    f = meas['Frequency (THz)'].data
    ys = [meas[var].data for var in vars_pow]
    n_sets = len(vars)

    f0 = np.array([f[np.argmax(ys[0])] for i in range(n_damped+n_over_damped)])
    damps0 = np.array([(2/np.max(x))*1e-12 for i in range(n_damped+n_over_damped)])
    d_amps0 = np.array([[np.max(ys[j]) for i in range(n_damped)] for j in range(n_sets)])
    p0=[f0, damps0, d_amps0, np.array([[]]), np.array([[]])]

    if bounds==None:
        upper_freqs = np.array([np.max(f)/2 for i in range(n_freqs)])
        upper_damps = np.array([50/np.max(x)*1e-12 for i in range(n_freqs)])
        upper_d_amps = np.array([[np.max(ys[j]) for i in range(n_damped)] for j in range(n_sets)])
        upper_bounds= [upper_freqs, upper_damps, upper_d_amps, np.array([[]]), np.array([[]])]

        lower_freqs = np.array([0 for i in range(n_freqs)])
        lower_damps = np.array([0 for i in range(n_freqs)])
        lower_d_amps = np.array([[0 for i in range(n_damped)] for j in range(n_sets)])
        lower_bounds= [lower_freqs, lower_damps, lower_d_amps, np.array([[]]), np.array([[]])]
        bounds = (lower_bounds, upper_bounds)
    else:
        bounds=bounds

    return p0, bounds

def read_sliders(sliders, n_sets):
    
    sfreqs, sdamps, scs, sd_amps, sd_phis, sod_amp1, sod_amp2 = sliders
    freqs = np.array([s.val for s in sfreqs])
    damps = np.array([s.val for s in sdamps])
    cs = np.array([s.val for s in scs])
    d_amps = np.array([[s.val for s in sd_amps[i]] for i in range(n_sets)])
    d_phis = np.array([[s.val for s in sd_phis[i]] for i in range(n_sets)])
    p0 = [freqs, damps, cs, d_amps, d_phis, np.array([[]]), np.array([[]])]
    return p0

def update_sliders_range(sliders, fig):
    sfreqs, sdamps, scs, sd_amps, sd_phis, sod_amp1, sod_amp2 = sliders
    for s in sfreqs:
        #find_new_slider_range(s)
        pass
    for s in scs:
        find_new_slider_range(s)
    for s in sdamps:
        find_new_slider_range(s)
    for i in sd_amps:
        for s in i:
            find_new_slider_range(s)

def find_new_slider_range(s):
    rate = 0.1
    if s.val==s.valmin:
        s.valmin = (1-rate)*s.valmin
        s.valmax = (1-rate)*s.valmax
        s.ax.set_xlim(s.valmin,s.valmax)
    elif s.val==s.valmax:
        s.valmin = (1+rate)*s.valmin
        s.valmax = (1+rate)*s.valmax
        s.ax.set_xlim(s.valmin,s.valmax)

def save_p0(p0, filename, fmt_string='%12.3e'):
    freqs, damps, cs, amps, phis, amps1_od, amps2_od = p0
    freqs = np.insert(freqs, 0, 0)
    damps = np.insert(damps, 0, 0)
    amps = np.array([np.insert(amps[ii], 0, cs[ii]) for ii in range(len(amps))])
    phis = np.array([np.insert(phis[ii], 0, 0) for ii in range(len(phis))])
    data = [freqs, damps]
    for ii in range(len(amps)):
        data.append(amps[ii])
    for ii in range(len(phis)):
        data.append(phis[ii])
    data = np.transpose(np.array(data))

    headers = ['Frequency', 'Damping']
    for i in range(len(amps)):
        headers.append(f'Amplitude {i+1}')
    for i in range(len(amps)):
        headers.append(f'Phase {i+1}')

    # Format string for the data
    fmt = [fmt_string for _ in range(len(headers))]

    # Save data with headers
    np.savetxt(filename, data, header=' '.join(headers), fmt=fmt)

def load_p0(filename):
    data = np.genfromtxt(filename, skip_header=1)
    nsets = int((len(data[0,:])-2)/2)
    freqs = data[1:,0]
    damps = data[1:,1]
    cs = data[0,2:2+nsets]
    amps = data[1:,2:2+nsets].transpose()
    phis = data[1:,2+nsets:].transpose()
    p0 = [freqs, damps, cs, amps, phis, np.array([[]]), np.array([[]])]
    return p0

def add_peak_to_p0(p0, idx, freq):
    freqs0, damps0, cs0, amps_d0, phis_d0, amps_od0, amps2_od0 = p0
    freqs0 = np.insert(freqs0, idx, freq)
    damps0 = np.insert(damps0, idx, damps0[-1])
    amps_d0 = np.array([np.insert(ii, idx, 1e-8) for ii in amps_d0])
    phis_d0 = np.array([np.insert(ii, idx, 1e-8) for ii in phis_d0])
    p0 = [freqs0, damps0, cs0, amps_d0, phis_d0, amps_od0, amps2_od0]
    return p0

def remove_peak_to_p0(p0, idx):
    freqs0, damps0, cs0, amps_d0, phis_d0, amps_od0, amps2_od0 = p0
    freqs0 = np.delete(freqs0, idx)
    damps0 = np.delete(damps0, idx)
    amps_d0 = np.array([np.delete(ii, idx) for ii in amps_d0])
    phis_d0 = np.array([np.delete(ii, idx) for ii in phis_d0])
    p0 = [freqs0, damps0, cs0, amps_d0, phis_d0, amps_od0, amps2_od0]
    return p0

def save_fit_settings(settings_dict, filename):
    with open(filename+'.dat', 'w') as f:
        for key in list(settings_dict.keys()):
            f.write(f'{key}:\t{settings_dict[key]}\n')
    with open(filename, 'wb') as f:
        pickle.dump(settings_dict, f)