import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
from inspect import signature

####################
### Peak fitting ###
####################

def gauss(x, a, x0, s):
    """
    Gaussian peak function.

    Args:
        - x (array of float): domain to be applied to function
        - a (float): peak area
        - x0 (float): peak centre
        - s (float): peak width
    Return:
        - array of float
    """

    return a*np.sqrt(1./(2*np.pi*s**2))*np.exp(-0.5*(x-x0)**2/s**2)

def lorentz(x, a, x0, g):
    """
    Lorentzian peak function.

    Args:
        - x (array of float): domain to be applied to function
        - a (float): peak area
        - x0 (float): peak centre
        - g (float): peak width (FWFM/2)
    Return:
        - array of float
    """
    return a/(np.pi * g) * g**2 / ((x - x0)**2 + g**2)

def npoly_func(npoly):
    """
    Arbitrary degree polynomial, zero-based (npoly=0 -> constant)
    Args:
        - npoly (int): degree of polynomial (0 based, i.e. 2 is linear)
    Return:

        - lambda function defining the composite function
    """

    return lambda x, *A: np.sum([A[ii]*x**ii for ii in range(npoly)], axis = 0)

def npk_func(npk, peak_type=0):
    """
    Multi-peak function using Lorentzian (0) or Gaussian (1) peaks
    Args:
        - npk (int): number of peaks to fit
    Return:
        - lambda function defining the multipeak function
    """
    peak_function, nparams = get_peak_type(peak_type)

    return lambda x,*A: np.sum([peak_function(x,*A[nparams*ii:(nparams)*(ii+1)]) for ii in range(npk)],axis = 0)

def npoly_npk(npks, npoly, peak_type=0):
    """
    Multi-peak function, defined with an arbitrary degree polynomial, combined
    with an arbitrary number of Lorentzian peak functions

    Args:

        - npks (int): number of peaks to fit

        - npoly (int): degree of polynomial (note 0 based, i.e. 2 is linear)

    *return*:

        - lambda function defining the composite function
    """
    peak_function, nparams = get_peak_type(peak_type)

    return lambda x, *A: npoly_func(npoly)(x,*A[:npoly]) + npk_func(npks, peak_type=peak_type)(x,*A[npoly:])

def get_peak_type(peak_type):
    """
    Process peak type to select executable peak function, and identify the number
    of arguments required for the specified function (minus 1 for the domain variable).

    Args:
        - peak_type (int or exec): if int, 0 (Lorentzian) or 1 (Gaussian).
            User-defined executable is also permitted

    Return:
        - peak_function (executable): peak function
        - nparams (int): number of peak parameters
    """

    if peak_type == 0:
        peak_function = lorentz
    elif peak_type == 1:
        peak_function = gauss
    else:
        if callable(peak_type):
            peak_function = peak_type
        else:
            raise TypeError('peak_type must be int (0/1) or executable method.')

    peak_signature = signature(peak_function)
    nparams = len(peak_signature.parameters) - 1

    return peak_function, nparams

########################
### Fourier Analysis ###
########################

def damped_cos_phi(x, f0, g, A, phi):
    '''
    Damped oscillatory function
    '''
    return A*np.exp(-x*g)*np.cos(2*np.pi*f0*x+phi)

def FFT_damped_sin(f, f0, g, A):
    '''
    Fourier transform of damped sine function
    '''
    return A*f0*(np.pi*2)/((g+1j*f*np.pi*2)**2+(f0*np.pi*2)**2)

def FFT_damped_cos(f, f0, g, A):
    '''
    Fourier transform of damped sine function
    '''
    return A*(g+1j*(2*np.pi*f))/((g+1j*f*np.pi*2)**2+(f0*np.pi*2)**2)

def FFT_damped_cos_phi(f, f0, g, A, phi):
    '''
    Fourier transform of damped sinusoid with a phase
    '''
    return A*np.cos(phi)*FFT_damped_cos(f, f0, g, A) - A*np.sin(phi)*FFT_damped_sin(f, f0, g, A)

def over_damped(x, f0, g, A1, A2):
    '''
    Over damped oscillator
    '''
    u1=g-np.sqrt(g**2-(2*np.pi*f0)**2)
    u2=g+np.sqrt(g**2-(2*np.pi*f0)**2)   
    return A1*np.exp(-x*u1)+A2*np.exp(-x*u2)

def FFT_over_damped(f, f0, g, A1, A2):
    '''
    Fourier transform of over damped oscillator
    '''
    u1=g-np.sqrt(g**2-(2*np.pi*f0)**2)
    u2=g+np.sqrt(g**2-(2*np.pi*f0)**2)
    return A1/(u1+1j*2.*np.pi*f)+A2/(u2+1j*2.*np.pi*f)

def ndamped_osc(n_damped, n_over_damped):
    '''
    Function of arbitrary combinations of damped and overdamped oscillators
    '''
    def func(x,params):
        fun=0
        c, params1, params2 = params[0], params[1:4*n_damped+1], params[4*n_damped+1:]
        # Number of damped modes 
        for ii in range (0, n_damped):
            fun = fun+damped_cos_phi(x,*params1[(0+ii*4):(4+ii*4)])
            
        # Number of overdamped modes 
        for jj in range (0, n_over_damped):
            fun = fun+over_damped(x,*params2[(0+jj*4):(4+jj*4)])
            
        return fun + c

    return func

def FFT_ndamped_osc(n_damped, n_over_damped):
    '''
    Function of arbitrary combinations of damped and overdamped oscillators
    '''
    def func(x,params):
        fun=0
        c, params1, params2 = params[0], params[1:4*n_damped+1], params[4*n_damped+1:]
        # Number of damped modes 
        for ii in range (0, n_damped):
            fun = fun+FFT_damped_cos_phi(x,*params1[(0+ii*4):(4+ii*4)])
            
        # Number of overdamped modes 
        for jj in range (0, n_over_damped):
            fun = fun+FFT_over_damped(x,*params2[(0+jj*4):(4+jj*4)])
            
        return fun #+c

    return func