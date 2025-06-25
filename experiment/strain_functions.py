import numpy as np

# defining razorbill paramters
AREA = (5.95)*((1e-3)**2)
D0 = 68.68e-6
C_A = 0.04
C_0 = 0.808
EPS = ((C_0-C_A))*D0/AREA # had factor of 1e-12 for some reason...? 
ALPHA = (C_0 - C_A)*D0 #EPS*AREA

####
#### Ti thermal expansion
####
Ti_THERMAL_EXPANSION = np.loadtxt('/Users/oxide/Documents/research/orenstein/code/notebooks/strain_cell_diagnostics/titanium_thermal_expansion.txt', delimiter=',')
Ti_THERMAL_EXP = Ti_THERMAL_EXPANSION[:,1]*1e-5
Ti_THERMAL_EXP_TEMPS = Ti_THERMAL_EXPANSION[:,0]

###
### dummy experiment
###
DATA_DUMMY = np.loadtxt('/Users/oxide/Documents/research/orenstein/code/notebooks/strain_cell_diagnostics/titanium_dummy_capacitance_vs_temperature_cooldown.dat', skiprows=1)
IDX = np.argsort(DATA_DUMMY[:,0])
TEMP_DUMMY = DATA_DUMMY[IDX,0]
C_DUMMY = DATA_DUMMY[IDX,1]
CALIBRATION_CAP = [TEMP_DUMMY, C_DUMMY]

##
##  Basic conversion between capacitance and displacement
##

def cap_to_dl(cap):
    return ALPHA/(cap - C_A) - D0

def dl_to_cap(dl):
    return ALPHA/(dl + D0) + C_A

###############################################
### Methods for relative strain measurement ###
###############################################

def get_target_capacitance(cap_curr, temp, strain_applied_curr, strain_targ, l0, thermal_expansion='none', transmission_frac=1):
    '''
    given a current capacitance, current strain, temperature, and a target difference in strain, returns the desired final capacitance to aim for.

    strain is calculated as

        strain = \eta (dL/L0 + differential_thermal_expansion)

    args:
        = cap_curr:                         current reading on capacitor (pF)
        - temp:                             current temperature in K
        - applied_strain_curr:              currenet applied strain (unitless)
        - strain_targ:                      target net strain (unitless)
        - l0:                               length of sample (gap) at 0 V and 295 K in m
        - thermal_expansion:                (temp (K), thermal_expansion (unitless)) tuple or list with thermal expansion data dL(T)/L(T)
        - transmission_frac:                \eta in the above equation, giving the fraction of dL that transmits to the sample

    returns:
        - dl:                                displacement
    '''

    if thermal_expansion!='none':
        thermal_exp_temps = thermal_expansion[0]
        thermal_exp = thermal_expansion[1]
        thermal_exp_val = np.interp(temp, thermal_exp_temps, thermal_exp)
        Ti_thermal_exp_val = np.interp(temp, Ti_THERMAL_EXP_TEMPS, Ti_THERMAL_EXP)
        diff_therm_exp = Ti_thermal_exp_val - thermal_exp_val
        #l0 = (1+thermal_exp_val)*l0
    else:
        diff_therm_exp = 0

    dl_curr = cap_to_dl(cap_curr)

    #strain_targ = (dl_new/l0 + strain_applied_curr + diff_therm_exp)*transmission_frac
    dl_new = l0*(strain_targ/transmission_frac - strain_applied_curr - diff_therm_exp)

    dl_target = dl_curr + dl_new
    cap_target = dl_to_cap(dl_target)

    new_applied_strain = dl_new/l0

    return cap_target, new_applied_strain

###############################################
### Methods for absolute strain measurement ###
###############################################

##
## Different methods for correcting displacement measurement
##

def cap_to_dl_corr_1(cap, temp, cap0_295K, temperature_calibration='dummy'):
    '''
    correcting dL as capacitive offsets

    args:
        - cap:                              current capacitance in pF
        - temp:                             current temperature in K
        - cap0_295K:                        capacitance measured at 0 V and 295 K in pF
        - temperature_calibration:          (temp (K), capacitances (pF)) tuple or list with data from a Ti dummy calibration run over temperature range of interest

    returns:
        - dl_cor:                           corrected displacement dl
    '''

    # temperature correction
    if temperature_calibration=='dummy':
        temp_dummy, c_dummy = CALIBRATION_CAP
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K
    elif temperature_calibration=='none':
        Cp1 = 0
    else:
        temp_dummy, c_dummy = temperature_calibration
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K

    # parasitic offset
    Cp2 = cap0_295K - C_0

    cap_offsets = Cp1 + Cp2
    dl_offsets = 0

    dl_corr = cap_to_dl(cap - cap_offsets) - dl_offsets

    return dl_corr

def dl_to_cap_corr_1(dl_corr, temp, cap0_295K, temperature_calibration='dummy', thermal_expansion='none', transmission_frac=1):
    '''
    correcting dL as capacitive offsets

    args:
        - dl_corr:                          corrected displacement dl
        - temp:                             current temperature in K
        - cap0_295K:                        capacitance measured at 0 V and 295 K in pF
        - temperature_calibration:          (temp (K), capacitances (pF)) tuple or list with data from a Ti dummy calibration run over temperature range of interest

    returns:
        - cap:                              uncorrected capacitance
    '''

    # temperature correction
    if temperature_calibration=='dummy':
        temp_dummy, c_dummy = CALIBRATION_CAP
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K
    elif temperature_calibration=='none':
        Cp1 = 0
    else:
        temp_dummy, c_dummy = temperature_calibration
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K

    # parasitic offset
    Cp2 = cap0_295K - C_0

    cap_offsets = Cp1 + Cp2
    dl_offsets = 0

    dl_corr = cap_to_dl(cap - cap_offsets) - dl_offsets
    cap = dl_to_cap(dl_corr - dl_offsets) + cap_offsets

    return cap

def cap_to_dl_corr_2(cap, temp, cap0_295K, temperature_calibration='dummy'):
    '''
    correcting dL as displacement offsets

    args:
        - cap:                              current capacitance in pF
        - temp:                             current temperature in K
        - cap0_295K:                        capacitance measured at 0 V and 295 K in pF
        - temperature_calibration:          (temp (K), capacitances (pF)) tuple or list with data from a Ti dummy calibration run over temperature range of interest

    returns:
        - dl_cor:                           corrected displacement dl
    '''

    # temperature correction
    if temperature_calibration=='dummy':
        temp_dummy, c_dummy = CALIBRATION_CAP
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K
    elif temperature_calibration=='none':
        Cp1 = 0
    else:
        temp_dummy, c_dummy = temperature_calibration
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K

    # parasitic offset
    Cp2 = cap0_295K - C_0

    cap_offsets = 0
    dl_offsets = cap_to_dl(C_0 + Cp1 + Cp2)

    dl_corr = cap_to_dl(cap - cap_offsets) - dl_offsets

    return dl_corr

def dl_to_cap_corr_2(dl_corr, temp, cap0_295K, temperature_calibration='dummy', thermal_expansion='none', transmission_frac=1):
    '''
    correcting dL as displacement offsets

    args:
        - dl_corr:                          corrected displacement dl
        - temp:                             current temperature in K
        - cap0_295K:                        capacitance measured at 0 V and 295 K in pF
        - temperature_calibration:          (temp (K), capacitances (pF)) tuple or list with data from a Ti dummy calibration run over temperature range of interest

    returns:
        - cap:                              uncorrected capacitance
    '''

    # temperature correction
    if temperature_calibration=='dummy':
        temp_dummy, c_dummy = CALIBRATION_CAP
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K
    elif temperature_calibration=='none':
        Cp1 = 0
    else:
        temp_dummy, c_dummy = temperature_calibration
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K

    # parasitic offset
    Cp2 = cap0_295K - C_0

    cap_offsets = 0
    dl_offsets = cap_to_dl(C_0 + Cp1 + Cp2)

    dl_corr = cap_to_dl(cap - cap_offsets) - dl_offsets
    cap = dl_to_cap(dl_corr - dl_offsets) + cap_offsets

    return cap

def cap_to_dl_corr_3(cap, temp, cap0_295K, temperature_calibration='dummy'):
    '''
    correcting dL as combined displacement and capacitive offsets

    args:
        - cap:                              current capacitance in pF
        - temp:                             current temperature in K
        - cap0_295K:                        capacitance measured at 0 V and 295 K in pF
        - temperature_calibration:          (temp (K), capacitances (pF)) tuple or list with data from a Ti dummy calibration run over temperature range of interest

    returns:
        - dl_cor:                           corrected displacement dl
    '''

    # temperature correction
    if temperature_calibration=='dummy':
        temp_dummy, c_dummy = CALIBRATION_CAP
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K
    elif temperature_calibration=='none':
        Cp1 = 0
    else:
        temp_dummy, c_dummy = temperature_calibration
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K

    # parasitic offset
    Cp2 = cap0_295K - C_0

    cap_offsets = Cp2
    dl_offsets = cap_to_dl(C_0 + Cp1)

    dl_corr = cap_to_dl(cap - cap_offsets) - dl_offsets

    return dl_corr

def dl_to_cap_corr_3(dl_corr, temp, cap0_295K, temperature_calibration='dummy', thermal_expansion='none', transmission_frac=1):
    '''
    correcting dL as combined displacement and capacitive offsets

    args:
        - dl_corr:                          corrected displacement dl
        - temp:                             current temperature in K
        - cap0_295K:                        capacitance measured at 0 V and 295 K in pF
        - temperature_calibration:          (temp (K), capacitances (pF)) tuple or list with data from a Ti dummy calibration run over temperature range of interest

    returns:
        - cap:                              uncorrected capacitance
    '''

    # temperature correction
    if temperature_calibration=='dummy':
        temp_dummy, c_dummy = CALIBRATION_CAP
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K
    elif temperature_calibration=='none':
        Cp1 = 0
    else:
        temp_dummy, c_dummy = temperature_calibration
        c_dummy_temp = np.interp(temp, temp_dummy, c_dummy)
        c_dummy_295K = np.interp(295, temp_dummy, c_dummy)
        Cp1 = c_dummy_temp - c_dummy_295K

    # parasitic offset
    Cp2 = cap0_295K - C_0

    cap_offsets = Cp2
    dl_offsets = cap_to_dl(C_0 + Cp1)

    dl_corr = cap_to_dl(cap - cap_offsets) - dl_offsets
    cap = dl_to_cap(dl_corr - dl_offsets) + cap_offsets

    return cap

##  
##  methods for calculating strain
##

def dl_to_strain(dl, temp, l0, thermal_expansion='none', transmission_frac=1):
    '''
    correcting dL as capacitive offsets

    strain is calculated as

        strain = \eta (dL/L0 + differential_thermal_expansion)

    args:
        - dl:                               measured (corrected) displacement
        - temp:                             current temperature in K
        - l0:                               length of sample (gap) at 0 V and 295 K in m
        - thermal_expansion:                (temp (K), thermal_expansion (unitless)) tuple or list with thermal expansion data dL(T)/L(T)
        - transmission_frac:                \eta in the above equation, giving the fraction of dL that transmits to the sample

    returns:
        - strain:                            estimated strain in percent
    '''

    if thermal_expansion!='none':
        thermal_exp_temps = thermal_expansion[0]
        thermal_exp = thermal_expansion[1]
        thermal_exp_val = np.interp(temp, thermal_exp_temps, thermal_exp)
        Ti_thermal_exp_val = np.interp(temp, Ti_THERMAL_EXP_TEMPS, Ti_THERMAL_EXP)
        diff_therm_exp = Ti_thermal_exp_val - thermal_exp_val
        #l0 = (1+thermal_exp_val)*l0
    else:
        diff_therm_exp = 0

    strain = transmission_frac*(dl/l0 + diff_therm_exp)

    return strain

def strain_to_dl(strain, temp, l0, thermal_expansion='none', transmission_frac=1):
    '''
    correcting dL as capacitive offsets

    strain is calculated as

        strain = \eta (dL/L0 + differential_thermal_expansion)

    args:
        - strain                            strain (in percent)
        - temp:                             current temperature in K
        - l0:                               length of sample (gap) at 0 V and 295 K in m
        - thermal_expansion:                (temp (K), thermal_expansion (unitless)) tuple or list with thermal expansion data dL(T)/L(T)
        - transmission_frac:                \eta in the above equation, giving the fraction of dL that transmits to the sample

    returns:
        - dl:                                displacement
    '''

    if thermal_expansion!='none':
        thermal_exp_temps = thermal_expansion[0]
        thermal_exp = thermal_expansion[1]
        thermal_exp_val = np.interp(temp, thermal_exp_temps, thermal_exp)
        Ti_thermal_exp_val = np.interp(temp, Ti_THERMAL_EXP_TEMPS, Ti_THERMAL_EXP)
        diff_therm_exp = Ti_thermal_exp_val - thermal_exp_val
        #l0 = (1+thermal_exp_val)*l0
    else:
        diff_therm_exp = 0

    #strain = transmission_frac*(dl/l0 + diff_therm_exp)
    dl = l0*(strain/transmission_frac - diff_therm_exp)

    return dl


##
## Macros
##

def cap_to_strain_1(cap, temp, cap0_295K, l0, temperature_calibration='dummy', thermal_expansion='none', transmission_frac=1):
    dl_corr = cap_to_dl_corr_1(cap, temp, cap0_295K, temperature_calibration=temperature_calibration)
    strain = dl_to_strain(dl_corr, temp, l0, thermal_expansion=thermal_expansion, transmission_frac=1)
    return strain

def cap_to_strain_2(cap, temp, cap0_295K, l0, temperature_calibration='dummy', thermal_expansion='none', transmission_frac=1):
    dl_corr = cap_to_dl_corr_2(cap, temp, cap0_295K, temperature_calibration=temperature_calibration)
    strain = dl_to_strain(dl_corr, temp, l0, thermal_expansion=thermal_expansion, transmission_frac=1)
    return strain

def cap_to_strain_3(cap, temp, cap0_295K, l0, temperature_calibration='dummy', thermal_expansion='none', transmission_frac=1):
    dl_corr = cap_to_dl_corr_3(cap, temp, cap0_295K, temperature_calibration=temperature_calibration)
    strain = dl_to_strain(dl_corr, temp, l0, thermal_expansion=thermal_expansion, transmission_frac=1)
    return strain

def strain_to_cap_1(strain, temp, cap0_295K, l0, temperature_calibration='dummy', thermal_expansion='none', transmission_frac=1):
    dl_corr = strain_to_dl(strain, temp, l0, thermal_expansion=thermal_expansion, transmission_frac=transmission_frac)
    cap = cap_to_dl_corr_1(dl_corr, temp, l0, thermal_expansion=thermal_expansion, transmission_frac=1)
    return cap

def strain_to_cap_2(strain, temp, cap0_295K, l0, temperature_calibration='dummy', thermal_expansion='none', transmission_frac=1):
    dl_corr = strain_to_dl(strain, temp, l0, thermal_expansion=thermal_expansion, transmission_frac=transmission_frac)
    cap = cap_to_dl_corr_2(dl_corr, temp, l0, thermal_expansion=thermal_expansion, transmission_frac=1)
    return cap

def strain_to_cap_3(strain, temp, cap0_295K, l0, temperature_calibration='dummy', thermal_expansion='none', transmission_frac=1):
    dl_corr = strain_to_dl(strain, temp, l0, thermal_expansion=thermal_expansion, transmission_frac=transmission_frac)
    cap = cap_to_dl_corr_3(dl_corr, temp, l0, thermal_expansion=thermal_expansion, transmission_frac=1)
    return cap