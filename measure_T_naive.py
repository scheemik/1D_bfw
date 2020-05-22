"""

This script measures the transmissino coefficient (T) in a very naive way.
I measure the incoming wave I' by taking the oscillation periods between when it crosses into the display domain and when the reflection comes back up to that point, then taking the root mean squared amplitude over that period.
I measure the transmitted wave T' in a similar way, but looking the depth just below the step and the time between when the wave first transmits and the end of the simulation.
The transmission coefficient is then calculated as T = T' / I'

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from bisect import bisect_left

###############################################################################
# Checking command line arguments
import sys
# Arguments must be passed in the correct order
arg_array = sys.argv
# argv[0] is the name of this file
run_name = str(arg_array[1])
if run_name == None:
    run_name = 'test run'
switchboard = str(arg_array[2])
if switchboard == None:
    switchboard = 'switchboard'

# Add functions in helper file
import helper_functions as hf

###############################################################################
# Import SwitchBoard Parameters (sbp)
#   This import assumes the switchboard is in the same directory as the core code
import switchboard as sbp
# Physical parameters
nu          = sbp.nu            # [m^2/s] Viscosity (momentum diffusivity)
#kappa       = sbp.kappa         # [m^2/s] Thermal diffusivity
f_0         = sbp.f_0           # [s^-1]        Reference Coriolis parameter
g           = sbp.g             # [m/s^2] Acceleration due to gravity
# Problem parameters
N_0         = sbp.N_0           # [rad/s]       Reference stratification
lam_x       = sbp.lam_x         # [m]           Horizontal wavelength
lam_z       = sbp.lam_z         # [m]           Vertical wavelength
k           = sbp.k             # [m^-1]        Horizontal wavenumber
m           = sbp.m             # [m^-1]        Vertical wavenumber
k_total     = sbp.k_total       # [m^-1]        Total wavenumber
theta       = sbp.theta         # [rad]         Propagation angle from vertical
omega       = sbp.omega         # [rad s^-1]    Wave frequency
T           = sbp.T             # [s]           Wave period
# Domain parameters
z0_dis      = sbp.z0_dis        # [m]           The bottom of the displayed domain
zf_dis      = sbp.zf_dis        # [m]           The top of the displayed z domain
step_th     = sbp.step_th       # [m]           The thickness of the layer
z_I         = zf_dis - (2/m)    # [m]           Depth at which I' will be measured
z_T = (z0_dis-zf_dis-step_th)/2 - (2/m) # [m]           Depth at which T' will be measured

###############################################################################
# Get depth and wavenumber axes
z           = sbp.z
ks          = sbp.ks

###############################################################################

def take_closest(myList, myNumber, returns='index'):
    """
    See: https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    Assumes myList is sorted.
    If returns='value', returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    If returns='index', returns indext of closest value to myNumber
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        if returns=='index':
            return pos
        elif returns=='value':
            return after
    else:
        if returns=='index':
            return pos-1
        elif returns=='value':
            return before

def plot_I_and_T(z, i_I, i_T, t_array, data_array, k, m, omega, title_str='Forced 1D Wave'):
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(0.5)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1,1],
                   'width_ratios': [1]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(figsize=(w,h), nrows=2, ncols=1, gridspec_kw=plot_ratios, sharey=True)
    # plot lines of psi vs. t
    axes[0].plot(t_array, data_array[i_I])
    hf.format_labels_and_ticks_v(axes[0], r'$\Psi^2$')
    axes[1].plot(t_array, data_array[i_T])
    hf.format_labels_and_ticks_v(axes[1], r'$\Psi^2$')
    #
    axes[0].set_xticklabels([])
    axes[1].set_xlabel(r'$t/T$')
    axes[0].set_title(r'$\mathcal{I}$ at $z=$'+str(z[i_I]))
    axes[1].set_title(r'$\mathcal{T}$ at $z=$'+str(z[i_T]))
    param_formated_str = hf.latex_exp(k)+', '+hf.latex_exp(m)+', '+hf.latex_exp(omega)
    fig.suptitle(r'%s, $(k,m,\omega)$=(%s)' %(title_str, param_formated_str))
    plt.savefig('f_1D_I_and_T.png')

def calc_I_and_T(i_I, i_T, t_array, data_array, t_interval_I, t_interval_T):
    # This returns the root mean square values of I and T
    #   I'm using arbitrary time intervals, i.e. t_interval_I=(I_t_i, I_t_f)
    #
    # Square of data at the two choosen depths
    I_slice = np.square(data_array[i_I])
    T_slice = np.square(data_array[i_T])
    # Find indicies of choosen time intervals
    ti_I_i = take_closest(t_array, t_interval_I[0])
    ti_I_f = take_closest(t_array, t_interval_I[1])
    ti_T_i = take_closest(t_array, t_interval_T[0])
    ti_T_f = take_closest(t_array, t_interval_T[1])
    # Filter to choosen times
    I_slice = I_slice[ti_I_i:ti_I_f]
    T_slice = T_slice[ti_T_i:ti_T_f]
    # Mean and square root
    I = np.sqrt(np.mean(I_slice))
    T = np.sqrt(np.mean(T_slice))
    return I, T

###############################################################################
# Save arrays to files
arrays = {'psi_g_array':[],
          'psi_c_reals':[],
          'psi_c_imags':[],
          't_array':[],
          'BP_array':[]}
for arr in arrays:
    file = open('arrays/'+arr, "rb")
    arrays[arr] = np.load(file)
    file.close

# Find indices of closest values of z to z_I and z_T
i_I = take_closest(z, z_I)
i_T = take_closest(z, z_T)

plot_I_and_T(z, i_I, i_T, arrays['t_array']/T, np.square(arrays['psi_g_array']), k, m, omega)

I, T = calc_I_and_T(i_I, i_T, arrays['t_array']/T, arrays['psi_g_array'], (7.0,10.0), (12.0,15.0))

print("I' = ",I)
print("T' = ",T)
print("T'/I' = ",T/I)
