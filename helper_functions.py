# Helper functions for Dedalus experiment
"""
Description:
This contains helper functions for the Dedalus code so the same version of functions can be called by multiple scripts
"""

import numpy as np
import matplotlib.pyplot as plt
from dedalus.extras.plot_tools import quad_mesh, pad_limits

# Background profile in N_0
def BP_n_steps(n, z, z0_dis, zf_dis, th):
    """
    n           number of steps
    z           array of z values
    z0_dis      bottom of display domain
    zf_dis      top of display domain
    th          step thickness
    """
    # create blank array the same size as z
    BP_array = z*0+1
    # divide the display range for n steps
    Lz_dis = zf_dis - z0_dis
    # find step separation
    step_sep = Lz_dis / (n+1)
    for i in range(n):
        step_c   = zf_dis - (i+1)*step_sep
        step_top = step_c + (th/2)
        step_bot = step_c - (th/2)
        for j in range(len(BP_array)):
            if z[j] < step_top and z[j] > step_bot:
                BP_array[j] = 0
    return BP_array

def add_dis_bounds(ax, z0_dis=None, zf_dis=None):
    line_color = my_clrs['k']
    if z0_dis != None:
        ax.axhline(y=z0_dis, color=line_color, linestyle='--')
        ax.axhline(y=zf_dis, color=line_color, linestyle='--')

# Plot background profile
def plot_BP(ax, BP, z, omega=None, z0_dis=None, zf_dis=None):
    ax.plot(BP, z, color=my_clrs['N_0'], label=r'$N_0$')
    ax.set_xlabel('$N_0$')
    ax.set_ylabel(r'$z$')
    ax.set_title(r'Background Profile')
    ax.set_ylim([min(z),max(z)])
    if omega != None:
        ax.axvline(x=omega, color=my_clrs['tab:gray'], linestyle='--', label=r'$\omega$')
        ax.legend()

def plot_v_profiles(BP_array, bf_array, sp_array, z, omega=None, z0_dis=None, zf_dis=None):
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,3]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw=plot_ratios, sharey=True)
    #
    plot_BP(axes[0], BP_array, z, omega)
    add_dis_bounds(axes[0], z0_dis, zf_dis)
    #
    axes[1].plot(bf_array, z, color=my_clrs['bf'], label='Boundary forcing')
    axes[1].plot(sp_array, z, color=my_clrs['sp'], label='Sponge layer')
    add_dis_bounds(axes[1], z0_dis, zf_dis)
    axes[1].set_xlabel('Amplitude')
    #axes[1].set_ylabel(r'$z$')
    axes[1].set_title(r'Windows')
    axes[1].legend()
    #
    plt.savefig('f_1D_windows.png')

def plot_z_vs_t(z, t_array, w_array, k, m, omega, c_map='RdBu_r'):
    xmesh, ymesh = quad_mesh(x=t_array, y=z)
    plt.figure()
    im = plt.pcolormesh(xmesh, ymesh, w_array, cmap=c_map)
    plt.axis(pad_limits(xmesh, ymesh))
    # Find max of absolute value for colorbar for limits symmetric around zero
    cmax = max(abs(w_array.flatten()))
    if cmax==0.0:
        cmax = 0.001 # to avoid the weird jump with the first frame
    # Set upper and lower limits on colorbar
    im.set_clim(-cmax, cmax)
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('z')
    plt.title(r'Forced 1D Wave, $(k,m,\omega)$=(%g,%g,%g)' %(k, m, omega))
    plt.savefig('f_1D_wave.png')

###############################################################################
# Plotting colors from style guide

my_clrs       = {'diff'  : (0, 0.5, 0),         # g
                 'visc': '#2ca02c',             # tab:green
                 'N_0': (0, 0, 1),              # b
                 'buoy': '#1f77b4',             # tab:blue
                 'advec': '#d62728',            # tab:red
                 'press': '#9467bd',            # tab:purple
                 'bf': '#17becf',               # tab:cyan
                 'sp': '#ff7f0e',               # tab:orange
                 'tab:brown': '#8c564b',
                 'tab:pink': '#e377c2',
                 'tab:gray': '#7f7f7f',
                 'tab:olive': '#bcbd22',
                 'r': (1, 0, 0),
                 'c': (0, 0.75, 0.75),
                 'm': (0.75, 0, 0.75),
                 'y': (0.75, 0.75, 0),
                 'k': (0, 0, 0),
                 'w': (1, 1, 1)}
