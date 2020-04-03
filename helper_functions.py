# Helper functions for Dedalus experiment
"""
Description:
This contains helper functions for the Dedalus code so the same version of functions can be called by multiple scripts
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from dedalus.extras.plot_tools import quad_mesh, pad_limits

###############################################################################
# Takes an exponential number and returns a string formatted nicely for latex
#   Expects numbers in the format 7.0E+2
def latex_exp(num, pos=None):
    if (isinstance(num, int)):
        # integer type, don't reformat
        return num
    else:
        float_str = "{:.1E}".format(num)
        if "E" in float_str:
            base, exponent = float_str.split("E")
            exp = int(exponent)
            b   = float(base)
            str1 = '$'
            if (exp == -1):
                str1 = str1 + str(b/10.0)
            elif (exp == 0):
                str1 = str1 + str(base)
            elif (exp == 1):
                str1 = str1 + str(b*10.0)
            elif (exp == 2):
                str1 = str1 + str(b*100.0)
            else:
                str1 = str1 + str(base) + r'\cdot10^{' + str(exp) + '}'
            str1 = str1 + '$'
            return r"{0}".format(str1)
        else:
            return float_str

###############################################################################

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
    line_color = my_clrs['black']
    if z0_dis != None:
        ax.axhline(y=z0_dis, color=line_color, linestyle='--')
        ax.axhline(y=zf_dis, color=line_color, linestyle='--')

# Plot background profile
def plot_BP(ax, BP, z, omega=None, z0_dis=None, zf_dis=None):
    ax.plot(BP, z, color=my_clrs['N_0'], label=r'$N_0$')
    ax.set_xlabel(r'$N_0$ (s$^{-1}$)')
    ax.set_ylabel(r'$z$ (m)')
    ax.set_title(r'Background Profile')
    ax.set_ylim([min(z),max(z)])
    if omega != None:
        ax.axvline(x=omega, color=my_clrs['tab:gray'], linestyle='--', label=r'$\omega$')
        ax.legend()

def plot_v_profiles(BP_array, bf_array, sp_array, z, omega=None, z0_dis=None, zf_dis=None):
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,4]}
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

###############################################################################

def plot_z_vs_t(z, t_array, T, w_array, BP_array, k, m, omega, z0_dis=None, zf_dis=None, c_map='RdBu_r'):
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(0.5)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,5]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(figsize=(w,h), nrows=1, ncols=2, gridspec_kw=plot_ratios, sharey=True)
    #
    plot_BP(axes[0], BP_array, z, omega)
    add_dis_bounds(axes[0], z0_dis, zf_dis)
    #
    xmesh, ymesh = quad_mesh(x=t_array/T, y=z)
    im = axes[1].pcolormesh(xmesh, ymesh, w_array, cmap=c_map)
    # Find max of absolute value for colorbar for limits symmetric around zero
    cmax = max(abs(w_array.flatten()))
    if cmax==0.0:
        cmax = 0.001 # to avoid the weird jump with the first frame
    # Set upper and lower limits on colorbar
    im.set_clim(-cmax, cmax)
    # Add colorbar to im
    cbar = plt.colorbar(im)#, format=ticker.FuncFormatter(latex_exp))
    cbar.ax.ticklabel_format(style='sci', scilimits=(-2,2), useMathText=True)
    axes[1].set_xlabel(r'$t/T$')
    axes[1].set_title(r'$w$ (m/s)')
    param_formated_str = latex_exp(k)+', '+latex_exp(m)+', '+latex_exp(omega)
    fig.suptitle(r'Forced 1D Wave, $(k,m,\omega)$=(%s)' %(param_formated_str))
    plt.savefig('f_1D_wave.png')

###############################################################################

# Make a plot for one time slice
def plot_task(ax, time_i, task_j, z_ax, dsets):
    # plot line of w vs. z
    im = ax.plot(dsets[task_j][time_i][1], z_ax, color=my_clrs['v_w'])
    # Find max of absolute value for data to make symmetric around zero
    xmax = max(abs(max(dsets[task_j][time_i][1].flatten())), abs(min(dsets[task_j][time_i][1].flatten())))
    if xmax==0.0:
        xmax = 0.001 # to avoid the weird jump with the first frame
    # format range of plot extent
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(z_ax[0], z_ax[-1])

def format_labels_and_ticks(ax, hori_label):
    # add labels
    ax.set_xlabel(hori_label)
    # fix horizontal ticks
    x0, xf = ax.get_xlim()
    ax.xaxis.set_ticks([x0, 0.0, xf])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(latex_exp))

###############################################################################
# Plotting colors from style guide

my_clrs       = {'diff'  : (0, 0.5, 0),         # g
                 'visc': '#2ca02c',             # tab:green
                 'buoy': (0, 0, 1),             # b
                 'N_0': '#1f77b4',              # tab:blue
                 'v_w': (1, 0, 0),              # r
                 'v_u': '#e377c2',              # tab:pink
                 'advec': '#d62728',            # tab:red
                 'p': '#ff7f0e',                # tab:orange
                 'bf': '#17becf',               # tab:cyan
                 'sp': '#bcbd22',               # tab:olive
                 'tab:brown': '#8c564b',
                 'tab:gray': '#7f7f7f',
                 'tab:purple': '#9467bd',
                 'c': (0, 0.75, 0.75),
                 'm': (0.75, 0, 0.75),
                 'y': (0.75, 0.75, 0),
                 'black': (0, 0, 0),
                 'white': (1, 1, 1)}
