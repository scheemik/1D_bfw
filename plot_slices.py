"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py NAME SWITCHBOARD <files>... [--output=<dir>]

Options:
    NAME            # Name to put in plot title
    SWITCHBOARD     # Name of switchboard file
    --output=<dir>  # Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ioff()

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parse input parameters
from docopt import docopt
args = docopt(__doc__)
name = args['NAME']
switchboard = args['SWITCHBOARD']
h5_files = args['<files>']

import switchboard as sbp
import helper_functions as hfuncts

import pathlib
output_path = pathlib.Path(args['--output']).absolute()

# Check if output path exists, if not, create it
import os
if not os.path.exists(output_path):
    if rank==0:
        os.makedirs(output_path)

# Labels
hori_label = r'$w$ (m/s)'
vert_label = r'$z$ (m)'

# Parameters
BP = sbp.BP_array
dpi = 100
tasks = ['w'] # usually 'b', 'p', 'u', or 'w'
rows = 1
cols = 2
n_x_ticks = 3
n_z_ticks = 7
round_to_decimal = 1
title_size = 'medium'
suptitle_size = 'large'
T = sbp.T

# number of oscillation periods to skip at the start
skip_nT = 0

###############################################################################
# Helper functions

# Make a plot for one time slice
def plot_task(ax, time_i, task_j, z_ax, dsetss):
    # plot line of w vs. z
    im = ax.plot(dsets[task_j][time_i][1], z_ax)
    # Find max of absolute value for data to make symmetric around zero
    xmax = max(abs(max(dsets[task_j][time_i][1].flatten())), abs(min(dsets[task_j][time_i][1].flatten())))
    if xmax==0.0:
        xmax = 0.001 # to avoid the weird jump with the first frame
    # format range of plot extent
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(z_ax[0], z_ax[-1])
    # format axis labels and ticks
    format_labels_and_ticks(ax)
    # add title
    ax.set_title(tasks[task_j], fontsize=title_size)

def format_labels_and_ticks(ax):
    # add labels
    ax.set_xlabel(hori_label)
    # fix horizontal ticks
    x0, xf = ax.get_xlim()
    ax.xaxis.set_ticks([x0, 0.0, xf])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(latex_exp))

# Saves figure as a frame
def save_fig_as_frame(fig, index, output, dpi):
    savename = 'write_{:06}.png'.format(index)
    savepath = output.joinpath(savename)
    fig.savefig(str(savepath), dpi=dpi)
    fig.clear()

# Takes an exponential number and returns a string formatted nicely for latex
#   Expects numbers in the format 7.0E+2
def latex_exp(num, pos=None):
    if (isinstance(num, int)):
        # integer type, don't reformat
        return num
    else:
        r_num = round(num, round_to_decimal)
        # print('num',num)
        # print('r_num',r_num)
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

# dsets will be an array containing all the data
#   it will have a size of: tasks x timesteps x 2 x nx x nz (5D)
dsets = []
for task in tasks:
    task_tseries = []
    for filename in h5_files:
        #print(filename)
        with h5py.File(filename, mode='r') as f:
            dset = f['tasks'][task]
            # Check dimensionality of data
            if len(dset.shape) != 2:
                raise ValueError("This only works for 2D datasets")
            # The [()] syntax returns all data from an h5 object
            task_grid = np.array(dset[()])
            z_scale = f['scales']['z']['1.0']
            z_axis = np.array(z_scale[()])
            t_scale = f['scales']['sim_time']
            t_axis = np.array(t_scale[()])
            for i in range(len(t_axis)):
                # Skip any times before specified number of T's
                if(t_axis[i] > skip_nT*T):
                    time_slice = [t_axis[i], np.transpose(task_grid[i])]
                    task_tseries.append(time_slice)
    dsets.append(task_tseries)

# Find length of time series
t_len = len(dsets[0])



# Iterate across time, plotting and saving a frame for each timestep
for i in range(t_len):
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,3]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, ax = plt.subplots(nrows=rows, ncols=cols, gridspec_kw=plot_ratios, sharey=True)
    # Plot background profile
    hfuncts.plot_BP(ax[0], BP, z_axis)
    # Plot the task
    plot_task(ax[1], i, 0, z_axis, dsets)
    # Add title for overall figure
    t = dsets[0][i][0]
    title_str = '{:}, $t/T=${:2.2f}'
    fig.suptitle(title_str.format(name, t/T), fontsize=suptitle_size)
    # Save figure as image in designated output directory
    save_fig_as_frame(fig, i, output_path, dpi)
    plt.close(fig)
