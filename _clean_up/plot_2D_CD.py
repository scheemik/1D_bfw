"""
Performs and plots complex demodulation in 2 spatial dimensions
Based on Mercier et al. 2008


Written by Mikhail Schee
Jan 2020
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
# from docopt import docopt
# args = docopt(__doc__)
# name = args['NAME']
# h5_files = args['<files>']
# import pathlib
# output_path = pathlib.Path(args['--output']).absolute()

# Check if output path exists, if not, create it
# import os
# if not os.path.exists(output_path):
#     if rank==0:
#         os.makedirs(output_path)

# Labels
hori_label = r'$x$ (m)'
vert_label = r'$z$ (m)'

# Parameters
dpi = 100
# tasks = ['w'] # usually 'b', 'p', 'u', or 'w'
# rows = 1
# cols = 2
cmap = 'RdBu_r'
n_ticks = 5
n_cb_ticks = 3
round_to_decimal = 1
title_size = 'medium'
suptitle_size = 'large'

###############################################################################
# Helper functions

# Make a plot for one task
def plot_task(fig, axes, rows, cols, time_i, task_j, x_ax, z_ax, dsets, cmap, AR):
    # get coordinates of desired axis, avoiding dumb indexing with if statements
    if rows==1 or cols==1:
        if rows==1 and cols==1:
            ax = axes
        else:
            ax = axes[task_j]
    else:
        ax = axes[task_j//cols, task_j%cols]
    # plot task colormesh
    im = ax.pcolormesh(x_ax, z_ax, dsets[task_j][time_i][1], cmap=cmap)
    # format axis labels and ticks
    format_labels_and_ticks(ax)
    # Find max of absolute value for colorbar for limits symmetric around zero
    cmax = max(abs(max(dsets[task_j][time_i][1].flatten())), abs(min(dsets[task_j][time_i][1].flatten())))
    if cmax==0.0:
        cmax = 0.001 # to avoid the weird jump with the first frame
    # format colorbar
    cax = format_colorbar(im, ax, cmax)
    # Set aspect ratio for plot
    ax.set_aspect(AR)
    # add title
    cax.set_title(tasks[task_j], fontsize=title_size)

# Make a plot for one task
def make_one_subplot(fig, ax, x_ax, z_ax, data, title, cmap, AR):
    # plot task colormesh
    im = ax.pcolormesh(x_ax, z_ax, data, cmap=cmap)
    # format axis labels and ticks
    format_labels_and_ticks(ax)
    # Find max of absolute value for colorbar for limits symmetric around zero
    cmax = max(abs(max(data.flatten())), abs(min(data.flatten())))
    if cmax==0.0:
        cmax = 0.001 # to avoid the weird jump with the first frame
    # format colorbar
    cax = format_colorbar(im, fig, ax, cmax)
    # Set aspect ratio for plot
    ax.set_aspect(AR)
    # add title
    cax.set_title(title, fontsize=title_size)

def format_colorbar(image, fig, axis, cmax):
    # this is the only way I found to put the colorbar on top of the plot
    divider = make_axes_locatable(axis)
    caxis = divider.append_axes('top', size='5%', pad=0.03)
    # Set upper and lower limits on colorbar
    image.set_clim(-cmax, cmax)
    # Create colorbar
    cbar = fig.colorbar(image, cax=caxis, orientation='horizontal',  format=ticker.FuncFormatter(latex_exp), ticks=[-cmax, 0.0, cmax])
    caxis.xaxis.set_ticks_position('top')
    # Return color axis for placing plot title
    return caxis

def format_labels_and_ticks(ax):
    # add labels
    ax.set_xlabel(hori_label)
    ax.set_ylabel(vert_label)
    # fix vertical and horizontal ticks
    x0, xf = ax.get_xlim()
    x0 = round(x0, round_to_decimal)
    xf = round(xf, round_to_decimal)
    ax.xaxis.set_ticks(np.linspace(x0, xf, n_ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(latex_exp))
    z0, zf = ax.get_ylim()
    z0 = round(z0, round_to_decimal)
    zf = round(zf, round_to_decimal)
    ax.yaxis.set_ticks(np.linspace(z0, zf, n_ticks))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(latex_exp))

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

# Main plotting function - called by other script
def plot_frames(plt_data, times, T, x_axis, z_axis, name, output_path):
    # Find length of time series
    t_len = len(times)
    print('t_len = ', t_len)
    # Find number of subplots
    n_plots = int(len(plt_data))

    # Calculate aspect ratio of plot based on extent
    extent_aspect = abs((x_axis[-1]-x_axis[0])/(z_axis[-1]-z_axis[0]))
    aspect_ratio = 1
    AR = extent_aspect/aspect_ratio

    # Select number of rows and columns - kinda hacky
    rows = (n_plots-1)//3 + 1
    if rows == 1:
        cols = (n_plots-1)%3 + 1
    else:
        cols = (n_plots + 1)//rows

    print('rows = ',rows)
    print('cols = ',cols)

    # Iterate across time, plotting and saving a frame for each timestep
    for i in range(t_len):
        fig, axes = plt.subplots(nrows=rows, ncols=cols)
        # Set aspect ratio for figure
        size_factor = 4.0
        w, h = cols*size_factor, (rows+0.4)*size_factor
        plt.gcf().set_size_inches(w, h)
        # Plot each task
        for j in range(n_plots):
            if rows==1 or cols==1:
                if rows==1 and cols==1:
                    ax = axes
                else:
                    ax = axes[j]
            else:
                ax = axes[j//cols, j%cols]
            make_one_subplot(fig, ax, x_axis, z_axis, plt_data[j]['data'][i], plt_data[j]['name'], cmap, AR)
        # Add title for overall figure
        t = times[i]
        current_T = t/T
        title_str = '{:}, $t/T=${:2.2f}'
        fig.suptitle(title_str.format(name, current_T), fontsize=suptitle_size)
        # this (mostly) prevents axis labels from overlapping
        fig.tight_layout()
        # Save figure as image in designated output directory
        save_fig_as_frame(fig, i, output_path, dpi)
        plt.close(fig)
