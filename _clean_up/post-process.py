"""
Performs post-processing on snapshots from a simulation
Based on Mercier et al. 2008

To run:
mpiexec -n 2 python3 post-process.py NAME snapshots/*.h5

Usage:
    post-process.py NAME <files>... [--output=<dir>]

Options:
    NAME            # Name to put in plot title
    --output=<dir>  # Output directory [default: ./frames]

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
from docopt import docopt
args = docopt(__doc__)
name = args['NAME']
h5_files = args['<files>']
import pathlib
output_path = pathlib.Path(args['--output']).absolute()

# Check if output path exists, if not, create it
import os
if not os.path.exists(output_path):
    if rank==0:
        os.makedirs(output_path)

# Import parameters and modules
#   This import assumes these files are in the same directory as this code
import plot_2D_CD as p2DCD

# Parameters
tasks = ['b', 'p', 'u', 'w']
T = 8.885765876316732
T_start = 0
T_stop  = 3
dt = 0.125

###############################################################################
# fourier transform in time, filter negative freq's, inverse fourier transform
def FT_in_time(t, x, z, data, dt):
    # FT in time of the data (axis 0 is time)
    ftd = np.fft.fft(data, axis=0)
    # find relevant frequencies
    freq = np.fft.fftfreq(len(t), dt)
    f_grid, x_grid, z_grid = np.meshgrid(freq, x, z, indexing='ij')
    # Create mask to keep positive frequencies - add 1 to sign of f_grid
    f_mask = np.sign(f_grid) + 1.0
    #  -1 becomes 0, negative frequencies are masked out
    #   0 becomes 1, no change to frequencies of 0
    #   1 becomes 2, double positive frequencies to correct for lost amplitude
    ftd = ftd * f_mask
    # inverse fourier transform in time of the data
    iftd = np.fft.ifft(ftd, axis=0)
    #   a complex valued signal where iftd.real == data, or close enough
    return iftd

###############################################################################
# Gather the data

# dsets will be an array containing all the data
#   it will have a size of: tasks x timesteps x nx x nz (5D)
#   where timesteps is the number of time slices between T start and stop
dsets = []
t_s   = []
for task in tasks:
    task_tseries = []
    task_ts = []
    for filename in h5_files:
        #print(filename)
        with h5py.File(filename, mode='r') as f:
            dset = f['tasks'][task]
            # Check dimensionality of data
            if len(dset.shape) != 3:
                raise ValueError("This only works for 3D datasets")
            # The [()] syntax returns all data from an h5 object
            task_grid = np.array(dset[()])
            x_scale = f['scales']['x']['1.0']
            x_axis = np.array(x_scale[()])
            z_scale = f['scales']['z']['1.0']
            z_axis = np.array(z_scale[()])
            t_scale = f['scales']['sim_time']
            t_axis = np.array(t_scale[()])
            for i in range(len(t_axis)):
                time = t_axis[i]
                period = time/T
                if period > T_start and period < T_stop:
                    task_ts.append(time)
                    time_slice = np.transpose(task_grid[i])
                    task_tseries.append(time_slice)
    dsets.append(np.array(task_tseries))
    t_s.append(np.array(task_ts))

print(dsets[0].shape)
t = t_s[0]
x = x_axis
z = z_axis

## Step 1
# print('taking FT in time')
# ft_t_data = FT_in_time(t, x, z, dsets[3], dt)

# Create array of dictionaries for items to be plotted
plot_data = [
            {'data':   dsets[0],
            'name':    tasks[0]},

            {'data':   dsets[1],
             'name':   tasks[1]},

            {'data':   dsets[2],
             'name':   tasks[2]},

            {'data':   dsets[3],
             'name':   tasks[3]'}
            ]

p2DCD.plot_frames(plot_data, t, T, x_axis, z_axis, name, output_path)
