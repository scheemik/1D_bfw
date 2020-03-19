"""
1D Forced wave equation:

d_{tt}w + a*d_{zz}w = F(z, t)

This script should be ran serially (because it is 1D), and creates a space-time
plot of the computed solution.

"""

import numpy as np
import matplotlib.pyplot as plt
import time

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from dedalus.core.operators import GeneralFunction

import logging
logger = logging.getLogger(__name__)

###############################################################################

# Domain parameters
nz = 1024
z0, zf = -2, 0
Lz = zf - z0

# Problem parameters
a=1.
lambda_z = 1.0
m = 2*np.pi / lambda_z
T = 9.0
omega = 2*np.pi / T

# Run parameters
sim_time_stop = 3*T
dt = 0.125 #2e-2
adapt_dt = False
snap_dt = 3*dt
snap_max_writes = 100
# Output
fh_mode = 'overwrite' # or 'append'

###############################################################################

# Bases and domain
z_basis = de.Fourier('z', nz, interval=(z0, zf), dealias=3/2)
# something weird happens when I use dealias, the number of grid points in x don't line up anymore. I put in 1024 but get back 1536
domain = de.Domain([z_basis], np.float64)

#Z grid
z_da = domain.grid(0, scales=domain.dealias)
z = domain.grid(0)

###############################################################################
# Boundary forcing

# amplitude
a_bf = 1.0
# Centered around
z_cbf = -Lz/16
# Full width at half max
b_bf = Lz/32
win_bf = a_bf*np.exp(-4*np.log(2)*((z - z_cbf)/b_bf)**2)

# Sponge

# amplitude
a_sp = 1.0
# Centered around
z_csp = -15*Lz/16
# Full width at half max
b_sp = Lz/32
win_sp = a_sp*np.exp(-4*np.log(2)*((z - z_csp)/b_sp)**2)

###############################################################################

# plot boundary forcing window
plt.figure()
plt.plot(win_bf, z, label='bf')
plt.plot(win_sp, z, label='sp')
plt.xlabel('amplitude')
plt.ylabel('z')
plt.legend()
plt.title(r'Windowing functions')
plt.savefig('f_1D_windows.png')
