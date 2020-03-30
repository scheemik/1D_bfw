# Helper functions for Dedalus experiment
"""
Description:
This contains helper functions for the Dedalus code so the same version of functions can be called by multiple scripts
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import switchboard as sbp

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

# Plot background profile
def plot_BP(ax, BP, z):
    ax.plot(BP, z, label='Background profile')
    ax.set_xlabel('$N_0$')
    ax.set_ylabel(r'$z$')
    ax.set_title(r'Background Profile')

# fig, axes = plt.subplots(nrows=1, ncols=1)
# #
# z = np.linspace(0.2, -1.2, 1024)
# z0_dis = -1
# zf_dis = 0
# BP_array = BP_n_steps(1, z, z0_dis, zf_dis, 0.2)
# plot_BP(axes, BP_array, z)
# plt.show()
