# Plotting the
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

x_min = 0
x_max = np.pi/2
nx = 1000
y_min = 0
y_max = 5
ny = 1000

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)

theta, mL = np.meshgrid(x, y)
# Sutherland and Yewchuk 2004 eq 2.4
def T_for_1_layer(theta, mL):
    return 1 / (1 + (np.sinh(mL)/np.sin(2*theta))**2)
Tcoeff = T_for_1_layer(theta, mL)

fig, axes = plt.subplots(nrows=1, ncols=1)
im = axes.pcolormesh(theta, mL, Tcoeff, cmap='jet')
cbar = plt.colorbar(im)#, format=ticker.FuncFormatter(latex_exp))
axes.set_xlabel(r'$\theta_1$')
axes.set_ylabel(r'$m_0L$')
axes.set_title(r'$|\mathcal{T}|/|\mathcal{I}|$ for 1 layer')

plt.savefig('Tcoeff_Ghaemsaidi_fig3a.png')
