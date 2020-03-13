"""
A script for visualizing by boundary forcing functions

"""

# !pip3 install --upgrade
# !pip3 install sympy
# !pip3 install matplotlib

# %%
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import *

###############################################################################
# %%
# Boundary forcing parameters
Lx = 1.5 #m
x_min = -Lx/3.0
x_max =  2*Lx/3.0
# Bounds of the forcing window
fl_edge = -1.0*Lx/12.0
fr_edge = 0.0 #-1.0*Lx/12.0
# Angle of beam w.r.t. the horizontal
theta = np.pi/4
# Horizontal wavelength
lam_x = fr_edge - fl_edge
# Horizontal wavenumber
kx    = 2.0*np.pi/lam_x
# Vertical wavenumber = 2*pi/lam_z, or from trig:
kz    = kx * np.tan(theta)
# Other parameters
A     = 3.0e-4
N_0 = 1
omega = N_0 * np.cos(theta) # [s^-1], from dispersion relation
period = 2*np.pi/omega

fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(x_min, x_max), ylim=(-1, 1), xlabel='x (m)', ylabel='Amplitude')
fig.tight_layout()
x = np.linspace(x_min, x_max, 91)
t = np.linspace(0, period, 30)
X2, T2 = np.meshgrid(x, t)

# Windowing function (multiplying tanh's)
slope = 40
left_side = 0.5*(np.tanh(slope*(X2-fl_edge))+1)
right_side = 0.5*(np.tanh(slope*(-X2+fr_edge))+1)
win = left_side*right_side

# Pick arbirary z
z = 0

# Boundary forcing for u
Fu = -np.sin(kx*X2 + kz*z - omega*T2)*win
# Boundary forcing for w
Fw =  np.sin(kx*X2 + kz*z - omega*T2)*win
# Boundary forcing for b
Fb = -np.cos(kx*X2 + kz*z - omega*T2)*win

line0 = ax.plot(x, win[0, :], '--', lw=1)[0]
line1 = ax.plot(x, Fu[0, :], color='r', lw=2)[0]
line2 = ax.plot(x, Fw[0, :], color='k', lw=2)[0]
line3 = ax.plot(x, Fb[0, :], color='b', lw=2)[0]

def animate(i):
    line0.set_ydata(win[i, :])
    line1.set_ydata(Fu[i, :])
    line2.set_ydata(Fw[i, :])
    line3.set_ydata(Fb[i, :])

anim = FuncAnimation(
    fig, animate, interval=period, frames=len(t)-1)

name = '2D_CD_test'
#filename = '_boundary_forcing/' + name + '.mp4'
#anim.save(filename)
filename = name + '.gif'
anim.save(filename, dpi=600, writer='imagemagik')

#plt.draw()
#plt.show()

# %%
