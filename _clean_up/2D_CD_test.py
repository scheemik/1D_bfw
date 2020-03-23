"""
Testing out complex demodulation in 2 spatial dimensions
Based on Mercier et al. 2008

Written by Mikhail Schee
Jan 2020
"""

import numpy as np
#import math
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
#from sympy import *
import sys

def plot_t_slice(hori, vert, data, fig, ax, h_label, v_label, title, t_font, b_font):
    im = ax.pcolormesh(hori, vert, data, cmap='RdBu_r', shading='gouraud')
    ax.set_xlabel(h_label, **b_font)
    ax.set_ylabel(v_label, **b_font)
    ax.set_title(title, **t_font)
    fig.colorbar(im, ax=ax)
    return im

# Saves figure as a frame
def save_fig_as_frame(fig, index, output, dpi):
    savename = 'write_{:06}.png'.format(index)
    savepath = output + '/' + savename
    fig.savefig(str(savepath), dpi=dpi)
    fig.clear()

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

# fourier transform in spatial dimensions
#   similar to FT in time, but switch dimensions around
def FT_in_space(t, x, z, data, dx, dz):
    # FT in space (x) of the data (axis 1 is x) for positive k_x
    AB = np.fft.fft(data, axis=1)
    # make a copy for the negative k_x
    CD = AB.copy()
    # find relevant wavenumbers
    k_xs = np.fft.fftfreq(len(x), dx)
    t_grid, kx_grid, z_grid = np.meshgrid(t, k_xs, z, indexing='ij')
    # Create mask to keep positive k_xs - relies on integer arithmetic
    kx_p_mask = np.ceil((np.sign(kx_grid) + 1.0)/2)
    #  -1 becomes 0, negative kxs are masked out
    #   0 becomes 1, no change to kxs of 0 - masked out
    #   1 becomes 1, no change to positive kxs
    CD = CD * kx_p_mask
    # Create mask to keep negative k_xs - relies on integer arithmetic
    kx_n_mask = np.abs(np.ceil((np.sign(kx_grid) - 1.0)/2))
    #  -1 becomes 1, no change to negative kxs
    #   0 becomes 1, no change to kxs of 0
    #   1 becomes 0, positive kxs are masked out
    AB = AB * kx_n_mask
    # inverse fourier transform in space (x)
    kx_p = np.fft.ifft(AB, axis=1)
    kx_n = np.fft.ifft(CD, axis=1)
    ##
    # FT in space (z) of the data (axis 2 is z) for positive k_z
    A = np.fft.fft(kx_p, axis=2)
    C = np.fft.fft(kx_n, axis=2)
    # make copies for the negative k_x
    B = A.copy()
    D = C.copy()
    # find relevant wavenumbers
    k_zs = np.fft.fftfreq(len(z), dz)
    t_grid, x_grid, kz_grid = np.meshgrid(t, x, k_zs, indexing='ij')
    # Create mask to keep positive k_zs - relies on integer arithmetic
    kz_p_mask = np.ceil((np.sign(kz_grid) + 1.0)/2)
    #  -1 becomes 0, negative kzs are masked out
    #   0 becomes 1, no change to kzs of 0 - masked out
    #   1 becomes 1, no change to positive kzs
    B = B * kz_p_mask
    D = D * kz_p_mask
    # Create mask to keep negative k_zs - relies on integer arithmetic
    kz_n_mask = np.abs(np.ceil((np.sign(kz_grid) - 1.0)/2))
    #  -1 becomes 1, no change to negative kzs
    #   0 becomes 1, no change to kzs of 0
    #   1 becomes 0, positive kzs are masked out
    A = A * kz_n_mask
    C = C * kz_n_mask
    # inverse fourier transform in space (z)
    A_xp_zp = np.fft.ifft(A, axis=2)
    B_xp_zn = np.fft.ifft(B, axis=2)
    C_xn_zp = np.fft.ifft(C, axis=2)
    D_xn_zn = np.fft.ifft(D, axis=2)
    return A_xp_zp, B_xp_zn, C_xn_zp, D_xn_zn

###############################################################################
# %%

output_path = '2D_CD_frames'
dpi = 100
name = ''

# Parameters
omega = 2.0
kx    = 2.0
kz    = 2.0
A = 1.0
B = A * 1
C = A * 0
D = A * 0

# number of oscillation periods
n_o = 4
t0 = 0.0
tf = (2*np.pi) / omega * n_o
nt = 128
dt = (tf-t0)/nt

# number of horizontal wavelengths
n_lambda_x = 2
x0 = 0.0
xf = 1.0
xf = (2*np.pi) / kx * n_lambda_x
nx = 128
dx = (xf-x0)/nx

# number of vertical wavelengths
n_lambda_z = 3
z0 = 0.0
zf = 1.0
zf = (2*np.pi) / kz * n_lambda_z
nz = 128
dz = (zf-z0)/nz

t = np.linspace(t0, tf, nt)
x = np.linspace(x0, xf, nx)
z = np.linspace(z0, zf, nz)
# using indexing='ij' insures the input order is the output order for dimensions
tm, xm, zm = np.meshgrid(t, x, z, indexing='ij')
# individual fields, following Merier et al. 2008 eq 10-13
Af = A*np.exp(1j*(omega*tm - kx*xm - kz*zm))
Bf = B*np.exp(1j*(omega*tm - kx*xm + kz*zm))
Cf = C*np.exp(1j*(omega*tm + kx*xm - kz*zm))
Df = D*np.exp(1j*(omega*tm + kx*xm + kz*zm))
# up and down fields
up = Af.real + Cf.real
dn = Bf.real + Df.real
# total field
y = up + dn

## Step 1
print('taking FT in time')
ift_t_y = FT_in_time(t, x, z, y, dt)

## Step 2
print('taking FT in space')
A_com, B_com, C_com, D_com = FT_in_space(t, x, z, ift_t_y, dx, dz)

# Take real parts of components
A_cd = A_com.real
B_cd = B_com.real
C_cd = C_com.real
D_cd = D_com.real

# Get up and down CD fields
up_cd = A_cd + C_cd
dn_cd = B_cd + D_cd
# Get total CD field
y_cd = up_cd + dn_cd


###############################################################################
# sys.exit('stopping exeuction')
plot_frames = True

title_font = {'fontname':'RobotoBold'}
body_font = {'fontname':'EBGaramond08'}

if plot_frames == True:
    n_frames = int(len(t)/n_o)
    print('Now plotting ' + str(n_frames) + ' frames')
    # Iterate across time, plotting and saving a frame for each timestep
    for i in range(n_frames):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        # Set aspect ratio for figure
        # size_factor = 4.0
        # w, h = cols*size_factor, (rows+0.4)*size_factor
        # plt.gcf().set_size_inches(w, h)
        # Add title for overall figure
        title_str = '{:}, $t=${:2.2f}'
        fig.suptitle(title_str.format(name, t[i]))
        # Plot
        plot_t_slice(xm[i], zm[i], y[i], fig, ax[0][0], r'$x$', r'$z$', r'Input', title_font, body_font)
        plot_t_slice(xm[i], zm[i], y_cd[i], fig, ax[0][1], r'$x$', r'$z$', r'Output', title_font, body_font)
        # plot_t_slice(xm[i], zm[i], up[i]-up_cd[i], fig, ax[1][0], r'$x$', r'$z$', r'Up Diff')
        # plot_t_slice(xm[i], zm[i], dn[i]-dn_cd[i], fig, ax[1][1], r'$x$', r'$z$', r'Down Diff')
        plot_t_slice(xm[i], zm[i], A_cd[i].real, fig, ax[1][0], r'$x$', r'$z$', r'$\tilde{A}$', title_font, body_font)
        plot_t_slice(xm[i], zm[i], B_cd[i].real, fig, ax[1][1], r'$x$', r'$z$', r'$\tilde{B}$', title_font, body_font)
        # plot_t_slice(xm[i], zm[i], C_cd[i].real, fig, ax[1][0], r'$x$', r'$z$', r'C')
        # plot_t_slice(xm[i], zm[i], D_cd[i].real, fig, ax[1][1], r'$x$', r'$z$', r'D')
        fig.tight_layout() # this (mostly) prevents axis labels from overlapping
        # Save figure as image in designated output directory
        save_fig_as_frame(fig, i, output_path, dpi)
        plt.close(fig)

# then run
# python3 create_gif.py 2D_CD_test 2D_CD_test.gif 2D_CD_frames
