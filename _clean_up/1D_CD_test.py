import numpy as np
import matplotlib.pyplot as plt

def make_subplot(hori, vert, data, fig, ax, h_label, v_label, title):
    im = ax.pcolormesh(hori, vert, data)
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

# fourier transform in time, filter negative freq's, inverse fourier transform
def FT_in_time(t, z, data, dt):
    # FT in time of the data (axis 1 is time)
    ftd = np.fft.fft(data, axis=1)
    # find relevant frequencies
    freq = np.fft.fftfreq(len(t), dt)
    f_grid, z_grid = np.meshgrid(freq, z)
    # Filter out negative frequencies
    for i in range(f_grid.shape[0]):
        for j in range(f_grid.shape[1]):
            if f_grid[i][j] < 0.0:
                # Gets rid of negative freq's
                ftd[i][j] = 0
            else:
                # Corrects for lost amplitude
                ftd[i][j] = ftd[i][j] * 2.0
    # inverse fourier transform in time of the data
    iftd = np.fft.ifft(ftd, axis=1)
    #   a complex valued signal where iftd.real == data, or close enough
    return iftd

# fourier transform in spatial dimension (z)
#   similar to FT in time, but switch dimensions around
def FT_in_space(t, z, data, dz):
    # FT in space (z) of the data (axis 0 is z) for positive wave numbers
    fzdp = np.fft.fft(data, axis=0)
    # make a copy for the negative wave numbers
    fzdn = fzdp.copy()
    # find relevant wavenumbers
    k_zs = np.fft.fftfreq(len(z), dz)
    t_grid, k_grid = np.meshgrid(t, k_zs)
    # Filter out one half of wavenumbers to separate up and down
    for i in range(k_grid.shape[0]):
        for j in range(k_grid.shape[1]):
            if k_grid[i][j] > 0.0:
                # for up, remove values for positive wave numbers
                fzdp[i][j] = 0.0
            else:
                # for down, remove values for negative wave numbers
                fzdn[i][j] = 0.0
    # inverse fourier transform in space (z)
    ifzdp = np.fft.ifft(fzdp, axis=0)
    ifzdn = np.fft.ifft(fzdn, axis=0)
    return ifzdp, ifzdn

omega = 2.0
omega2 = omega
kz    = 2.0
kz2   = kz
A = 1.0
B = A * 1

# number of oscillation periods
n_o = 4
t0 = 0.0
tf = (2*np.pi) / omega * n_o
nt = 512
dt = (tf-t0)/nt

z0 = 0.0
zf = 1.0
zf = (2*np.pi) / kz
nz = 512
dz = (zf-z0)/nz

t = np.linspace(t0, tf, nt)
z = np.linspace(z0, zf, nz)
tm, zm = np.meshgrid(t, z)
# Total field
y = A*np.cos(omega * tm - kz*zm) + B*np.cos(omega2 * tm + kz2*zm)
# Separate up and down analytically for confirmation that CD is working
up = A*np.cos(omega * tm - kz*zm)
dn = B*np.cos(omega2 * tm + kz2*zm)

t_then_z = False
if t_then_z == True:
    ## Step 1
    ift_t_y = FT_in_time(t, z, y, dt)
    ### Step 2
    ift_z_y_p, ift_z_y_n = FT_in_space(t, z, ift_t_y, dz)
    # Get up and down fields as F = |mag_f| * exp(i*phi_f)
    up_field = ift_z_y_p.real * np.exp(np.real(1j * ift_z_y_p.imag))
    dn_field = ift_z_y_n.real * np.exp(np.real(1j * ift_z_y_n.imag))
else:
    ## Step 1
    ift_z_y_p, ift_z_y_n = FT_in_space(t, z, y, dz)
    ## Step 2
    up_f = FT_in_time(t, z, ift_z_y_p, dt)
    dn_f = FT_in_time(t, z, ift_z_y_n, dt)
    # Get up and down fields as F = |mag_f| * exp(i*phi_f)
    up_field = up_f.real * np.exp(np.real(1j * up_f.imag))
    dn_field = dn_f.real * np.exp(np.real(1j * dn_f.imag))


# plotting
diff = True
if diff == True:
    rows = 1
    cols = 3
    fig, ax = plt.subplots(rows, cols)
    make_subplot(tm, zm, y-(up_field+dn_field), fig, ax[0], r'$t$', r'$z$', r'Total difference')
    make_subplot(tm, zm, up-up_field, fig, ax[1], r'$t$', r'$z$', r'Up difference')
    make_subplot(tm, zm, dn-dn_field, fig, ax[2], r'$t$', r'$z$', r'Down difference')
else:
    rows = 2
    cols = 3
    fig, ax = plt.subplots(rows, cols)
    make_subplot(tm, zm, y, fig, ax[0][0], r'$t$', r'$z$', r'$A\cos(\omega t-k_z z)+B\cos(\omega t+k_z z)$')
    make_subplot(tm, zm, up, fig, ax[0][1], r'$t$', r'$z$', r'$A\cos(\omega t-k_z z)$')
    make_subplot(tm, zm, dn, fig, ax[0][2], r'$t$', r'$z$', r'$B\cos(\omega t+k_z z)$')
    make_subplot(tm, zm, up_field+dn_field, fig, ax[1][0], r'$t$', r'$z$', 'Up + Down')
    make_subplot(tm, zm, up_field, fig, ax[1][1], r'$t$', r'$z$', 'Up')
    make_subplot(tm, zm, dn_field, fig, ax[1][2], r'$t$', r'$z$', 'Down')

# Set aspect ratio for figure
size_factor = 3.0
w, h = cols*size_factor, (rows+0.4)*size_factor
plt.gcf().set_size_inches(w, h)
plt.tight_layout()
plt.show()
