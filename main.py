"""
1D Bousinessq equations:

dz(w) + iku = 0
dt(b) - ka(dzz-k^2)b = -N^2*w -(wdz + iku)b
dt(u) - nu(dzz-k^2)u + ikp = -(wdz + iku)u
dt(w) - nu(dzz-k^2)w + dz(p) - b = -(wdz + iku)w

where k is the horizontal wavenumber

This script should be ran serially (because it is 1D).

"""

import numpy as np
import matplotlib.pyplot as plt
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from dedalus.core.operators import GeneralFunction

import logging
logger = logging.getLogger(__name__)

###############################################################################
# Checking command line arguments
import sys
# Arguments must be passed in the correct order
arg_array = sys.argv
switchboard = str(arg_array[0])

###############################################################################
# Import SwitchBoard Parameters (sbp)
#   This import assumes the switchboard is in the same directory as the core code
import switchboard as sbp
# Physical parameters
nu          = sbp.nu            # [m^2/s] Viscosity (momentum diffusivity)
kappa       = sbp.kappa         # [m^2/s] Thermal diffusivity
g           = sbp.g             # [m/s^2] Acceleration due to gravity
# Problem parameters
N_0         = sbp.N_0           # [rad/s]       Reference stratification
lam_x       = sbp.lam_x         # [m]           Horizontal wavelength
lam_z       = sbp.lam_z         # [m]           Vertical wavelength
k           = sbp.k             # [m^-1]        Horizontal wavenumber
m           = sbp.m             # [m^-1]        Vertical wavenumber
k_total     = sbp.k_total       # [m^-1]        Total wavenumber
theta       = sbp.theta         # [rad]         Propagation angle from vertical
omega       = sbp.omega         # [rad s^-1]    Wave frequency
T           = sbp.T             # [s]           Wave period
print('phase speed is',omega/m,'m/s')

###############################################################################

# # Bases and domain
z_basis = sbp.z_basis#de.Fourier('z', sbp.nz, interval=(sbp.z0, sbp.zf), dealias=sbp.dealias)
domain = sbp.domain#de.Domain([z_basis], np.float64)

# Z grid
z_da = sbp.z_da#domain.grid(0, scales=domain.dealias)
z = sbp.z#domain.grid(0)

# Define problem
problem = de.IVP(domain, variables=['b', 'p', 'u', 'w'])
problem.parameters['NU'] = nu
problem.parameters['KA'] = kappa
problem.parameters['N0'] = N_0

###############################################################################
# Forcing from the boundary

# Boundary forcing parameters
A         = 2.0e-4
problem.parameters['k']     = k
problem.parameters['m']     = m
problem.parameters['omega'] = omega

# Polarization relation from boundary forcing file
PolRel = {'u': -A*(g*omega*m)/(N_0**2*k),
          'w':  A*(g*omega)/(N_0**2),
          'b':  A*g}
        # 'p': -A*(g*m)/(k**2+m**2)}
# Creating forcing amplitudes
for fld in ['u', 'w', 'b']:#, 'p']:
    BF = domain.new_field()
    BF['g'] = PolRel[fld]
    problem.parameters['BF' + fld] = BF  # pass function in as a parameter.
    del BF

# Temporal ramp for boundary forcing
if sbp.temporal_ramp:
    problem.parameters['T']     = T       # [s] period of oscillation
    problem.parameters['nT']    = sbp.nT  # [] number of periods for the ramp
    problem.substitutions['ramp']   = "(1/2)*(tanh(4*t/(nT*T) - 2) + 1)"
else:
    problem.substitutions['ramp']   = "1"
# Substitutions for boundary forcing (see C-R & B eq 13.7)
problem.substitutions['fu'] = "BFu*sin(m*z - omega*t)*ramp"
problem.substitutions['fw'] = "BFw*sin(m*z - omega*t)*ramp"
problem.substitutions['fb'] = "BFb*cos(m*z - omega*t)*ramp"
# problem.substitutions['fp'] = "BFp*sin(m*z - omega*t)*ramp"

###############################################################################
# Background Profile for N_0
BP = domain.new_field(name = 'BP')
BP_array = z*0+1
for i in range(len(BP_array)):
    if z[i] < -0.3 and z[i] > -0.4:
        BP_array[i] = 0
BP['g'] = BP_array
problem.parameters['BP'] = BP

###############################################################################
# Boundary forcing window
win_bf = domain.new_field(name = 'win_bf')
a_bf    = sbp.a_bf              # [] amplitude ("height") of the forcing window
b_bf    = sbp.b_bf              # [m] full width at half max of forcing window
c_bf    = sbp.c_bf              # [m] location of center of boundary forcing window
problem.parameters['tau_bf'] = sbp.tau_bf
# The equation for the window in z
win_bf_array = a_bf*np.exp(-4*np.log(2)*((z - c_bf)/b_bf)**2)
win_bf['g'] = win_bf_array
problem.parameters['win_bf'] = win_bf

# Creating forcing terms
for fld in ['u', 'w', 'b']:#, 'p']:
    # terms will be = win_bf * (f(psi) - psi)
    problem.substitutions['F_term_' + fld] = "win_bf * (f"+fld+" - "+fld+")/tau_bf"
# problem.substitutions['bf_term'] = " win_bf * (a*sin(-m*z - omega*t) - w)*ramp"

###############################################################################
# Sponge window
win_sp = domain.new_field(name = 'win_sp')
a_sp    = sbp.a_sp              # [] amplitude ("height") of the sponge window
b_sp    = sbp.b_sp              # [m] full width at half max of sponge window
c_sp    = sbp.c_sp              # [m] location of center of sponge window
problem.parameters['tau_sp'] = sbp.tau_sp
# The equation for the window in z
win_sp_array = a_sp*np.exp(-4*np.log(2)*((z - c_sp)/b_sp)**2)
win_sp['g'] = win_sp_array
problem.parameters['win_sp'] = win_sp

# Creating sponge terms
for fld in ['u', 'w', 'b']:#, 'p']:
    # terms will be = win_bf * (f(psi) - psi)
    problem.substitutions['S_term_' + fld] = "win_sp * "+fld+" / tau_sp"
# problem.substitutions['sp_term'] = "-win_sp * w / tau"

###############################################################################
# Plotting windows
if sbp.plot_windows:
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,3]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw=plot_ratios, sharey=True)
    #
    axes[0].plot(BP_array, z, label='Background profile')
    axes[0].set_xlabel('$N_0$')
    axes[0].set_ylabel(r'$z$')
    axes[0].set_title(r'Background Profile')
    #
    axes[1].plot(win_bf_array, z, label='Boundary forcing')
    axes[1].plot(win_sp_array, z, label='Sponge layer')
    axes[1].set_xlabel('Amplitude')
    #axes[1].set_ylabel(r'$z$')
    axes[1].set_title(r'Windows')
    axes[1].legend()
    #
    plt.savefig('f_1D_windows.png')

###############################################################################
# Define equations

problem.add_equation("dz(w) - k*u = 0")
problem.add_equation("dt(b) - KA*(dz(dz(b)) - (k**2)*b) " \
                     " = -((N0*BP)**2)*w - (w*dz(b) + k*u*b) " \
                     " + F_term_b - S_term_b ")
problem.add_equation("dt(u) - NU*(dz(dz(u)) - (k**2)*u) + k*p " \
                     " = - (w*dz(u) + k*u*u) " \
                     " + F_term_u - S_term_u ")
problem.add_equation("dt(w) - NU*(dz(dz(w)) - (k**2)*w) + dz(p) - b " \
                     " = - (w*dz(w) + k*u*w) " \
                     " + F_term_w - S_term_w ")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')
solver.stop_sim_time  = sbp.sim_time_stop
solver.stop_wall_time = sbp.stop_wall_time
solver.stop_iteration = sbp.stop_iteration

# Above code modified from here: https://groups.google.com/forum/#!searchin/dedalus-users/%22wave$20equation%22%7Csort:date/dedalus-users/TJEOwHEDghU/g2x00YGaAwAJ

###############################################################################

# Initial conditions
b = solver.state['b']
u = solver.state['u']
w = solver.state['w']

b['g'] = 0.0
u['g'] = 0.0
w['g'] = 0.0

###############################################################################
# Analysis
def add_new_file_handler(snapshot_directory='snapshots/new', sdt=sbp.snap_dt):
    return solver.evaluator.add_file_handler(snapshot_directory, sim_dt=sdt, max_writes=sbp.snap_max_writes, mode=sbp.fh_mode)

# Add file handler for snapshots and output state of variables
snapshots = add_new_file_handler('snapshots')
snapshots.add_system(solver.state)

###############################################################################

# CFL
CFL = flow_tools.CFL(solver, initial_dt=sbp.dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('w'))

###############################################################################

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("(k*u + m*w)/omega", name='Lin_Criterion')

###############################################################################

# Store data for final plot
w.set_scales(1)
w_list = [np.copy(w['g'])]
t_list = [solver.sim_time]

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    dt = sbp.dt
    while solver.proceed:
        solver.step(dt)
        if solver.iteration % 1 == 0:
            w.set_scales(1)
            w_list.append(np.copy(w['g']))
            t_list.append(solver.sim_time)
        if solver.iteration % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))


# Create space-time plot
w_array = np.transpose(np.array(w_list))
t_array = np.array(t_list)

xmesh, ymesh = quad_mesh(x=t_array, y=z)
plt.figure()
im = plt.pcolormesh(xmesh, ymesh, w_array, cmap='RdBu_r')
plt.axis(pad_limits(xmesh, ymesh))

# Find max of absolute value for colorbar for limits symmetric around zero
cmax = max(abs(w_array.flatten()))
if cmax==0.0:
    cmax = 0.001 # to avoid the weird jump with the first frame
# Set upper and lower limits on colorbar
im.set_clim(-cmax, cmax)

plt.colorbar()
plt.xlabel('t')
plt.ylabel('z')
plt.title(r'Forced 1D Wave, $(k,m,\omega)$=(%g,%g,%g)' %(problem.parameters['k'], m, omega))
plt.savefig('f_1D_wave.png')

# plot forcing at boundary vs. times
# plt.clf()
# plt.figure()
# plt.plot(t_array, bf_array)
# plt.xlabel('t')
# plt.ylabel('Forcing amplitude')
# plt.title(r'Boundary forcing vs. time, $(a,m,\omega)$=(%g,%g,%g)' %(problem.parameters['a'], m, omega))
# plt.savefig('f_1D_boundary.png')
