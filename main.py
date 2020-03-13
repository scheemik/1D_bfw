"""
Main code of Dedalus quickstart script

Modified by Mikhail Schee from the script for 2D Rayleigh-Benard convection in the Dedalus example files.

An attempt to implement counter rotation (complex demodulation) to separate the upward and downward portions of a propagating internal wave. If I add those two back together, I should get the original back.
Referencing these resources:
    Mercier et al. (2008) "Reflection and diffraction of internal waves analyzed..."
    Grisouard and Thomas (2015) "Critical and near-critical reflections of near-inertial..."
    Individual meeting notes - Nov  4, 2019
    Individual meeting notes - Jul 26, 2019

*This script is NOT meant to be run directly.*

To run this program, execute the command (-c and -v are optional):
    $ bash run.sh -n <my_new_exp> -c 2 -v 1

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

###############################################################################

# Domain parameters
nx, nz = 512, 512
x0, xf =  0.0, 1.0
z0, zf = -1.0, 0.0

# Physical parameters
nu          = 1.0E-6        # [m^2/s] Viscosity (momentum diffusivity)
kappa       = 1.4E-7        # [m^2/s] Thermal diffusivity
g           = 9.81          # [m/s^2] Acceleration due to gravity

# Boundary forcing parameters
set_lams = True # first option, set lambda's
N_0     = 1.0                   # [rad/s]
if set_lams:    # specify lam_x and lam_z, calculate the rest
    lam_x   = (xf - x0) / 2.0       # [m]
    lam_z   = (zf - z0) / 1.25       # [m]
    #
    k_x     = 2*np.pi / lam_x       # [m^-1]
    k_z     = 2*np.pi / lam_z       # [m^-1]
    k       = np.sqrt(k_x**2 + k_z**2) # [m^-1]
    theta   = np.arcsin(k_z/k)      # [rad]
    omega   = N_0 * np.cos(theta)   # [rad s^-1]
else:           # specify theta and lam_x, calculate the rest
    theta   = np.pi / 4.0 #(45deg)  # [rad]
    lam_x   = (xf - x0) / 2.0       # [m]
    #
    k_x     = 2*np.pi / lam_x       # [m^-1]
    omega   = N_0 * np.cos(theta)   # [rad s^-1]
    k       = k_x * N_0 / omega     # [m^-1]
    k_z     = k*np.sin(theta)       # [m^-1]
    lam_z   = 2*np.pi / k_z         # [m]

T       = 2*np.pi / omega       # [s]
print('T =', T)

# Run parameters
stop_sim_time = 10*T
dt = 0.125
adapt_dt = False
snap_dt = 3*dt
snap_max_writes = 50

###############################################################################

fourier_fourier = False
windowed_boundary_forcing = False
start_at_steady_state = True

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(x0, xf), dealias=3/2)
if fourier_fourier:
    z_basis = de.Fourier('z', nz, interval=(z0, zf), dealias=3/2)
else:
    z_basis = de.Chebyshev('z', nz, interval=(z0, zf), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
x = domain.grid(0)
z = domain.grid(1)

# 2D Boussinesq hydrodynamics
if fourier_fourier:
    problem = de.IVP(domain, variables=['p','b','u','w'])
else:
    problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
    #   variables are dirichlet by default
    problem.meta['p']['z']['dirichlet'] = False
    problem.meta['p','bz','uz','wz']['z']['dirichlet'] = False
problem.parameters['NU'] = nu
problem.parameters['KA'] = kappa
problem.parameters['N0'] = N_0

###############################################################################
# Forcing from the boundary

# Boundary forcing parameters
A         = 2.0e-4
buffer    = 0.05
problem.parameters['T']         = T   # [s] period of oscillation
problem.parameters['nT']        = 3.0 # number of periods for the ramp
problem.parameters['slope']     = 25
problem.parameters['left_edge'] = buffer + 0.0
problem.parameters['right_edge']= buffer + lam_x
problem.parameters['kx']        = k_x
problem.parameters['kz']        = k_z
problem.parameters['omega']     = omega
# Polarization relation from boundary forcing file
PolRel = {'u': A*(g*omega*k_z)/(N_0**2*k_x),
          'w': A*(g*omega)/(N_0**2),
          'b': A*g}
# Creating forcing amplitudes
for fld in ['u', 'w', 'b']:#, 'p']:
    BF = domain.new_field()
    BF.meta['x']['constant'] = True  # means the NCC is constant along x
    BF['g'] = PolRel[fld]
    problem.parameters['BF' + fld] = BF  # pass function in as a parameter.
    del BF
# Spatial window and temporal ramp for boundary forcing
if windowed_boundary_forcing:
    #problem.substitutions['window'] = "(1/2)*(tanh(slope*(x-left_edge))+1)*(1/2)*(tanh(slope*(-x+right_edge))+1)"
    problem.substitutions['window'] = "1.0*exp(-((x-0.5)**2/0.2 + (z+0.5)**2/0.2))"
else:
    problem.substitutions['window'] = "1"

if start_at_steady_state:
    problem.substitutions['ramp']   = "1"
else:
    problem.substitutions['ramp']   = "(1/2)*(tanh(4*t/(nT*T) - 2) + 1)"
# Substitutions for boundary forcing (see C-R & B eq 13.7)
problem.substitutions['fu']     = "-BFu*sin(kx*x + kz*z - omega*t)*window*ramp"
problem.substitutions['fw']     = " BFw*sin(kx*x + kz*z - omega*t)*window*ramp"
problem.substitutions['fb']     = "-BFb*cos(kx*x + kz*z - omega*t)*window*ramp"

###############################################################################
# Equations of motion (non-linear terms on RHS)
if fourier_fourier:
    problem.add_equation("dx(u) + dz(w) = 0")
    problem.add_equation("dt(b) - KA*(dx(dx(b)) + dz(dz(b)))" \
                    + "= -(N0**2)*w - (u*dx(b) + w*dz(b))")
    problem.add_equation("dt(u) -NU*dx(dx(u)) - NU*dz(dz(u)) + dx(p)" \
                    + "= - (u*dx(u) + w*dz(u))")
    problem.add_equation("dt(w) -NU*dx(dx(w)) - NU*dz(dz(w)) + dz(p) - b" \
                    + "= - (u*dx(w) + w*dz(w))")
else:
    problem.add_equation("dx(u) + wz = 0")
    problem.add_equation("dt(b) - KA*(dx(dx(b)) + dz(bz))" \
                    + "= -(N0**2)*w - (u*dx(b) + w*bz)")
    problem.add_equation("dt(u) -NU*dx(dx(u)) - NU*dz(uz) + dx(p)" \
                    + "= - (u*dx(u) + w*uz)")
    problem.add_equation("dt(w) -NU*dx(dx(w)) - NU*dz(wz) + dz(p) - b" \
                    + "= - (u*dx(w) + w*wz)")
    problem.add_equation("bz - dz(b) = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    # Boundary contitions
    problem.add_bc("left(u) = 0")
    problem.add_bc("right(u) = right(fu)")
    problem.add_bc("left(w) = 0", condition="(nx != 0)")
    problem.add_bc("right(w) = right(fw)")
    problem.add_bc("left(b) = 0")
    problem.add_bc("right(b) = right(fb)")
    problem.add_bc("left(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

###############################################################################
# Adding fields for Complex Demodulation (Hilbert Transform, kinda)

# Reference local grid and state fields
b = solver.state['b']
u = solver.state['u']
w = solver.state['w']
if not fourier_fourier:
    bz = solver.state['bz']

# Get local wavenumbers (from: https://groups.google.com/forum/#!searchin/dedalus-users/new_field$20state%7Csort:date/dedalus-users/-cM7jkfWl68/YtFqpksRCQAJ)
ks_x = domain.elements(0)
ks_z = domain.elements(1)

# Testing
w_g = domain.new_field(name='w_g')
w_g['g'] = w['g']
w_c = domain.new_field(name='w_c')
w_c['c'] = w['c']

###############################################################################

# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():
    # Initial conditions

    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    # Linear background + perturbations damped at walls
    zb, zt = z_basis.interval
    pert =  1e-3 * noise * (zt - z) * (z - zb)
    if start_at_steady_state:
        # Reconstructing the substitutions for boundary forcing, sans t
        b['g'] = -1*PolRel['b']*(np.cos(k_x*x + k_z*z) + np.cos(k_x*x - k_z*(z + lam_z/4.0)))
        u['g'] = -1*PolRel['u']*(np.sin(k_x*x + k_z*z) + np.sin(k_x*x - k_z*(z + lam_z/4.0)))
        w['g'] =  1*PolRel['w']*(np.sin(k_x*x + k_z*z) + np.sin(k_x*x - k_z*(z + lam_z/4.0)))
    else:
        b['g'] = 0.0
        u['g'] = 0.0
        w['g'] = 0.0
    if not fourier_fourier:
        b.differentiate('z', out=bz)

    # Output
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    # Timestepping and output
    dt = last_dt
    stop_sim_time = 50
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = 180 * 60.0 # to get minutes
solver.stop_iteration = np.inf

###############################################################################
# Analysis
def add_new_file_handler(snapshot_directory='snapshots/new', sdt=snap_dt):
    return solver.evaluator.add_file_handler(snapshot_directory, sim_dt=sdt, max_writes=snap_max_writes, mode=fh_mode)

# Add file handler for snapshots and output state of variables
snapshots = add_new_file_handler('snapshots')
snapshots.add_system(solver.state)

# Add file handler for Complex Demodulation (CD)
# CD = add_new_file_handler('snapshots/CD')
# CD.add_task("w_g", layout='g', name='w_g')
# CD.add_task("w_c", layout='c', name='w_c')

###############################################################################

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

###############################################################################

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("(kx*u + kz*w)/omega", name='Lin_Criterion')

###############################################################################

# Main loop
try:
    logger.info('Starting loop')
    logger.info('Simulation end time: %e' %(stop_sim_time))
    start_time = time.time()
    while solver.proceed:
        if (adapt_dt):
            dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, of: %e' %(solver.iteration, solver.sim_time, stop_sim_time))
            #logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max linear criterion = {0:f}'.format(flow.max('Lin_Criterion')))
            if np.isnan(flow.max('Lin_Criterion')):
                raise NameError('Code blew up it seems')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
    logger.info('Oscillation period (T): %f' %T)
