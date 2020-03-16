"""
1D Forced wave equation:

d_{tt}w + a*d_{zz}w = F

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

# Bases and domain
z_basis = de.Fourier('z', 1024, interval=(-2, 16), dealias=3/2)
# something weird happens when I use dealias, the number of grid points in x don't line up anymore. I put in 1024 but get back 1536
domain = de.Domain([z_basis], np.float64)

# Problem parameters
a=1.

#Z grid
z_da = domain.grid(0, scales=domain.dealias)
z = domain.grid(0)

#define forcing function
def forcing(z_da,solver):
     return np.cos(z_da-solver.sim_time);
#     return solver.sim_time;
#     return np.cos(z);

#assign GeneralFunction to parameter 'F'
F= GeneralFunction(domain,'g',forcing,args=[])

# Problem
problem = de.IVP(domain, variables=['w', 'wt', 'wz'])

problem.parameters['a'] = a
problem.parameters['F'] = F
problem.add_equation("dt(wt) + a*dz(wz)= F")
problem.add_equation("wt - dt(w) = 0")
problem.add_equation("wz + a*dz(w) = 0")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
solver.stop_wall_time = 10000
solver.stop_iteration = 10000

#pass the relevant arguments to the forcing function
F.args = [z_da,solver]
F.original_args = [z_da,solver]

# Above code modified from here: https://groups.google.com/forum/#!searchin/dedalus-users/%22wave$20equation%22%7Csort:date/dedalus-users/TJEOwHEDghU/g2x00YGaAwAJ

###############################################################################

# Initial conditions
w  = solver.state['w']
wt = solver.state['wt']
wz = solver.state['wz']

w['g'] = 0.0*z
w.differentiate('z', out=wz)
wt['g'] = 0.0
#w.differentiate('t', out=wt)

# Store data for final plot
w.set_scales(1)
w_list = [np.copy(w['g'])]
t_list = [solver.sim_time]

# Main loop
dt = 2e-3
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        solver.step(dt)
        if solver.iteration % 20 == 0:
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
print('shape of w:', w_array.shape)
t_array = np.array(t_list)
print('shape of t:',t_array.shape)
print('shape of z:',z.shape)
xmesh, ymesh = quad_mesh(x=t_array, y=z)
plt.figure()
plt.pcolormesh(xmesh, ymesh, w_array, cmap='RdBu_r')
plt.axis(pad_limits(xmesh, ymesh))
plt.colorbar()
plt.xlabel('t')
plt.ylabel('z')
plt.title('Forced 1D Wave, (a,F)=(%g,%g)' %(problem.parameters['a'], problem.parameters['a']))
plt.savefig('f_1D_wave.png')
