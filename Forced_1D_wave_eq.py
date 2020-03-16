"""
1D Forced wave equation:

d_{tt}u + a*d_{xx}u = F

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
x_basis = de.Fourier('x', 1024, interval=(-2, 16), dealias=3/2)
domain = de.Domain([x_basis], np.float64)

# Problem parameters
a=1.

#X grid
#x = domain.grids(scales=domain.dealias)
x = domain.grid(0, scales=domain.dealias)

#define forcing function
def forcing(x,solver):
     return np.cos(x-solver.sim_time);
#     return solver.sim_time;
#     return np.cos(x);

#assign GeneralFunction to parameter 'F'
F= GeneralFunction(domain,'g',forcing,args=[])

# Problem
problem = de.IVP(domain, variables=['u', 's', 'r'])

problem.parameters['a'] = a
problem.parameters['F'] = F
problem.add_equation("dt(s) + a*dx(r)= F")
problem.add_equation("s - dt(u) = 0")
problem.add_equation("r + a*dx(u) = 0")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
solver.stop_wall_time = 10000
solver.stop_iteration = 10000

#pass the relevant arguments to the forcing function
F.args = [x,solver]
F.original_args = [x,solver]

# Above code modified from here: https://groups.google.com/forum/#!searchin/dedalus-users/%22wave$20equation%22%7Csort:date/dedalus-users/TJEOwHEDghU/g2x00YGaAwAJ

###############################################################################

# Initial conditions
u = solver.state['u']
s = solver.state['s']
r = solver.state['r']

u['g'] = 0.0
u.differentiate('x', out=r)
s['g'] = 0.0
#u.differentiate('t', out=s)

# Store data for final plot
u.set_scales(1)
u_list = [np.copy(u['g'])]
t_list = [solver.sim_time]

# Main loop
dt = 2e-3
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        solver.step(dt)
        if solver.iteration % 20 == 0:
            u.set_scales(1)
            u_list.append(np.copy(u['g']))
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
u_array = np.array(u_list)
t_array = np.array(t_list)
xmesh, ymesh = quad_mesh(x=x, y=t_array)
plt.figure()
plt.pcolormesh(xmesh, ymesh, u_array, cmap='RdBu_r')
plt.axis(pad_limits(xmesh, ymesh))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Forced 1D Wave, (a,F)=(%g,%g)' %(problem.parameters['a'], problem.parameters['F']))
plt.savefig('f_1D_wave.png')
