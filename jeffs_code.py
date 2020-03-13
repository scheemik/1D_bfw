from dedalus import public as de
import numpy as np
import matplotlib.pyplot as plt
import h5py


pi2 = np.pi*2

#grid constants
nx = 128
ny = 1024
Ly = 1024e3 # meters
dy = Ly/ny
Lx = nx*dy
dt = 1e1/2


#physical constants
g = 9.81 # gravity
mu = 1e12# viscosity
H = 4000 #4e3 # meters, mean depth
f = 1.03e-4 # coriolis

xbasis = de.Fourier('x', nx, interval=(-Lx/2,Lx/2), dealias=1) #dealias = 3/2 # this oworks with fourier(wihtout BC) and Chebyshev
ybasis = de.Fourier('y', ny, interval=(-Ly/2,Ly/2), dealias=1)

domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64)
x = domain.grid(0)
y = domain.grid(1)


# Create problem
problem = de.IVP(domain, variables=['u', 'v', 'h'])
problem.parameters['g'] = g
problem.parameters['mu'] = mu
problem.parameters['f'] = f
problem.parameters['H'] = H


#wave forcing
l = 6000./1000*pi2/Ly #poincare wave number
omega = np.sqrt(g*l**2*H + f**2) #dispersion relation. Omega is essentially f at high wave numbers
height_forcing = 0.25
u_forcing = height_forcing*f/l/H
v_forcing = height_forcing*omega/l/H
T  = pi2/omega/60/60 #Period in hours
wavelen = pi2/l/1000 #Wavelength in kmi
tau = 10


problem.parameters['u_force'] = u_forcing
problem.parameters['v_force'] = v_forcing
problem.parameters['h_force'] = height_forcing
problem.parameters['l'] = l
problem.parameters['omega'] = omega
problem.parameters['tau'] = tau
problem.parameters['pi'] = np.pi


ncc = domain.new_field(name = 'window')
ncc['g'] = np.exp(-((y+ 7*Ly/8./2)/(Ly/16./2.))**2)
ncc.meta['x']['constant'] = True
problem.parameters['window'] = ncc

sp = domain.new_field(name = 'sponge_window')
sp['g'] = np.exp(-((y- 7*Ly/8/2)/(Ly/16./2.))**2)
sp.meta['x']['constant'] = True
problem.parameters['sponge'] = sp

ramptime = 10
problem.parameters['ramptime'] = ramptime

# hyper viscosity?
#problem.substitutions['dis(A)'] = "mu*(dx(dx(A)) + dy(dy(A)))"
problem.substitutions['dis(A)'] = "-mu*(dx(dx((dx(dx(A))))) + dy(dy(dy(dy(A)))))"

# Main equation, with linear terms on the LHS and nonlinear terms on the RHS
problem.add_equation("dt(u) -f*v + g*dx(h) - dis(u) = \
                     -u*dx(u) - v*dy(u)  + (u_force*cos(l*y-omega*t) -u)/tau*window*arctan(t*f/ramptime)*2/pi \
                     + (-u)/tau*sponge")

problem.add_equation("dt(v) +f*u + g*dy(h) - dis(v) = \
                     -u*dx(v) - v*dy(v)   + (v_force*cos(l*y-omega*t) -v)/tau*window*arctan(t*f/ramptime)*2/pi \
                     + (-v)/tau*sponge ")

problem.add_equation("dt(h) - dis(h) = -h*(dx(u) + dy(v)) - u*dx(h) - v*dy(h) + \
                     (H + h_force*sin(l*y-omega*t) -h)/tau*window*arctan(t*f/ramptime)*2/pi \
                     + (H-h)/tau*sponge")

# its going over one

solver = problem.build_solver(de.timesteppers.RK443)
u = solver.state['u']
v = solver.state['v']
h = solver.state['h']


h0 = 1.0 # meters, initial height of pertubations

l = 2*np.pi/Lx #geo_mode wave number
k = 2*np.pi/Lx

hf = h5py.File('init_turb.h5', 'r')
slices = domain.dist.grid_layout.slices(scales=(1,1))


h['g'] = hf.get('iso_turbulence')[slices]*np.exp(-(3*y/(Ly/2))**2)
h.differentiate('y', out=u)
u['g'] *= -1*g/f
h.differentiate('x', out=v)
v['g'] *= g/f

#h['g'] = H
#u['g'] = 0
#v['g'] = 0

solver.stop_iteration = 4000
solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf

analysis = solver.evaluator.add_file_handler('../data/state', iter=10, max_writes=1000)

analysis.add_system(solver.state, layout='g')

from dedalus.extras import flow_tools
# CFL
#CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=0.8,
#                     max_change=1.5, threshold=0.05)
#CFL.add_velocities(('u', 'v'))

import time

# Main loop
start_time = time.time()
while solver.ok:
#    dt = CFL.compute_dt()
    solver.step(dt)
    if solver.iteration % 100 == 0:
        print('Completed iteration {}'.format(solver.iteration))
end_time = time.time()
print('Runtime:', end_time-start_time)
