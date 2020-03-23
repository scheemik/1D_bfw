from dedalus import public as de
import numpy as np
import matplotlib.pyplot as plt

de.logging_setup.rootlogger.setLevel('ERROR')

xbasis = de.Fourier('x', 32, interval=(0,5), dealias=3/2)
zbasis = de.Chebyshev('z', 32, interval=(0,5), dealias=3/2)

grid_Fourier = xbasis.grid(scale=3/2)
grid_Chebyshev = zbasis.grid(scale=3/2)

plt.figure(figsize=(10, 1))
ax = plt.subplot(111)
ax.plot(grid_Fourier, np.zeros_like(grid_Fourier)+1, 'o', markersize=5, label='Fourier')
ax.plot(grid_Chebyshev, np.zeros_like(grid_Chebyshev)-1, 'o', markersize=5, label='Chebyshev')
plt.ylim([-2, 2])
plt.gca().yaxis.set_ticks([]);

chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.1, 0.8), shadow=True, ncol=1)

plt.show()
