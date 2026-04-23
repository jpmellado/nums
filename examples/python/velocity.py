import numpy as np
import matplotlib.pyplot as plt

import pynums.fdms.fdm1 as fdms1
import pynums.fdms.fdm2 as fdms2
from partials import *

# Create grid
nx = 50 
x = np.linspace(0.,1.,nx)
hx = (x[-1]-x[0]) /(nx-1)

ny = 20
y = np.linspace(0.,1.,ny)
hy = (y[-1]-y[0]) /(ny-1)

xc, yc = np.meshgrid( x, y )

# Create object for the partial derivatives
# for biased boundary conditions, use the following ones
partial_1 = partial(scheme=fdms1.fdm1_e121, step_x=hx, step_y=hy) 
partial_2 = partial(scheme=fdms2.fdm2_e121, step_x=hx*hx, step_y=hy*hy)
# for periodic boundary conditions, use the following ones
# partial_1 = partial(scheme=fdms1.fdm1_e2p, step_x=hx, step_y=hy) 
# partial_2 = partial(scheme=fdms2.fdm2_e2p, step_x=hx, step_y=hy) 

# Create stream function
psi = xc*(1.-xc)*yc*(1-yc)

fig = plt.figure( figsize = (4,3) )
c = plt.pcolormesh(x, y, psi)
plt.colorbar(c)

# velocity
u = partial_1.dy(psi)
v = -partial_1.dx(psi)

fig = plt.figure( figsize = (4,3) )
c = plt.pcolormesh(x, y, np.sqrt(u**2+v**2))
plt.streamplot(x, y, u, v)
plt.colorbar(c)

# vorticity
omega =-partial_2.dx(psi) -partial_2.dy(psi)

fig = plt.figure( figsize = (4,3) )
c = plt.pcolormesh(x, y, omega)
plt.colorbar(c)

# strain
s =0.5*(partial_2.dy(psi)-partial_2.dx(psi)) 

fig = plt.figure( figsize = (4,3) )
c = plt.pcolormesh(x, y, s)
plt.colorbar(c)

plt.show()

