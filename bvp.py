#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Finite difference equations. Steady-state solutions of heat equation

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from template import *

# Figure names and counter
tag = 'bvp'
fig_id = 0

# Input data
n = 20
u1 = 0.0                        # Dirichlet boundary condition at x_1
un = 1.0                        # Dirichlet boundary condition at x_n

def s(x):                       # Define source term
    s = np.empty(np.size(x))
    s = 10. *np.sin( np.pi *x )
    return s

# Create grid
x = np.linspace( 0.0, 1.0, n )
h = (x[-1]-x[0]) /(n-1)

# Create vector for solution
u = np.empty(n)
u[ 0] = u1
u[-1] = un

# Create system array
A2 = np.diag(np.full(n-2, 2.))                      # Create array and fill main diagonal
np.fill_diagonal(A2[1:,:],np.full(n-3,-1.))         # Fill lower diagonal
np.fill_diagonal(A2[:,1:],np.full(n-3,-1.))         # Fill upper diagonal

# Create forcing term
b = s(x)[1:-1] *h **2.                              # Source term
b[ 0] += u1                                         # Boundary conditions at x1
b[-1] += un                                         # Boundary conditions at xn

# Solve the system. We use generic routines, but one could use solve_banded
u[1:-1] = scipy.linalg.solve( A2, b )

#################################################
# Plotting
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )

plt.plot( x, u,   )
plt.xlabel(r'Horizontal distance $x$')
plt.ylabel(r'Temperature $u$')
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

#################################################
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )

plt.plot( x, s(x), 'C1' )
plt.xlabel(r'Horizontal distance $x$')
plt.ylabel(r'Heat source $s$')
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

plt.show()
