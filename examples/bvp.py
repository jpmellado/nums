#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Finite difference equations. Steady-state solutions of heat equation

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from template import *

TypesToShow = []
# TypesToShow += ['bvp1d']
TypesToShow += ['bvp2d']

###############################################################################
tag = 'bvp1d'
if tag in TypesToShow:
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

###############################################################################
tag = 'bvp2d'
if tag in TypesToShow:
    fig_id = 0

    # Input data
    nx = 6; lx = 2.0
    ny = 5; ly = 1.0
    # nx = 20; lx = 2.0
    # ny = 20; ly = 1.0

    def s(x, y):                      # Define source term
        s = np.empty_like(x)
        s = np.sin( np.pi *x/lx )*np.sin( np.pi *y/ly )
        return s

    # Create grid
    x1 = np.linspace( 0.0, lx, nx )
    hx = (x1[-1]-x1[0]) /(nx-1)
    y1 = np.linspace( 0.0, ly, ny )
    hy = (y1[-1]-y1[0]) /(ny-1)
    x, y = np.meshgrid(x1,y1)

    # Create array for the field
    u = np.empty_like(x)
    u[:,0] = 1.0        # Dirichlet boundary condition at x_1
    u[:,-1] = 1.0       # Dirichlet boundary condition at x_n
    u[0,:] = 1.0        # Dirichlet boundary condition at y_1
    u[-1,:] = 1.0       # Dirichlet boundary condition at y_n
    
    # Create system array
    n = (nx-2)*(ny-2)
    Ax = np.diag(np.full(n, -2.))                       # Create array and fill main diagonal
    np.fill_diagonal(Ax[1:,:],np.full(n-1,1.))          # Fill lower diagonal
    Ax[nx-2::nx-2,nx-3::nx-2] = 0.
    np.fill_diagonal(Ax[:,1:],np.full(n-1,1.))          # Fill upper diagonal
    Ax[nx-3::nx-2,nx-2::nx-2] = 0.
    # print(Ax)
    
    Ay = np.diag(np.full(n, -2.))                       # Create array and fill main diagonal
    np.fill_diagonal(Ay[nx-2:,:],np.full(n-(nx-2),1.))  # Fill lower diagonal
    np.fill_diagonal(Ay[:,nx-2:],np.full(n-(nx-2),1.))  # Fill upper diagonal
    # print(Ay)

    A = (1./hx**2)*Ax + (1./hy**2)*Ay
    
    # Create array for the source
    source = s(x,y)
    
    # Create forcing term
    b = -source[1:-1,1:-1].flatten()
    
    b[0::nx-2] -= (1./hx**2)*u[1:ny-1,0]           # Dirichlet boundary condition at x_1
    b[nx-3::nx-2] -= (1./hx**2)*u[1:ny-1,-1]       # Dirichlet boundary condition at x_n
    b[:nx-2] -= (1./hy**2)*u[0,1:nx-1]             # Dirichlet boundary condition at y_1
    b[n-(nx-2):] -= (1./hy**2)*u[-1,1:nx-1]        # Dirichlet boundary condition at y_n
    # print(b)
        
    # Solve the system. We use generic routines, but one could use solve_banded
    sol = scipy.linalg.solve( A, b )
    u[1:-1,1:-1] = sol.reshape((ny-2,nx-2))

    #################################################
    fig_id = fig_id +1
    fig = plt.figure( figsize = figsize11 )

    c = plt.contourf(x,y,u)
    plt.xlabel(r'Horizontal distance $x$')
    plt.ylabel(r'Horizontal distance $y$')
    plt.title(r'solution of $\nabla^2u +s=0$')
    plt.grid()
    plt.colorbar(c,label=r"$u$")
    plt.tight_layout(pad=0.1)

    plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

    #################################################
    fig_id = fig_id +1
    fig = plt.figure( figsize = figsize11 )

    c = plt.contourf(x,y,source)
    plt.xlabel(r'Horizontal distance $x$')
    plt.ylabel(r'Horizontal distance $y$')
    plt.title(r'source $s$')
    plt.grid()
    plt.colorbar(c,label=r"$s$")
    plt.tight_layout(pad=0.1)

    plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

plt.show()

