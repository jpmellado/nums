#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Stability regions of time marching schemes
# Eigenvalues of Jacobian various difference equations
# Working with complex numbers in python

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from template import *

TypesToShow = []
# TypesToShow += ['regions']
TypesToShow += ['fdm-eigenvalues']

###############################################################################
# Definitions

# Characteristic polynomial
def rEuler(w):
    return 1. + w           

def rEulerInverse(w):
    return 1. /( 1. - w )

def rLeapFrog(w):
    return w +np.sqrt( 1. + w **2.), w -np.sqrt( 1. + w **2.)

def rAdamsBashforth2(w):
    return 0.5 *( tmp + np.sqrt(tmp **2. -2. *w) ), 0.5 *( tmp - np.sqrt(tmp **2. -2. *w) )

def rRungeKutta2(w):
    return 1. + w + w **2. /2.

def rRungeKutta3(w):
    return 1. + w + w **2. /2. + w **3. /6.

def rRungeKutta45(w):
    return 1. + w + w **2. /2. + w **3. /6. + w **4. /24. + w **5. /200.

# Plotting
def PlotRegion(w,r,tag,limits):
# Plots the statbility region
#   - w: region of the complex space
#   - r: amplification factor
#   - tag: String with the name of the time marching scheme
#   - limits: list with countour points in the real and imaginary axis
    fig = plt.figure( figsize = figsize11 )
    plt.contourf(np.real(w),np.imag(w),r,[0., 1.],colors=['#aedcc0'],alpha=0.5)
#    plt.xticks([-3,-2,-1,0,1,2])
    plt.title(r'Stability Region of '+tag)
    plt.xlabel(r'Re($\lambda\Delta t)$')
    plt.ylabel(r'Im($\lambda\Delta t)$')
    plt.gca().grid(True)
    plt.gca().set_aspect('equal')#,'box')
    for item in limits:
        plt.plot(item[0],item[1],'x',color='black',label=r'(${:3.2f},{:3.2f}$)'.format(item[0],item[1]))
        plt.legend(title='limits',loc="best",fontsize='small',handletextpad=0.0, handlelength=0,markerscale=0.)

    plt.tight_layout(pad=0.1)

    return

def PlotError(w,r,s,tag):
# Plots the statbility region
#   - w: region of the complex space
#   - r: amplification factor
#   - s: error factor
#   - tag: String with the name of the time marching scheme
    colors = [ '#507dbc', '#bbd1ea', '#f9b5ac', '#ee7674' ]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize21)

    id = 0
    axs[id].contourf(np.real(w),np.imag(w),r,[0., 1.],colors=['#aedcc0'],alpha=0.5)
    axs[id].contour( np.real(w),np.imag(w),r,    [1.],colors=['k'],linewidths=[1.0])
    axs[id].contourf(np.real(w),np.imag(w),abs(s),[0.9,0.99,1.,1.01,1.10],colors=colors,alpha=0.75)
    axs[id].contour( np.real(w),np.imag(w),abs(s),[1.],linewidths=[1.0],colors='w')
    axs[id].set_title(r'Amplitude Error of '+tag)
    axs[id].set_xlabel(r'Re($\lambda\Delta t)$')
    axs[id].set_ylabel(r'Im($\lambda\Delta t)$')
    axs[id].grid(True)
    axs[id].set_aspect('equal')#,'box')
    axs[id].text(0.05,0.10,r'Light colors: 1\% error',transform=axs[id].transAxes)
    axs[id].text(0.05,0.03,r'Dark colors:  10\% error',transform=axs[id].transAxes)

    id = 1
    axs[id].contourf(np.real(w),np.imag(w),r,[0., 1.],colors=['#aedcc0'],alpha=0.5)
    axs[id].contour( np.real(w),np.imag(w),r,    [1.],colors=['k'],linewidths=[1.0])
    axs[id].contourf(np.real(w),np.imag(w),np.angle(s) /np.pi,[-0.1,-0.01,0.,0.01,0.1],colors=colors,alpha=0.75)
    axs[id].contour( np.real(w),np.imag(w),np.angle(s) /np.pi,[0.],linewidths=[0.5],colors='w')
    axs[id].set_title(r'Phase Error of '+tag)
    axs[id].set_xlabel(r'Re($\lambda\Delta t)$')
#    axs[id].set_ylabel(r'Im($\lambda\Delta t)$')
#    axs[id].get_yaxis().set_visible(False)
    axs[id].set_yticklabels([])
    axs[id].grid(True)
    axs[id].set_aspect('equal')#,'box')

    plt.tight_layout(pad=0.1)

    return

def PlotEigenvalues(lambdas):
    m = np.size(lambdas)
    cmap = plt.get_cmap('magma_r')
    for il in range(m):
        markersize = float(m-il) /float(m) *5.
        factor = float(il+1) /float(m) *1.
        plt.plot(np.real(lambdas[il]),np.imag(lambdas[il]),marker='o',markersize=markersize,markeredgewidth=0.,
                alpha=factor,color='blue',label='_nolegend_')
        # plt.plot(np.real(lambdas[il]),np.imag(lambdas[il]),marker='o',markersize=5.,markeredgewidth=0.,
        #         color=cmap(factor),label='_nolegend_')

    plt.tight_layout(pad=0.1)

    return

# Create grid of complex numbers
x = np.linspace(-2.5,2.5,200)
# y = np.linspace(-2.0,2.0,200)
#x = np.linspace(-3.75,2.5,200)
y = np.linspace(-2.5,2.5,200)
x, y = np.meshgrid(x,y)
w = x + 1j *y 

###############################################################################
tag = 'regions'      # stability regions
if tag in TypesToShow:
    fig_id = 0

    # ---------------------------------
    name = 'Euler'
    r = rEuler(w)
    limits = [ [ -2.,0] ]

    PlotRegion(w,np.abs(r),name,limits)
    plt.savefig("{}.pdf".format('region-'+name),bbox_inches='tight')

    plt.title(r"Equation $\mathrm{d}_t u=-u$. Euler.")
    PlotEigenvalues(lambdas=[ -2.1, -0.8, -0.5 ])
    plt.savefig("{}.pdf".format('eigenvalues-'+name),bbox_inches='tight')

    PlotRegion(w,np.abs(r),name,limits)
    plt.title(r"Equation $u''+u=0$. Euler.")
    PlotEigenvalues(lambdas=[ -1.0 *1j, -0.5 *1j, 0.5 *1j, 1.0 *1j])
    plt.savefig("{}.pdf".format('eigenvalues-'+name+'Oscillation'),bbox_inches='tight')

    PlotError(w,np.abs(r),r/np.exp(w),name)
    plt.savefig("{}.pdf".format('error-'+name),bbox_inches='tight')

    # ---------------------------------
    name = 'EulerInverse'
    r = rEulerInverse(w)
    limits = []

    PlotRegion(w,np.abs(r),name,limits)
    plt.savefig("{}.pdf".format('region-'+name),bbox_inches='tight')

    PlotError(w,np.abs(r),r/np.exp(w),name)
    plt.savefig("{}.pdf".format('error-'+name),bbox_inches='tight')

    # ---------------------------------
    #name = 'Leap-Frog'
    #r1 = w +np.sqrt( 1. + w **2.)
    #r2 = w -np.sqrt( 1. + w **2.)
    #s = r1/np.exp(w)
    #r = np.maximum(np.abs(r1),np.abs(r2))
    #limits = []

    # Adams-BashForth 2. order
    #name = 'Adams-Bashforth 2'
    #tmp = 1. +1.5 *w
    #r1 = 0.5 *( tmp + np.sqrt(tmp **2. -2. *w) )
    #r2 = 0.5 *( tmp - np.sqrt(tmp **2. -2. *w) )
    #s = r1/np.exp(w)
    #r = np.maximum(np.abs(r1),np.abs(r2))
    #limits = []

    # ---------------------------------
    name = 'RungeKutta2'
    r = rRungeKutta2(w)
    limits = [ [ -2.,0] ]

    PlotRegion(w,np.abs(r),name,limits)
    plt.savefig("{}.pdf".format('region-'+name),bbox_inches='tight')

    PlotError(w,np.abs(r),r/np.exp(w),name)
    plt.savefig("{}.pdf".format('error-'+name),bbox_inches='tight')

    # ---------------------------------
    name = 'RungeKutta3'
    r = rRungeKutta3(w)
    limits = [ [0,1.73], [0, -1.73], [ -2.57,0] ]

    PlotRegion(w,np.abs(r),name,limits)
    plt.savefig("{}.pdf".format('region-'+name),bbox_inches='tight')

    plt.title(r"Equation $u''+u=0$. Runge-Kutta 3.")
    PlotEigenvalues(lambdas=[ -1.0 *1j, -0.5 *1j, 0.5 *1j, 1.0 *1j])
    plt.savefig("{}.pdf".format('eigenvalues-'+name+'Oscillation'),bbox_inches='tight')

    PlotError(w,np.abs(r),r/np.exp(w),name)
    plt.savefig("{}.pdf".format('error-'+name),bbox_inches='tight')

    # ---------------------------------
    # name = 'RungeKutta45'
    # r = rRungeKutta45(w)
    # limits = []

#%% Calculate eigenvalues of Jacobian
###############################################################################
tag = 'fdm-eigenvalues'
if tag in TypesToShow:
    fig_id = 0

    n = 20          # # of grid points

    #######################
    # Diffusion equation with Dirichlet BC; (n-2)x(n-2) matrix
    fig_id = fig_id +1
    a = np.ones(n)
    L = np.diagflat(a[1:n-1],-1) -2. *np.diagflat(a[:n-1],0) + np.diagflat(a[1:n-1],1)
    L = 1./2. *L    # To get the condition for maximum diffusion number

    # # Effect of mass matrix with FE methods
    # A = 1./6. *( np.diagflat(a[1:n-1],-1) +4. *np.diagflat(a[:n-1],0) + np.diagflat(a[1:n-1],1) )
    # L = scipy.linalg.solve(A,L,overwrite_b=True)
    # L = 1/6. *L

    lambdas = scipy.linalg.eigvals(L)
    lambdas = lambdas[np.argsort(np.real(lambdas))]

    # ---------------------------------
    name = 'Euler'
    r = rEuler(w)
    limits = [ [ -2.,0] ]

    PlotRegion(w,np.abs(r),name,limits)
    PlotEigenvalues(lambdas)
    plt.title(r"Equation $u_t=u_{xx}$. Euler.")
    plt.savefig("{}.pdf".format(tag+name+str(fig_id)),bbox_inches='tight')

    #######################
    # Advection equation with periodic BC; nxn matrix
    fig_id = fig_id +1
    a = np.ones(n)
    c = np.ones(n)
    L = 0.5 *np.diagflat(a[1:],-1) - 0.5 *np.diagflat(c[:n-1],1)
    L[0,n-1] = 0.5
    L[n-1,0] =-0.5

    lambdas = scipy.linalg.eigvals(L)
    # lambdas = np.sort_complex( lambdas )
    lambdas = lambdas[np.argsort(np.imag(lambdas))]

    # ---------------------------------
    name = 'Euler'
    r = rEuler(w)
    limits = [ [ -2.,0] ]

    PlotRegion(w,np.abs(r),name,limits)
    PlotEigenvalues(lambdas)
    plt.title(r"Equation $u_t+cu_{x}$. Euler.")
    plt.savefig("{}.pdf".format(tag+name+str(fig_id)),bbox_inches='tight')

    # ---------------------------------
    name = 'RungeKutta3'
    r = rRungeKutta3(w)
    limits = [ [0,1.73], [0, -1.73], [ -2.57,0] ]

    PlotRegion(w,np.abs(r),name,limits)
    PlotEigenvalues(lambdas)
    plt.title(r"Equation $u_t=-cu_{x}$. RungeKutta3.")
    plt.savefig("{}.pdf".format(tag+name+str(fig_id)),bbox_inches='tight')

    #######################
    # Advection equation with Dirichlet BC; (n-1)x(n-1) matrix
    fig_id = fig_id +1
    a = np.ones(n-1)
    c = np.ones(n-1)
    L = 0.5 *np.diagflat(a[1:],-1) - 0.5 *np.diagflat(c[:n-2],1)
    L[n-2,n-2] =-1.
    L[n-2,n-3] = 1.

    lambdas = scipy.linalg.eigvals(L)
    lambdas = lambdas[np.argsort(np.imag(lambdas))]

    # ---------------------------------
    name = 'RungeKutta3'
    r = rRungeKutta3(w)
    limits = [ [0,1.73], [0, -1.73], [ -2.57,0] ]

    PlotRegion(w,np.abs(r),name,limits)
    PlotEigenvalues(lambdas)
    plt.title(r"Equation $u_t=-cu_{x}$. RungeKutta3.")
    plt.savefig("{}.pdf".format(tag+name+str(fig_id)),bbox_inches='tight')

    #######################
    # Advection-diffusion equation with Dirichlet and Neumann conditions; (n-2)x(n-2) matrix
    # fig_id = fig_id +1
    #a = np.ones(n-2)
    #B1 = 0.5 *np.diagflat(a[1:],-1) - 0.5 *np.diagflat(a[:n-3],1)
    #B1[n-3,n-4] = 2./3.
    #B1[n-3,n-3] =-2./3.
    #
    #for i in range(n-2): # loop over the rows
    #    B1[i,:] = B1[i,:] *( 1. - float(i+1)/float(n-1) )
    #
    #B2 = np.diagflat(a[1:],-1) -2. *np.diagflat(a,0) +np.diagflat(a[:n-3],1)
    #B2[n-3,n-4] = 2./3.
    #B2[n-3,n-3] =-2./3.
    #
    #mu1 = 1.5
    #mu2 = 0.5
    #
    #L = mu1 *B1 + mu2 *B2

###############################################################################
plt.show()