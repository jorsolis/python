#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:31:31 2019

https://colab.research.google.com/drive/1lIJ6guEAH5NQObefYBJ7S_Jm21IlJSOo#scrollTo=furqtEH2S1gS

@author: jordis
"""

import numpy as np
import scipy.special as spe
import plots_jordi as pts

def poisson_2d_solve(u_0, dx, dy, s, num_timesteps):
    def move_up(u, i, j):
        return ((dy**2) * (u[i+1][j] + u[i-1][j]) +
                (dx**2) * (u[i][j+1] + u[i][j-1]) -
                s[i][j] * dx**2 * dy**2) / (2*(dx**2 + dy**2))
  
    u_prev = u_0.copy()
    u_next = u_0.copy()
    
    for _ in range(num_timesteps):
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u_next[i][j] = move_up(u_prev, i, j)
    # boundery conditions
        u_next[0,:] = 0
        u_next[-1,:] = 0
        u_next[:,0] = 0
        u_next[:,-1] = 0
    
        u_prev = u_next.copy()

    #    print "u_next=", u_next[-1,:]
    print np.shape(u_next)
    return u_next  
   
nx = ny = 51
lx = ly = np.pi

dx = lx / (nx-1)
dy = ly / (ny-1)

nt = 501

x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)

u_0 = np.zeros((nx, ny))

##########  LA FUENTE   #############################

#s = np.zeros((nx, ny))
#s[nx//4, ny//4] = 100
#s[3*nx//4, 3*ny//4] = -100

nu = 1
kmu = 0.458263#k/mu
X, Y = np.meshgrid(x, y)
#s = spe.jv(nu,kmu*X)**2 * spe.legendre(1)(np.cos(Y))**2   
s = -5*np.sin(X)*np.sin(2*Y) 
#s = np.tanh(X)*np.cosh(Y)
    
u_end = poisson_2d_solve(u_0, dx, dy, s, nt)

pts.plotfunc3d(X,Y,u_end,r"$x$",r"$y$","z","Solution",
               rot_azimut=-60)
pts.plotfunc3d(X,Y,np.sin(X)*np.sin(2*Y),r"$x$",r"$y$","z","Analytic solution",
               rot_azimut=-60)

#pts.plotfunc3d(X,Y,s,r"$x$",r"$y$","z","Source",
#               rot_azimut=-60)