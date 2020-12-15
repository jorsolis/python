#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:40:23 2019

@author: jordis
"""
import numpy as np
import scipy.special as spe
import plots_jordi as pts

def poisson_2d_solve(u_0, dr, dth, s, num_timesteps):
    def move_up(u, i, j):
        return ((dth**2)* r[i]**2 * (u[i+1][j] + u[i-1][j]) -
                2.*r[i]*dr*dth**2*u[i-1][j] -
                dr**2 * dth *u[i][j-1]/np.tan(th[j])+
                (dr**2) * (u[i][j+1] + u[i][j-1]) -
                s[j][i] * r[i]**2 * dr**2 * dth**2) / (2.*(1+ dth/(2.*np.tan(th[j])))*dr**2 + 2.*r[i]*(r[i]-dr)*dth**2)
  
    u_prev = u_0.copy()
    u_next = u_0.copy()
    
    for _ in range(num_timesteps):
        for i in range(1, nr-1):
            for j in range(1, nth-1):
                u_next[i][j] = move_up(u_prev, i, j)
    

#     boundery conditions
        u_next[0,:] = 0
        u_next[-1,:] = 0
        u_next[:,0] = 0
        u_next[:,-1] = 0
    
        u_prev = u_next.copy()
        
    return u_next  
    
nr = 20
nth = 51
lr = 10.
lth = np.pi

dr = lr / (nr-1)
dth = lth / (nth-1)

nt = 5
r = np.linspace(0, lr, nr)
th = np.linspace(0, lth, nth)

u_0 = np.zeros((nr, nth))
print "u_0_shape=", np.shape(u_0)

##########  LA FUENTE   #############################
nu = 1
kmu = 0.458263#k/mu
#Th, R = np.meshgrid(th, r)

R, Th = np.meshgrid(r, th)
print "R_shape=", np.shape(R)
print "Th_shape=", np.shape(Th)
#s = (0.6575)*spe.spherical_jn(nu,kmu*R)**2 * spe.legendre(1)(np.cos(Th))**2

s = 4*np.cos(Th)
print "s_shape=", np.shape(s)

u_end = poisson_2d_solve(u_0, dr, dth, s, nt)
print "u_end_shape=", np.shape(u_end)


pts.plotfunc3d(R,Th,s,r"$r$",r"$\theta$","z","Source",name= None,elev=45,
               rot_azimut=-60)

Th, R = np.meshgrid(th, r)
print "R_shape=", np.shape(R)
print "Th_shape=", np.shape(Th)

pts.plotfunc3d(R,Th,u_end,r"$r$",r"$\theta$","z","Solution",
               rot_azimut=45)

pts.plotfunc3d(R,Th,R**2*np.cos(Th)/10,r"$r$",r"$\theta$","z",
               " Analytic Solution",rot_azimut=45)
#
#u_0 = np.ones((5, 8))
#print u_0
#print "u_0_shape=", np.shape(u_0)
#u_0[3][5]=3
#print u_0
#print "u_0_shape=", np.shape(u_0)
