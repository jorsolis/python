#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas

"""
import scipy.integrate as spi
import numpy as np
from plots_jordi import *

###########################################################################
############################################################################
def resolvedor(u0,uf,y0,method,teval,L,a,b,d,f,M):
    #method= 'RK45', 'Radau' or 'LSODA'
    def fun(t,y):
        return [L**2/y[1]**3 -b*y[1]/np.sqrt(y[1]**2+y[3]**2)**3 
                - M*y[1]/np.sqrt((d+np.sqrt(f**2+y[3]**2))**2+y[1]**2)**3 
                - a*y[3]*(np.pi-2*np.arctan2(y[3],y[1]))/(y[1]**2+y[3]**2),
                y[0],
                -b*y[3]/np.sqrt(y[1]**2+y[3]**2)**3 
                - M*y[3]*(1+d/np.sqrt(f**2+y[3]**2))/np.sqrt((d+np.sqrt(f**2+y[3]**2))**2+y[1]**2)**3 
                - a*y[1]*(np.pi-2*np.arctan2(y[3],y[1]))/(y[1]**2+y[3]**2),
                y[2],
                L/y[1]**2]   
    sol = spi.solve_ivp(fun, [u0,uf], y0, method=method, t_eval=teval)
    if sol.status==0:
        plotscoordscyl(sol.t,sol.y[1],sol.y[3],sol.y[4],sol.y[0],sol.y[2],
                           uf,"cil/new/rho(t)_%d" % ncor,
                           "cil/new/z(t)_mim_%d" % ncor,
                           "cil/new/phi(t)_cyl_mim_%d" % ncor,
                           "cil/new/zvsrho_mim_%d" % ncor,
                           "cil/new/xy_cyl_mim_%d" % ncor,
                           labcond % conds)
        los2plots3dcyl(sol.y[1],sol.y[3],sol.y[4],sol.t,
                    "cil/new/cyl_mim_3d_%d" % ncor,
                    "cil/new/cyl_mim_3d_2_%d" % ncor, labcond % conds)

    else:
        print( "status", sol.status)
#
a = 0.2957
b = 1378.
d = 4943.
f = 1096.
M = 11590.
#
labcond = r"$L=$ %f, $v_\rho(0)=$ %f $c$, $\rho(0)=$ %d, $v_z(0)=$ %f $c$, $\hat{\mu} z(0)=$ %f"

mu = 15.6378 # en 1/pc
c = 0.3  # pc/año
#####       condiciones iniciales
tf = 2*1e4   #años
ri = 10.        # parcecs
#
Mmw = 0.005  #pc
Mhal= 0.05
#b = mu*Mhal

L = ri*mu*np.sqrt(Mhal)/np.sqrt(ri)
u0 = 0.
uf = c*mu*tf
du = uf/10000
u = np.arange(u0, uf, du)

ncor = 5

vrho0 = 0.
rho0 = ri
z0 = -30000.
vz0 = 1e-8

print( "r=", np.sqrt(rho0**2 + z0**2))
#def pot(a,b,d,f,y):
#    return -a*(np.arctan(y[0]/y[1]) - np.pi/2 )**2 - b/np.sqrt(y[0]**2 + y[1]**2) - M/np.sqrt((d+ np.sqrt(f**2+y[1]**2))**2 + y[0]**2)
#y=[rho0,z0]
#vz0 =  np.sqrt(2*(En -pot(a,b,d,f,y) + L**2/(2*rho0**2) - vrho0**2/2))
#print  vz0

conds = (L,vrho0, rho0, vz0, z0)
y0 = [vrho0, rho0, vz0, z0,0]
resolvedor(u0,uf,y0,'LSODA',u,L,a,b,d,f,M)
