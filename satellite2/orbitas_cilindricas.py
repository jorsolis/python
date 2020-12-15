#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas

"""
import scipy.integrate as spi
import numpy as np
from plots_jordi import *

L = 0.2
v = 1.
q = 0.9    

labcond = r"$L=$ %f ,$v_\rho(0)=$ %f, $\rho(0)=$ %f , $v_z(0)=$ %f , $ z(0)=$ %f "

u0 = 0.
uf = 100.
du = 0.01
#
u = np.arange(u0, uf, du)

ncor = 8
    
def los3plotscoords(t,ro,z,phi,name1,name2,name3,title):
    coordsplot(t,ro,uf,r'$\hat\mu \rho$',title, name1)
    coordsplot(t,z,uf,r'$\hat\mu z$',title, name2)
    coordsplotang(t,phi, r'$\phi$', title, name3)
    parametricplot(ro,z,r'$\rho$', r'$z$', title, name3)

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step
##########################################################################
###########################################################################
def resolvedor(u0,uf,y0,method,teval,L):
    #method= 'RK45', 'Radau' or 'LSODA'
    def fun(t,y):#con potencial logaritmico
        return [L**2/y[1]**3 - v**2*y[1]/(y[1]**2 + y[3]**2/q**2),
                y[0],
                -v**2*y[3]/(q**2*(y[1]**2+y[3]**2/q**2)),
                y[2],
                L/y[1]**2]
      
    sol = spi.solve_ivp(fun, [u0,uf], y0, method=method, t_eval=teval)
    if sol.status==0:
        los3plotscoords(sol.t,sol.y[1],sol.y[3],sol.y[4],
                        "cil\rho(t)_%d" % ncor,
                        "cil\z(t)_%d" % ncor,"cil\phi_cyl(t)_%d" % ncor,
                        labcond % conds)
        los2plots3dcyl(sol.y[1],sol.y[3],sol.y[4],
                    "cil\cyl_3d_%d" % ncor,
                    "cil\cyl_3d_2_%d" % ncor, labcond % conds)
    else:
        print "status", sol.status

vrho0 = 0.1
rho0 = .5
vz0 = -1.
z0 = 3

conds = (L,vrho0, rho0, vz0, z0)
y0 = [vrho0, rho0, vz0, z0,0]
resolvedor(u0,uf,y0,'LSODA',u,L)

#
#        AHORA EN ESFERICAS
#def los3plotscoordsesf(t,r,theta,phi,name1,name2,name3,title):
#    ps.coordsplot(t,r,uf,r'$r$',title, name1)
#    ps.coordsplot(t,theta,uf,r'$\theta$',title, name2)
#    ps.coordsplotang(t,phi,uf, r'$\phi$', title, name3)
#    ps.parametricplot(r,theta,r'$r$', r'$\theta$', title, name3)
#def los2plots3desf(r,theta,phi,name1,name2,title):
#    ps.plot3d(r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi),r*np.cos(theta),
#              45,30,title, name1)
#    ps.plot3d(r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi),r*np.cos(theta),
#              20,0,title, name2)
#def resolvedor(u0,uf,y0,method,teval,L):
#    #method= 'RK45', 'Radau' or 'LSODA'
#    def dVdx(L,q,v,y):
#        "derivada parcial del potencial respecto a x"
#        return v**2/y[1] - L**2/(y[1]**3*np.sin(y[3])**2), 
#    
#    def dVdtheta(L,q,v,y):
#        "1/x^2 dV/dtheta"
#        return np.cos(y[3])*(((q**2-1)*v**2*np.sin(y[3])**2)/(q**2*np.sin(y[3])**2+np.cos(y[3])**2)- L**2/(y[1]**2*np.sin(y[3])**2))/(y[1]**2*np.sin(y[3])),
# 
#    def fun(t,y):
#        return [L**2/(y[1]**3*np.sin(y[3])**2) + y[1]*y[2]**2 - dVdx(L,q,v,y),
#                y[0],
#                -2*y[0]*y[2]/y[1] + (L**2*np.cos(y[3]))/(y[1]**4*np.sin(y[3])**3) 
#                - dVdtheta(L,q,v,y),
#                y[2],
#                L/(y[1]**2*np.sin(y[3])**2)]        
#    sol = spi.solve_ivp(fun, [u0,uf], y0, method=method, t_eval=teval)
#    if sol.status==0:
#        los3plotscoordsesf(sol.t,sol.y[1],sol.y[3],sol.y[4],"r(t)_log_%d" % ncor,
#                        "theta(t)_log_%d" % ncor,"phi_log(t)_%d" % ncor, labcond2 % conds2)
#        los2plots3desf(sol.y[1],sol.y[3],sol.y[4],
#                    "cyl_3d_%d" % ncor,
#                    "cyl_3d_2_%d" % ncor, labcond2 % conds2)
#    else:
#        print "status", sol.status
#
#vrho0 = 0.1
#rho0 = .5
#vz0 = -1.
#z0 = 3
#
#vr0 = 0. 
#r0 = np.sqrt(rho0**2+z0**2)
#vth0 = 0.
#th0 = np.arctan(rho0/z0)
#
#labcond2 = r"$L=$ %f ,$v_r(0)=$ %f, $r(0)=$ %f , $v_\theta(0)=$ %f , $ \theta(0)=$ %f "
#conds2 = (L,vr0, r0, vth0, th0)
#y0 = [vr0, r0, vth0, th0,0]
#resolvedor(u0,uf,y0,'LSODA',u,L)