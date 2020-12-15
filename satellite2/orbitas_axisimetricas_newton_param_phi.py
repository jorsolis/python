#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas

"""
import scipy.integrate as spi
import numpy as np
from plots_jordi import *

mu = 15.6378 # en pc
c = 0.3  # pc/año
#####       condiciones iniciales
ri = 8500.        # parcecs
Mmw = 0.005  #pc
Mhal= 0.05
b = mu*Mmw
L = ri*mu*np.sqrt(Mmw)/np.sqrt(ri)
print "b=", b
#########
y0_0 = 0.01      #\hat{\mu} dr/dphi
y1_0 = ri*mu    #x(0)= \mu r(0)
y2_0 = 0.0001      #dtheta/dphi (0)
y3_0 = np.pi/2  # theta(0)
y0 = [y0_0, y1_0, y2_0, y3_0]
print y0
ncor = 2
labcond = r"$d_\phi r(0)=$ %f , $r(0)=$ %f pc, $d_\phi \theta(0)=$ %f, $\theta(0)=$ %f"
##  u = phi
u0 = 0
uf= 4*np.pi

du = uf/10000

u = np.arange(u0, uf, du)
 
def func(y, t,b,L):
    return [y[1]*y[2]**2+y[1]*np.sin(y[3])**2*2.*y[0]**2/y[1] + 2.*y[0]*y[2]*np.cos(y[3])/np.sin(y[3]) - ((y[1]*np.sin(y[3])**4/L**2)*b/y[1]**2),
            y[0],
            -2.*y[2]**2*np.cos(y[3])/np.sin(y[3]) + np.sin(y[3])*np.cos(y[3]),
            y[2]]
    
def fun(t, y):
    return [y[1]*y[2]**2+y[1]*np.sin(y[3])**2*2.*y[0]**2/y[1] + 2.*y[0]*y[2]*np.cos(y[3])/np.sin(y[3]) - ((y[1]*np.sin(y[3])**4/L**2)*b/y[1]**2),
            y[0],
            -2.*y[2]**2*np.cos(y[3])/np.sin(y[3]) + np.sin(y[3])*np.cos(y[3]),
            y[2]]
####    SOLVER

y = spi.odeint(func, y0, u, args=(b,L))
#
####      PLOTS   
coordsplot(u, y[:,1], uf,r'$\phi$' ,r'$\hat\mu r$',
           labcond % (y0_0, ri, y2_0, y3_0),"esf/new/r(phi)_sol_%d" % ncor)
coordsplotang(u,y[:,3],r'$\phi$',r'$\theta$',labcond % (y0_0, ri,y2_0,y3_0),
              "esf/new/theta(phi)_sol_%d" % ncor)
parametricplot(u,y[:,0],r'$\phi$',r"$v_r$",
               labcond % (y0_0, ri,y2_0,y3_0),"esf/new/paramphi/vr(phi)_sol_%d" % ncor)
parametricplot(y[:,1]*np.cos(u),y[:,1]*np.sin(u),r"$x\mu$",r"$y\mu$",
               labcond % (y0_0, ri,y2_0,y3_0),"esf/new/paramphi/xy_proy_phi_sol_%d" % ncor)

los2plots3d(y[:,1],y[:,3],u,u,"esf/new/oparamphi/rb_phi_sol_1_%d" % ncor,
            "esf/new/paramphi/orb_phi_sol_2_%d" % ncor,labcond % (y0_0, ri,y2_0,y3_0))

sol = spi.solve_ivp(fun, [u0,uf], y0, method='LSODA', t_eval=u)

if sol.status == -1 or sol.status == 1:
    print "status", sol.status
else:
    coordsplot(sol.t,sol.y[1], uf, r'$\phi$', r'$\hat\mu r$',
               labcond % (y0_0, ri, y2_0, y3_0),"esf/new/paramphi/r(phi)_3_%d" % ncor)
    coordsplot(sol.t,sol.y[0], uf, r'$\phi$', r'$v_r$',
               labcond % (y0_0, ri, y2_0, y3_0),"esf/new/paramphi/prue%d" % ncor)  
    coordsplot(sol.t,sol.y[2], uf, r'$\phi$', r'$v_\theta$',
               labcond % (y0_0, ri, y2_0, y3_0),"esf/new/paramphi/prue%d" % ncor)  
    coordsplotang(sol.t, sol.y[3], r'$\phi$', r'$\theta $',
                  labcond % (y0_0, ri, y2_0, y3_0),
                  "esf/new/paramphi/theta(phi)_3_%d" % ncor)
    parametricplot(sol.y[1]*np.cos(sol.t),sol.y[1]*np.sin(sol.t),
                   r"$x\mu$",r"$y\mu$",
                   labcond % (y0_0, ri,y2_0,y3_0),
                   "xy_proy_phi_sol_2_%d" % ncor)
    print sol.y.shape, sol.y[1].shape, sol.t.shape      
    los2plots3d(sol.y[1],sol.y[3],sol.t,sol.t,"esf/new/paramphi/orb_phi_sol_3_%d" % ncor,
                "esf/new/paramphi/orb_phi_sol_4_%d" % ncor,labcond % (y0_0, ri,y2_0,y3_0))
