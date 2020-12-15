#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas

"""
import scipy.integrate as spi
import numpy as np
from plots_jordi import *


b = 1378
#####   Constantes
mu = 15.6378 # en pc
c = 0.3  # pc/año
#####       condiciones iniciales
ri = 500        # parcecs
########
L = 1000
y0_0 = 0.1      #\hat{\mu} dr/dphi
y1_0 = ri*mu    #x(0)= \mu r(0)

y0 = [y0_0, y1_0]

labcond = r"$v_r(0)=$ %f, $r(0)=$ %f pc, $L=$ %f"
##  u = phi
u0 = 0.
uf = 2*np.pi
du = uf/1000
 
def func(y, t):
    return [2*y[0]**2/y[1]+y[1]- (y[1]**4/L**2)*b/y[1]**2,
            y[0]]
#
####    SOLVER
u = np.arange(u0, uf, du)
y = spi.odeint(func, y0, u)
#
####      PLOTS   
coordsplot(u,y[:, 1],uf,r'$\hat\mu r$',r'$\phi$',labcond % (y0_0, ri, L),"r(phi)_esf")
parametricplot(y[:,1],y[:,0], r"$\phi$", r"$\mu r$",labcond % (y0_0, ri, L),"rvsphi_esf")
parametricplot(y[:,1]*np.cos(u),y[:,1]*np.sin(u),r"$x\mu$",r"$y\mu$",labcond % (y0_0, y1_0, L),"rvsphi_esf_2")
plot3d(y[:,1]*np.cos(u), y[:,1]*np.sin(u), 0)
