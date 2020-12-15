#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ecuaciones diferenciales
z''=-v^2 z /(q^2(r^2+z^2/q^2))
r''=l^2/r^3-r v^2/(r^2+z^2/q^2)
"""
import scipy.integrate as spi
from plots_jordi import *
import numpy as np

#constantes
v=1.
l=0.2
q=0.9
#condiciones iniciales [0.1,.5,-1,3]
y0_0 = 0.10  #r'(0)
y1_0 = 0.5 #r(0)
y2_0 = -1. #z'(0)
y3_0 = 3. # z(0)
y0 = [y0_0, y1_0, y2_0, y3_0]
print(y0)
def func(y, t):
    return [l**2/y[1]**3-v**2*y[1]/(y[1]**2+y[3]**2/q**2),
            y[0],
            -v**2*y[3]/(q**2*(y[1]**2+y[3]**2/q**2)),
            y[2]]
t0 = 0
tf = 200.01
dt = 0.01
t = np.arange(t0, tf, dt)
y = spi.odeint(func, y0, t)
#
#coords2plot(y[:, 1],y[:, 3],t,tf,r'$r(t)$',r'$z(t)$',
#           r"$v_r(0)=0.1$, $v_z(0)=-1$, $r(0)=0.5$, $z(0)=3$",
#           "coords_cyl_trimaine") 
#
parametricplot(y[:, 1], y[:, 3], r'$r(t)$', r'$z(t)$',
               r'$z$ vs $r$',
               "orbita_cilindrica_rvsz")