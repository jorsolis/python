#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scipy import arange, optimize, special

plt.figure()
w = []# arreglo de x_max
z = []# f(x_max)
x = arange(0,10,.01)

for k in arange(1,5,.5):
   y = special.jv(k,x)
   plt.plot(x,y)
   f = lambda x: -special.jv(k,x)
   x_max = optimize.fminbound(f,0,6)
   w.append(x_max)
   z.append(special.jv(k,x_max))

plt.plot(w,z, 'ro')
from scipy import interpolate
t = interpolate.splrep(w, z, k=1)
s_fit1 = interpolate.splev(x,t)
plt.plot(x,s_fit1, 'g-')
t3 = interpolate.splrep(w, z, k=3)
s_fit3 = interpolate.splev(x,t)
plt.plot(x,s_fit3, 'g-')
t5 = interpolate.splrep(w, z, k=5)
s_fit5 = interpolate.splev(x,t5)
plt.plot(x,s_fit5, 'y-')
