#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 17:00:32 2018

@author: jordis
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)

# Gala
from gala.mpl_style import mpl_style
#plt.style.use(mpl_style)
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

def F(t,w,sigma,rho,beta):
    x,y,z,px,py,pz = w
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z, 0., 0., 0.]).reshape(w.shape)
sigma, rho, beta = 10., 28., 8/3.
integrator = gi.DOPRI853Integrator(F, func_args=(sigma, rho, beta))
orbit = integrator.run([0.5,0.5,0.5,0,0,0], dt=1E-2, n_steps=1E4)
fig = orbit.plot()