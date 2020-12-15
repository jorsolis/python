#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:34:24 2018

@author: jordis
"""
#import astropy.units as u
#import astropy.coordinates as coord
#from astropy.io import ascii
import matplotlib.pyplot as plt
#import numpy as np
#from matplotlib import rcParams
#rcParams['figure.figsize'] = (10, 6)

# Gala
from gala.mpl_style import mpl_style
#plt.style.use(mpl_style)
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

Potential = gp.from_equation("1/2*k*x**2", vars="x", pars="k",
                          name='HarmonicOscillator')
p1 = Potential(k=1.)
print p1
orbit = p1.integrate_orbit([1.,0], dt=0.01, n_steps=1000)
fig = orbit.plot()
plt.savefig("harmonic_oscillator")