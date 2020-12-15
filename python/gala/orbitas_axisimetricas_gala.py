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
import numpy as np
from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)
import astropy.units as u
# Gala
from gala.mpl_style import mpl_style
#plt.style.use(mpl_style)
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

#Potential = from_equation("1/2*k*x**2", vars="x", pars="k",
#                          name='HarmonicOscillator')
#p1 = Potential(k=1.)
#orbit = p1.integrate_orbit([1.,0], dt=0.01, n_steps=1000)
#
#
#
#Potential = gp.from_equation("-1/sqrt(r**2 + a**2 + b**2 + 2*a*sqrt(r**2*cos(theta)**2 + b**2)",
#                             vars="r,theta", pars="a,b", name='axi')
Potential = gp.from_equation("-a/x**2+b/x",
                             vars="x", pars="a b", name='axi')
#pot = Potential(a=1.)
#print pot
#orbit = pot.integrate_orbit([1.,0], dt=0.01, n_steps=1000)
#fig = orbit.plot()
#plt.savefig("axi")
####################################
#pot = gp.NFWPotential.from_circular_velocity(v_c=200*u.km/u.s, r_s=10.*u.kpc, units=galactic)
#
#ics = gd.PhaseSpacePosition(pos=[10,0,0.] * u.kpc, vel=[0,175,0] * u.km/u.s)
#orbit = gp.Hamiltonian(pot).integrate_orbit(ics, dt=2., n_steps=2000)
#
#norbits = 128
#new_pos = np.random.normal(ics.pos.xyz.to(u.pc).value, 100., size=(norbits,3)).T * u.pc
#new_vel = np.random.normal(ics.vel.d_xyz.to(u.km/u.s).value, 1., size=(norbits,3)).T * u.km/u.s
#
#new_ics = gd.PhaseSpacePosition(pos=new_pos, vel=new_vel)
#orbits = gp.Hamiltonian(pot).integrate_orbit(new_ics, dt=2., n_steps=2000)
#
#grid = np.linspace(-15,15,64)
#fig,ax = plt.subplots(1, 1, figsize=(5,5))
#fig = pot.plot_contours(grid=(grid,grid,0), cmap='Greys', ax=ax)
#fig = orbits[-1].plot(['x', 'y'], color='red', s=1., alpha=0.5, axes=[ax], auto_aspect=False)
