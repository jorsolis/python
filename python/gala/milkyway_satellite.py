#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:42:56 2018

@author: jordis
"""

import astropy.units as u
import astropy.coordinates as coord
from astropy.io import ascii
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

potential = gp.MilkyWayPotential()

icrs = coord.ICRS(ra=coord.Angle('17h 20m 12.4s'),
                  dec=coord.Angle('57d54m55s'),
                  distance=76*u.kpc,
                  pm_ra_cosdec=0.0569*u.mas/u.yr,
                  pm_dec=-0.1673*u.mas/u.yr,
                  radial_velocity=-291*u.km/u.s)

icrs_err = coord.ICRS(ra=0*u.deg, dec=0*u.deg, distance=6*u.kpc,
                      pm_ra_cosdec=0.009*u.mas/u.yr,
                      pm_dec=0.009*u.mas/u.yr,
                      radial_velocity=0.1*u.km/u.s)
v_sun = coord.CartesianDifferential([11.1, 250, 7.25]*u.km/u.s)
gc_frame = coord.Galactocentric(galcen_distance=8.3*u.kpc,
                                z_sun=0*u.pc,
                                galcen_v_sun=v_sun)
gc = icrs.transform_to(gc_frame)
w0 = gd.PhaseSpacePosition(gc.data)
orbit = potential.integrate_orbit(w0, dt=-0.5*u.Myr, n_steps=10000)
fig = orbit.plot()

