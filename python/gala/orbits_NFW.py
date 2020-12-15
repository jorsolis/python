
# coding: utf-8
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import gala.integrate as gi
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
#get_ipython().magic(u'matplotlib inline')

pot = gp.NFWPotential.from_circular_velocity(v_c=200*u.km/u.s, r_s=10.*u.kpc, units=galactic)

ics = gd.PhaseSpacePosition(pos=[10,0,0.] * u.kpc, vel=[0,175,0] * u.km/u.s)
orbit = gp.Hamiltonian(pot).integrate_orbit(ics, dt=2., n_steps=2000)

norbits = 128
new_pos = np.random.normal(ics.pos.xyz.to(u.pc).value, 100., size=(norbits,3)).T * u.pc
new_vel = np.random.normal(ics.vel.d_xyz.to(u.km/u.s).value, 1., size=(norbits,3)).T * u.km/u.s

new_ics = gd.PhaseSpacePosition(pos=new_pos, vel=new_vel)
orbits = gp.Hamiltonian(pot).integrate_orbit(new_ics, dt=2., n_steps=2000)

grid = np.linspace(-15,15,64)
fig,ax = plt.subplots(1, 1, figsize=(5,5))
fig = pot.plot_contours(grid=(grid,grid,0), cmap='Blues', ax=ax)
fig = orbits[-1].plot(['x', 'y'], color='red', s=1., alpha=0.5, axes=[ax], auto_aspect=False)
