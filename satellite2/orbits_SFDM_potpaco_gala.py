#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:22:50 2019

@author: jordi
"""
import numpy as np
import gala.potential as gp
import gala.dynamics as gd
from pot_paco import potential_interpolado,fuerza_interpolada
from gala.units import galactic

nsol = 2
refina = 3
VXZ = np.loadtxt("Vxz_%d_%d.txt"%(nsol,refina))
pot_cyl = potential_interpolado(VXZ)
derrho, derz = fuerza_interpolada(VXZ)

class SFDM_Potential(gp.PotentialBase):
#    def __init__(self,A, units=None):
    def __init__(self,A, units=galactic):
        pars = dict(A=A)
        super(SFDM_Potential, self).__init__(units=units,parameters =pars,ndim=3)

    def _energy(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        return A*pot_cyl(rho, z)

    def _gradient(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        grad = np.zeros_like(xyz)
        grad[:,0] = A*derrho(rho,z)*x/rho
        grad[:,1] = A*derrho(rho,z)*y/rho
        grad[:,2] = A*derz(rho, z)
        return grad

def sixplots_orbits(orbit):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)    
    ax2.set_xlim(-0.4, 0.4)
    ax2.set_ylim(-2, 2)
    ax4.set_xlim(-1., 1.)
    ax5.set_xlim(-1., 1.)
    ax6.set_xlim(-1., 1.)
    fig = orbit.plot(['x', 'y'], color='green', alpha=0.3, axes=[ax1])
    fig = orbit.plot(['x', 'z'], color='green', alpha=0.3, axes=[ax2])
    fig = orbit.plot(['y', 'z'], color='green', alpha=0.3, axes=[ax3])
    fig = orbit.plot(['t', 'v_x'], color='green', alpha=0.3, axes=[ax4])
    fig = orbit.plot(['t', 'v_y'], color='green', alpha=0.3, axes=[ax5])
    fig = orbit.plot(['t', 'v_z'], color='green', alpha=0.3, axes=[ax6])
    
#potdm = SFDM_Potential(A=-1.,units=None)
potdm = SFDM_Potential(A=-1.,units=galactic)

rho0 = 0.3
phi0 = np.pi/2.
z0 = -1.98
vrho0 = 0.
vz0 = 0.
l = 0.1

#rho0= 1.
#z0 = 0.

x0 = rho0*np.cos(phi0)
y0 = rho0*np.sin(phi0)
vx0 = vrho0*np.cos(phi0) - l*np.sin(phi0)/rho0
vy0 = vrho0*np.sin(phi0) + l*np.cos(phi0)/rho0

w0 = gd.PhaseSpacePosition(pos=[x0,y0,z0], vel=[vx0,vy0,vz0])
orbit = gp.Hamiltonian(potdm).integrate_orbit(w0, dt=0.05, n_steps=400)


figs = orbit.plot(marker=',', linestyle='none')
sixplots_orbits(orbit[0:100])
w =  orbit.w(units=galactic)
print(np.shape(w))
#import astropy.coordinates as coord
#import matplotlib.pyplot as plt
#import astropy.units as u
#
#icrs = orbit.to_coord_frame(coord.ICRS)
#print(icrs[-1])
#fig, axes = plt.subplots(1, 1, figsize=(12, 15),
#                         subplot_kw={'projection': 'aitoff'})
#plt.grid()
#axes.set_title("ICRS")
#axes.plot(icrs.ra.wrap_at(180*u.deg).radian, icrs.dec.radian,
#             linestyle='none', marker='.', markersize= .5)
#plt.show()
#
#galac = orbit.to_coord_frame(coord.Galactic)
##print(icrs[-1])
#fig, axes = plt.subplots(1, 1, figsize=(12, 15),
#                         subplot_kw={'projection': 'aitoff'})
#plt.grid()
#axes.set_title("Galactic")
#axes.plot(galac.l.wrap_at(180*u.deg).radian, galac.b.radian,
#             linestyle='none', marker='.', markersize= .1)
#plt.show()