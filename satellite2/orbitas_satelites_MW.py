#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 19:56:52 2020

@author: jordi
"""
import matplotlib.pyplot as plt
import numpy as np
import gala.dynamics as gd
import gala.potential as gp
import plots_jordi as pts
from constants_grav import G
from gala.dynamics.nbody import DirectNBody
from gala.units import galactic, UnitSystem
import astropy.units as u
from scipy.interpolate import interp2d

def potencial_interpolado(nsol, refina = 3, di = 'baja_dens/'):
    rho = np.load("%scoordrho_%d_%d.npy"%(di,nsol,refina))
    z = np.load("%scoordz_%d_%d.npy"%(di,nsol,refina))
    potxz = np.load("%spotxz_%d_%d.npy"%(di,nsol,refina))    
    Vu = interp2d(rho, z, potxz, kind='linear', copy=True, bounds_error=False)
    return Vu  

def fuerza_interpolada(nsol, refina = 3, di = 'baja_dens/'):
    dVdrho = np.load("%sdVdrho_%d_%d.npy"%(di,nsol,refina))
    dVdz = np.load("%sdVdz_%d_%d.npy"%(di,nsol,refina))
    rho = np.load("%scoordrho_%d_%d.npy"%(di,nsol,refina))
    z = np.load("%scoordz_%d_%d.npy"%(di,nsol,refina))
    rhor = rho[:50]
    derrho = interp2d(rhor, z, dVdrho, kind='linear', copy=True, bounds_error=False)
    zr = z[:100]
    derz = interp2d(rho, zr, dVdz, kind='linear', copy=True, bounds_error=False)    
    return derrho, derz

class SFDM_Potential(gp.PotentialBase):
    def __init__(self, A, nsol, rlam, units=None):
        pars = dict(A=A, nsol=nsol, rlam = rlam)
        super(SFDM_Potential, self).__init__(units=units, parameters =pars,
             ndim=3)

    def _energy(self, xyz, t):
        A = self.parameters['A'].value
        nsol = self.parameters['nsol'].value
        rlam = self.parameters['rlam'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        pot_cyl = potencial_interpolado(nsol, refina = 3)
        return A*rlam*rlam*pot_cyl(rho, z)
   
    def _gradient(self, xyz, t):
        A = self.parameters['A'].value
        nsol = self.parameters['nsol'].value
        rlam = self.parameters['rlam'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        grad = np.zeros_like(xyz)
        derrho, derz = fuerza_interpolada(nsol, refina = 3) 
        grad[:,0] = A*rlam*rlam*rlam*derrho(rho,z)*x/rho
        grad[:,1] = A*rlam*rlam*rlam*derrho(rho,z)*y/rho
        grad[:,2] = A*rlam*rlam*rlam*derz(rho, z)
        return grad
    
############                MW satellites Data          ########################
nom = ("Sagittarius", 'LMC','SMC','Draco','Ursa Minor','Sculptor','Sextans','Carina','Fornax','Leo II','Leo I')
a = np.loadtxt('/home/jordi/satellite/MW_sat_pawlowski.txt').T
x_MW_c, y_MW_c, z_MW_c, vx, sigvx, vy, sigvy, vz, sigvz, lpol, bpol, delpol, h, _ = a
#####################            External Pot          ########################
#a = 0.5
#b = 0.5
#c = 1
#dens0 = 1.06e-3
#rs = 12. #kpc   the scale radius.
#vcrs = np.sqrt(4.*np.pi*G*rs**2*dens0*(np.log(2) - 1/2))#v_c   Circular velocity at the scale radius.
##external_pot = gp.LeeSutoTriaxialNFWPotential(vcrs, rs, a, b, c, 
###                                           units=UnitSystem(u.kpc, u.Myr, u.Msun,
###                                                            u.radian, u.km/u.s))
##                                           units = None)
external_pot = gp.MilkyWayPotential()
#nsol = 6
#from pot_paco import DE
#mue = 25
#mu = DE[nsol][mue]['mu']
#rlam = DE[nsol][mue]['rlam']
#lanp = 100./7.5
#rlam2 = rlam/lanp
#external_pot = SFDM_Potential(A=-1., nsol = nsol, rlam = rlam2)
##################           Initial positions          ########################
w = []
for i in range(0,11):
    w.append(gd.PhaseSpacePosition(pos=[x_MW_c[i], y_MW_c[i], z_MW_c[i]] * u.kpc,
                            vel=[vx[i], vy[i], vz[i]] * u.km/u.s))   
#    w.append(gd.PhaseSpacePosition(pos=[x_MW_c[i], y_MW_c[i], z_MW_c[i]],
#                            vel=[vx[i], vy[i], vz[i]]))  
w0 = gd.combine(tuple(w))
##################           Particle potentials          ########################
pot1 = gp.HernquistPotential(m=1e7*u.Msun, c=0.5*u.kpc, units=galactic)
#pot1 = gp.HernquistPotential(m=1e7, c=0.5, units=None)
#particle_pot = [pot1, pot1, pot1, pot1, pot1, pot1, pot1, pot1,
#                pot1, pot1, pot1]
particle_pot = [None, None,None, None,None, None,None, None, None,None, None,]
########################           Solver          #############################
nbody = DirectNBody(w0, particle_pot, external_potential=external_pot)
#orbits = nbody.integrate_orbit(dt=1e-2*u.Myr, t1=0, t2=1*u.Gyr)
orbits = nbody.integrate_orbit(dt=-1e-2*u.Myr, t1=8*u.Gyr, t2=0)
#orbits = nbody.integrate_orbit(dt=-1e-4, t1=2, t2=0)
fs =  orbits.w() 
print(np.shape(fs))
########################           Plots          #############################
fig= plt.figure()
orbits[:, 0].plot(color='green', alpha=0.3)#, axes=[ax1, ax2,ax3])
orbits[:, 1].plot(color='green', alpha=0.3)#,  axes=[ax4, ax5,ax6])
orbits[:, 2].plot(color='green', alpha=0.3)#, axes=[ax7, ax8,ax9])
orbits[:, 3].plot(color='green', alpha=0.3)#, axes=[ax10, ax11,ax12])
orbits[:, 4].plot(color='green', alpha=0.3)#, axes=[ax13, ax14,ax15])
orbits[:, 5].plot(color='green', alpha=0.3)#, axes=[ax16, ax17,ax18])
orbits[:, 6].plot(color='green', alpha=0.3)#, axes=[ax19, ax20,ax21])
orbits[:, 7].plot(color='green', alpha=0.3)#, axes=[ax22, ax23,ax24])
orbits[:, 8].plot(color='green', alpha=0.3)#, axes=[ax25, ax26,ax27])
orbits[:, 9].plot(color='green', alpha=0.3)#, axes=[ax28, ax29,ax30])
orbits[:, 10].plot(color='green', alpha=0.3)#, axes=[ax31, ax32,ax33])
plt.show()
