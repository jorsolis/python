#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:26:28 2019

@author: jordi
"""

import boson_star_shooting_solver_N as bss
from boson_star_shooting_solver_N import r95, write_catidades
import plots_jordi as pts
import numpy as np
import scipy.integrate as spi
from scipy.interpolate import interp1d, interp2d

DI = {0:{'V0_guess' : -1.3418, 'Phi0_guess' : 1., 'E_guess' : -0.70, 'rf':10.},
      1:{'V0_guess' : -1.5035, 'Phi0_guess' : 0.5, 'E_guess' : -0.65, 'rf':10.},
      2:{'V0_guess' : -1.5811, 'Phi0_guess' : 1., 'E_guess' : -0.64, 'rf':15.},
      3:{'V0_guess' : -1.6308, 'Phi0_guess' : 0.8, 'E_guess' : -0.63, 'rf':20.}}

ruta = '/home/jordi/satellite/schrodinger_poisson_shooting_N'

nodes = 0
#
#V_at_length = 0.
#Vprime_at_0 = 0.
#Phi_at_length = 0.
#Phi_prime_at_0 = 0.

r0 = 0.01
rf = DI[nodes]['rf']
stp = 0.01
#
#ye_boundvalue = [V_at_length, Vprime_at_0, Phi_at_length, Phi_prime_at_0, 0.]
#V0_guess = DI[nodes]['V0_guess'] 
#Phi0_guess = DI[nodes]['Phi0_guess'] 
#E_guess= DI[nodes]['E_guess']

#al0 = bss.main(ye_boundvalue, [V0_guess, Phi0_guess, E_guess], 0, r0, rf, stp)
##
#try:
#    Nxmax, N95, r95 = r95(al0, tol = 0.001)
#except TypeError:
#    print('no se calculo r95')
#    Nxmax, N95, r95 = [0,0,0]
    
#bss.plotses(al0[0], al0[1],
#            "Boson Star, $\hat{E}=%.3f, N = %.2f $"%(al0[2], Nxmax),
#            ruta, nodes, sv = True)

#write_catidades(al0, nodes, dec = ruta, N=[Nxmax, N95, r95])

def spher():
    bc = [-1.1207769,  2.27721120e-06,  9.87878294e-01, -2.95821031e-06, -4.79128209e-01,  0.]  
    integrator = spi.ode(bss.rhs).set_integrator("dopri5")
    integrator.set_initial_value(bc, t=r0) 
    al0 = bss.integrate_over_domain(bc, integrator, 0., r0, rf, stp, silent=True)
    np.save("%s/r_0.npy"%(ruta), np.array(al0[0]))
    np.save("%s/V_de_r_0.npy"%(ruta), np.array(al0[1][:,0]))
    np.save("%s/P_de_r_0.npy"%(ruta), np.array(al0[1][:,1]))
#    bss.plotses(al0[0], al0[1], "tit", "ruta", nodes, sv = False) 
    Pot = interp1d(al0[0], al0[1][:,0], kind='linear', copy=True,
                     bounds_error=False , fill_value= 'extrapolate')
    P = interp1d(al0[0], al0[1][:,1], kind='linear', copy=True,
                 bounds_error=False)#, fill_value= 'extrapolate')
    densi = interp1d(al0[0], al0[1][:,2]**2, kind='linear', copy=True,
                     bounds_error=False , fill_value= 'extrapolate')
    
    rho = np.linspace(r0,rf, 200)
    z = np.linspace(-rf,rf, 400)

    derrho = []
    derz = []
    dens = []
    poten = []

    for i in range(0,np.shape(rho)[0]):
        der3 =[]
        der2 = []
        den3 = []
        pot3 = []
        for j in range(0,np.shape(z)[0]):        
            if z[j]>0:
                if rho[i]**2 + z[j]**2 > rf**2:
                    the = np.arctan(z[j]/rho[i])
                    rh = rf*np.cos(the)
                    zh = rf*np.sin(the)
                    der3.append(z[j]*P(np.sqrt(rh**2 + zh**2))/np.sqrt(rh**2 + zh**2)**3)
                    der2.append(rho[i]*P(np.sqrt(rh**2 + zh**2))/np.sqrt(rh**2 + zh**2)**3)
                    den3.append(densi(np.sqrt(rho[i]**2 + z[j]**2)))
                    pot3.append(Pot(np.sqrt(rho[i]**2 + z[j]**2)))
                elif rho[i]**2 + z[j]**2 < r0**2:
                    der3.append(0.)
                    der2.append(0.)
                    den3.append(densi(r0)) 
                    pot3.append(Pot(r0))
                else:
                    der3.append(z[j]*P(np.sqrt(rho[i]**2 + z[j]**2))/np.sqrt(rho[i]**2 + z[j]**2)**3)
                    der2.append(rho[i]*P(np.sqrt(rho[i]**2 + z[j]**2))/np.sqrt(rho[i]**2 + z[j]**2)**3)
                    den3.append(densi(np.sqrt(rho[i]**2 + z[j]**2)))
                    pot3.append(Pot(np.sqrt(rho[i]**2 + z[j]**2)))

            elif z[j]<0:
                if rho[i]**2 + z[j]**2 > rf**2:
                    the = np.arctan(-z[j]/rho[i])
                    rh = rf*np.cos(the)
                    zh = rf*np.sin(the)
                    der3.append(z[j]*P(np.sqrt(rh**2 + zh**2))/np.sqrt(rh**2 + zh**2)**3)
                    der2.append(rho[i]*P(np.sqrt(rh**2 + zh**2))/np.sqrt(rh**2 + zh**2)**3)
                    den3.append(densi(np.sqrt(rho[i]**2 + z[j]**2)))
                    pot3.append(Pot(np.sqrt(rho[i]**2 + z[j]**2)))
                elif rho[i]**2 + z[j]**2 < r0**2:
                    der3.append(0.)
                    der2.append(0.)
                    den3.append(densi(r0))  
                    pot3.append(Pot(r0))
                else:
                    der3.append(z[j]*P(np.sqrt(rho[i]**2 + z[j]**2))/np.sqrt(rho[i]**2 + z[j]**2)**3)                                
                    der2.append(rho[i]*P(np.sqrt(rho[i]**2 + z[j]**2))/np.sqrt(rho[i]**2 + z[j]**2)**3)
                    den3.append(densi(np.sqrt(rho[i]**2 + z[j]**2)))
                    pot3.append(Pot(np.sqrt(rho[i]**2 + z[j]**2)))
        derrho.append(der2)
        dens.append(den3)
        derz.append(der3)
        poten.append(pot3)
    return np.transpose(np.array(derrho)), np.transpose(np.array(derz)), np.transpose(np.array(dens)),np.transpose(np.array(poten))


derrho, derz, dens, poten = spher()
#
#np.save("%s/derrho_%d.npy"%(ruta,nodes), derrho)
#np.save("%s/derz_%d.npy"%(ruta,nodes), derz)
#np.save("%s/dens_%d.npy"%(ruta,nodes), dens)
#np.save("%s/pot_%d.npy"%(ruta,nodes), poten)

#ruta = '/home/jordi/satellite/schrodinger_poisson_shooting_N'
#nodes = 0
#derrho = np.load("%s/derrho_%d.npy"%(ruta,nodes))
#derz = np.load("%s/derz_%d.npy"%(ruta,nodes))
#dens = np.load("%s/dens_%d.npy"%(ruta,nodes)) 
#poten = np.load("%s/pot_%d.npy"%(ruta,nodes)) 
#if np.isnan(derrho).any() == True:
#    print('derrho oui')
#else:
#    print('nel')
#
##    
#rho = np.linspace(r0, rf, 200)
#z = np.linspace(-rf ,rf, 400)
#Rho, Z = np.meshgrid(rho,z)
#
#pts.densityplot(Rho,Z,derrho,r'$\mu \rho$',
#                r'$\mu z$',r'$\frac{\partial \hat{V}}{\mu \partial \rho}$',
#                '',name=None, aspect = '1/2')
#pts.densityplot(Rho,Z,derz,r'$\mu \rho$',r'$\mu z$',
#                r'$\frac{\partial \hat{V}}{\mu \partial z}$','',
#                name=None, aspect = '1/2')
#pts.densityplot(Rho,Z,dens,r'$\mu \rho$',r'$\mu z$',r'$\Phi^2$','',
#                name=None, aspect = '1/2')
#
#pts.densityplot(Rho,Z,poten,r'$\mu \rho$',r'$\mu z$',r'$V$','',
#                name=None, aspect = '1/2')
#
#drho = interp2d(rho, z, derrho, kind='linear', copy=True, bounds_error=False)
#dz = interp2d(rho, z, derz, kind='linear', copy=True, bounds_error=False)
#
#rho = np.linspace(0,2, 500)
#z = np.linspace(-2 ,2, 1000)
#Rho, Z = np.meshgrid(rho,z)
#print(np.shape(drho(rho,z)))
#pts.densityplot(Rho,Z,drho(rho,z),r'$\rho$',r'$z$',r'$\frac{d\psi}{d\rho}$',
#                '',name=None, aspect = '1/2')
#pts.densityplot(Rho,Z,dz(rho,z),r'$\rho$',r'$z$',r'$\frac{d\psi}{dz}$','',
#                name=None, aspect = '1/2')
#


#DE = {21:{'mu': 156550.,'rlam': [1.0e-3]},
#      22:{'mu': 15655.0,'rlam': [1.0e-2]},
#      23:{'mu': 1565.5, 'rlam': [4.0e-3]},
#      24:{'mu': 156.55, 'rlam': [4e-3, 3e-3]},
#      25:{'mu': 15.655, 'rlam': [4e-3, 3e-3]}}
#
#mm = 25
#
#mu = DE[mm]['mu']
##pts.plotmultiple([rho],[derrho[0,:]],[],
##                 r'$\mu \rho$',r'$\frac{\partial \hat{V}}{\mu \partial \rho}$',
##                 'boson star','',save = False)
#cons = 1.65818e12
#lf = 500
#
#rla = DE[mm]['rlam']
#
#for i in range(0, np.shape(rla)[0]):
#    pts.plotmultiple([rho[:lf]/(rla[i]*mu)],
#                      [np.sqrt(rla[i]**2*rho[:lf]*derrho[0,:][:lf])*2.9e5],
#                      [r'$\mu = 10^{-%d}$eV/$c^2$'%mm],r'$\rho$(kpc)','$v$(km/s)',
#                      r'$\sqrt{\lambda}=%f$'%rla[i],'',save = False)
#    pts.plotmultiple([rho/(rla[i]*mu)],
#                      [cons*(mu/1000.)**2*dens(rho)*rla[i]**2],
#                      [r'$\mu = 10^{-%d}$eV/$c^2$'%mm],r'$\rho$(kpc)',
#                      r'$\Phi^2 (\frac{M_\odot}{pc^3})$',
#                      r'$\sqrt{\lambda}=%f$'%rla[i],'',save = False)

