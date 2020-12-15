#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 22:47:45 2020

@author: jordi
"""
import matplotlib.pyplot as plt
import numpy as np
import plots_jordi as pts
import galpy.potential as gpp

from time import time

def principal(Rf, tiem, ncor, y0a, v0a, amp=1.0, a=2.0, b=1.0, c=1.5,
              plotting=True,stps = 3000,
              direct = "/home/jordi/satellite/schrodinger_poisson/potpaco",
              de = '', k = 1):
    potdm = gpp.TriaxialNFWPotential(amp=amp, a=a, b=b, c=c, zvec=None,
                                    pa=None, normalize=False, conc=None, 
                                    mvir=None, glorder=50, vo=None, ro=None,
                                    H=70.0, Om=0.3, overdens=200.0,
                                    wrtcrit=False)
    
    def resolvedor2(y0a, v0a, du, stps):    
        w0 = gd.PhaseSpacePosition(pos=y0a, vel=v0a)
        orbit = gp.Hamiltonian(potdm).integrate_orbit(w0, dt=du, n_steps=stps)  
        figs = orbit.plot(marker=',', linestyle='none')
#        sixplots_orbits(orbit[0:1000])    
        w =  orbit.w()  
        return w
    
    du = tiem/stps
    uev = np.arange(0., tiem, du)
    fs = resolvedor2(y0a, v0a, du, stps)
    return uev, fs

if __name__ == '__main__':
    time0 = time()
    rut = "/home/jordi/satellite/schrodinger_poisson"
    di = 'NFW/'
    carp = 'triaxial'##
    
#    tiemp = 3200    
    nump= 2    
    vesc = 1. #########            REVISAR
    ri  = 0.2
    rf = 10.
    vmax = vesc/2

    a = 1.0
    b = 1.0
    c = 1.
    
#######           Escala de tiempo       ######################################
    tiemp= 60.
    vphi0 = vmax/2.
    rho0 = 4.
    y0a = np.array([rho0, 0., 0.])#estan en cartesianas
    v0a = np.array([0., vphi0, 0.])#estan en cartesianas
    ncor = 1
    _, _ = principal(rf, tiemp, ncor, y0a, v0a, a=a, b=b, c=c,
                     v_c=1.0, r_s = 10.0,
                     plotting = False, de=di)
#############################################################################
#    da2 = float(input('k'))
#    k = int(da2)
#    np.random.seed(12345)       
#    print(k)
#    ncor = 1
#    for i in range(1,nump,1):
#    ########      esfera de posiciones aleatorias de radio rf 
#        r0 = np.random.uniform(ri, rf)
#        th0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#        phi0 = np.random.uniform(0.,2.*np.pi) 
#    ########      esfera de velocidades aleatorias de radio vmax  
#        vr0 = np.random.uniform(0., vmax)
#        vt0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#        vf0 = np.random.uniform(0.,2.*np.pi)   
#    ########      posiciones iniciales
#        x0 = r0*np.sin(th0)*np.cos(phi0)
#        y0 = r0*np.sin(th0)*np.sin(phi0)
#        z0 = r0*np.cos(th0)       
#    ########      velocidades iniciales
#        vx0 = vr0*np.sin(vt0)*np.cos(vf0)
#        vy0 = vr0*np.sin(vt0)*np.sin(vf0)
#        vz0 = vr0*np.cos(vt0)
#    #######      condiciones  iniciales
#        y0a = np.array([x0, y0, z0])
#        v0a = np.array([vx0, vy0, vz0])
#    ########      Programa   
#        t, cord = principal(rf, tiemp, ncor, y0a,v0a,
#                            plotting = True ,de=di, k = k)
#        np.save("%s/%spot/%s/%d/cords_%d.npy"%(rut,di,carp,k,ncor), cord)
#        np.save("%s/%spot/%s/%d/tiemp_%d.npy"%(rut,di,carp,k,ncor), t)
#        ncor += 1
#
#    timef = time()
#    print(timef-time0,'sec')    

##############################################################################
##############     plots fase space       ##################################
################################# ##############################################
#    for i in range(975, 1000, 50):     
#        print(nsol, i)
#        mu = 25
##        mptodos(nsol,te= i,carp = carp,rut = rut, de=di, kmax=100, mue= mu)
#        mptodos_plot(nsol,te= i, Rf = rf, carp = carp, de=di, mue= mu,
#                      rut = rut, 
##                      histo = True, 
##                      histo2D = True,
##                      tresD=True,
#                      astro = True, #cord = 'Heliocentric',
##                      Animation= True,
##                 Fase = True
#                 )
##        mptodos_l(nsol,te= i,carp = carp,rut = rut, de=di, kmax=100)
##        mptodos_plot_l(nsol,te= i, carp = carp, de=di, rut = rut,
###                      histo = True, 
###                      histo2D = True,
###                      tresD=True,
##                 Fase = True,
###                 astro = True, cord = 'Galactocentric'
##                 )