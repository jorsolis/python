#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 22:47:45 2020

@author: jordi
"""
import matplotlib.pyplot as plt
import numpy as np
import gala.dynamics as gd
import gala.potential as gp
from time import time
from constants_grav import G
from triaxialNFW_analisis import (mptodos_plot, mptodos,scape_vel,taus_t,
                                  mptodos_plot_l,mptodos_l)
import plots_jordi as pts

def triax_NFW_density(pars, x,y,z):
#  pars: (alpha = 1)
#            0 - G
#            1 - v_c
#            2 - r_s
#            3 - a
#            4 - b
#            5 - c
    b_a2 = pars[4]*pars[4] / (pars[3]*pars[3]);
    c_a2 = pars[5]*pars[5] / (pars[3]*pars[3]);
    e_b2 = 1-b_a2;
    e_c2 = 1-c_a2;
    v_h2 = pars[1]*pars[1] / (np.log(2.) - 0.5 + (np.log(2.)-0.75)*e_b2 + (np.log(2.)-0.75)*e_c2);
    u = np.sqrt(x*x + y*y/b_a2 + z*z/c_a2) / pars[2];
    return v_h2 / (u * (1+u)*(1+u)) / (4.*np.pi*pars[2]*pars[2]*pars[0])


def principal(tiem, ncor, y0a, v0a, v_c=1.0, r_s = 12.0, a=1.0, b=1.0, c=1.,
              plotting=True,stps = 3000,
              direct = "/home/jordi/satellite/schrodinger_poisson/NFWpot/triaxial",
              de = '', k = 1):
    potdm = gp.LeeSutoTriaxialNFWPotential(v_c, r_s, a, b, c, 
#                                           units=UnitSystem(u.kpc, u.Myr, u.Msun,
#                                                            u.radian, u.km/u.s))
                                           units = None)
#    if plotting==True:
#        R = 300
##        plt.figure(figsize=(6,5))
##        X = np.linspace(-R,R)
##        potdm.plot_density_contours((X, X), ax = plt.axes())
#
#        pars = [G, vcrs, rs, a, b, c]
#        x, y = np.meshgrid(np.linspace(-R,R, 100), np.linspace(-R,R, 100))
#        z = 0.
#        f = triax_NFW_density(pars, x,y,z)
#        pts.densityplot(x,y,f,r'$x$[kpc]',r'$y$[kpc]',
#                        r'$\rho(x,y,0)[10^{10}M_\odot/\rm{kpc}^3]$',
#                        'Mass density', name="%s/dens_xy"%(direct), aspect='1/1',
#                log = True, rang = False, rango = [1e-7, 1.], show = True)    
#        x, z = np.meshgrid(np.linspace(-R,R, 100), np.linspace(-R,R, 100))
#        y = 0.
#        f = triax_NFW_density(pars, x,y,z)
#        pts.densityplot(x,z,f,r'$x$[kpc]',r'$z$[kpc]',r'$\rho(x,0,z)[10^{10}M_\odot/\rm{kpc}^3]$',
#                        'Mass density', name="%s/dens_xz"%(direct), aspect='1/1',
#                log = True, rang = False, rango = [1e-7, 1], show = True)         
#        y, z = np.meshgrid(np.linspace(-R,R, 100), np.linspace(-R,R, 100))
#        x = 0.
#        f = triax_NFW_density(pars, x,y,z)
#        pts.densityplot(y,z,f,r'$y$[kpc]',r'$z$[kpc]',r'$\rho(0,y,z)[10^{10}M_\odot/\rm{kpc}^3]$',
#                        'Mass density', name="%s/dens_yz"%(direct), aspect='1/1',
#                log = True, rang = False, rango = [1e-7, 1.], show = True)  
        
    def resolvedor2(y0a, v0a, du, stps):    
        w0 = gd.PhaseSpacePosition(pos=y0a, vel=v0a)
        orbit = gp.Hamiltonian(potdm).integrate_orbit(w0, dt=du, n_steps=stps)  
        if plotting==True:
            orbit.plot(marker=',', linestyle='none')
#            fig2 = orbit.plot(components = ['v_x', 'v_y', 'v_z'] ,marker=',', linestyle='none')   
            plt.savefig("%s/traj_%d"%(direct, ncor),transparent=False)
        w =  orbit.w() 
#        return w
        E = orbit.energy().value
        L = orbit.angular_momentum().value
        return w, E,L   
    du = tiem/stps
    uev = np.arange(0., tiem, du)
#    fs = resolvedor2(y0a, v0a, du, stps)
#    return uev, fs
    fs, E, L = resolvedor2(y0a, v0a, du, stps)
    return uev, fs, E, L

       
if __name__ == '__main__':
    time0 = time()
    rut = "/home/jordi/satellite/schrodinger_poisson"
    di = 'NFW'
    carp = 'triaxial'##
################           Paramettros Potencial      ##########################
##    a = 1.0 #######   a>=b>=c
##    b = 1.0
##    c = 0.5
    a = 0.5
    b = 0.5
    c = 1
    dens0 = 1.06e-3
    rs = 12. #kpc   the scale radius.
    vcrs = np.sqrt(4.*np.pi*G*rs**2*dens0*(np.log(2) - 1/2))#v_c   Circular velocity at the scale radius.
    print(vcrs)
###############################################################################   
#    tiemp = 200    
    nump= 1001 
    vesc = scape_vel(150, dens0, rs)
##    print(vesc)
    ri  = 0.2
    rf = 300.
    vmax = vesc/2    
#########           Escala de tiempo       ######################################
    tiemp= 10.
    vphi0 = vcrs
    rho0 = rs
    y0a = np.array([rho0, 0., 0.])#estan en cartesianas
    v0a = np.array([0., vphi0, 0.])#estan en cartesianas
    ncor = 1  
    t, _,E,L = principal(tiemp, ncor, y0a, v0a, v_c=vcrs, r_s = rs, a=a, b=b, 
                     c=c, plotting = True, de=di)
    plt.figure()
    plt.plot(t,E[:-1])
    plt.plot(t,L[2,:-1])
    plt.show()
    r = rs*3.086e19 ## km
    v = vcrs
    print(2.*np.pi*r/v/3.14e7/1e9, 'Gyrs')
###############################################################################
#    da2 = float(input('k'))
#    k = int(da2)
##    np.random.seed(12345)       
#    print(k)
#    ncor = 1
    
##    VX = []; VY = [];VZ = []
##    for k in range(0,101):
#
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
#
##            VX.append(vx0);VY.append(vy0);VZ.append(vz0)
##    
##    VX = np.array(VX); VY = np.array(VY);VZ = np.array(VZ)
##    pts.scater(VX,VY,r'$v_x$(km/s)',r'$v_y$(km/s)',r'$t = 0\tau_s$', z3D=True,
##               z=VZ, zlab=r'$v_z$(km/s)', 
##               initialview=[0,45], 
##              name = "%s/%spot/%s/pVarticles_%d_1"%(rut,di, carp, 0),
##              R = np.amax(np.sqrt(VX**2 + VY**2 +VZ**2)),
##              s = 5)  
##    pts.scater(VX,VY,r'$v_x$(km/s)',r'$v_y$(km/s)',r'$t = 0\tau_s$', z3D=True,
##               z=VZ, zlab=r'$v_z$(km/s)', 
##               initialview=[45,45],
##              name = "%s/%spot/%s/Vparticles_%d_2"%(rut,di, carp, 0),
##              R = np.amax(np.sqrt(VX**2 + VY**2 +VZ**2)),
##              s = 5)              
#    #######      condiciones  iniciales
#        y0a = np.array([x0, y0, z0])
#        v0a = np.array([vx0, vy0, vz0])
#    ########      Programa   
##        t, cord = principal(tiemp, ncor, y0a, v0a, v_c=vcrs, r_s = rs, a=a, b=b,
##                            c=c, plotting = False ,de=di, k = k,stps = 5000)
#        t, cord, E,L = principal(tiemp, ncor, y0a, v0a, v_c=vcrs, r_s = rs, a=a, b=b,
#                            c=c, plotting = False ,de=di, k = k,stps = 5000)
#        np.save("%s/%spot/%s/%d/cords_%d.npy"%(rut,di,carp,k,ncor), cord)
#        np.save("%s/%spot/%s/%d/E_%d.npy"%(rut,di,carp,k,ncor), E)
#        np.save("%s/%spot/%s/%d/L_%d.npy"%(rut,di,carp,k,ncor), L)
#        np.save("%s/%spot/%s/%d/tiemp_%d.npy"%(rut,di,carp,k,ncor), t)
#        ncor += 1
#
#    timef = time()
#    print(timef-time0,'sec')    

##############################################################################
##############     plots fase space       ##################################
################################# ##############################################
#    for i in range(0, 3000, 10000):     
#        print(i)
#        time0 = time()
#        mptodos(te= i)
#        timef = time()
#        print(timef-time0,'sec')  
#        mptodos_plot(te= i, 
#                      histo = True, 
##                      histo2D = True,
#                      tresD=True,
##                      astro = True, #cord = 'Heliocentric',
##                      Fase = True
#                 )
#        mptodos_l(te= i)
#        mptodos_plot_l(te= i,
##                      histo = True, 
##                      histo2D = True,
##                      tresD=True,
#                 Fase = True,
##                 astro = True, cord = 'Galactocentric'
#                 )
#
##
#        T = taus_t(te= i)    
#        np.save("%s/%spot/%s/Tau_all_%d.npy"%(rut,di,carp,i), T)
#
#        T =  np.load("%s/%spot/%s/Tau_all_%d.npy"%(rut,di,carp,i))
#        t =  np.load("%s/%spot/%s/1/tiemp_1.npy"%(rut,di,carp))
#        pts.histo(T, r'$\tau$', 
#                  rang=(-0.04,0.04), 
#                  bins = 200,
##                  title = r'$t = 20\tau_s$',
##                  title = r'$t = 0$',
#                  title = r'$t = %f $'%t[i],
#                  normalized = False,
#                  nom_archivo = "%s/%spot/%s/hist_tau_t%d"%(rut,di,carp,i))
