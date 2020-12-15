#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:01:22 2019

@author: jordi
"""
import numpy as np
from scipy.interpolate import interp1d#, interp2d
import matplotlib.pyplot as plt

import plots_jordi as pts
from pot_paco_orbitas_analisis import filtro    
import gala.potential as gp
import gala.dynamics as gd
from time import time
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord

nsol = 1
tol = 0.01
Rf = 10

ruta = '/home/jordi/satellite/schrodinger_poisson_shooting_N'

rnum = np.load("%s/r_0.npy"%(ruta))
Vnum = np.load("%s/V_de_r_0.npy"%(ruta))
Pnum = np.load("%s/P_de_r_0.npy"%(ruta))

Pot = interp1d(rnum, Vnum, kind='linear', copy=True,
                     bounds_error=False , fill_value= (Vnum[0] ,Vnum[-1]/2.))
der = interp1d(rnum, Pnum, kind='linear', copy=True,
                     bounds_error=False , fill_value= (Pnum[0] ,Pnum[-1]))

class SFDM_Potential(gp.PotentialBase):
    def __init__(self,A, units=None):
        pars = dict(A=A)
        super(SFDM_Potential, self).__init__(units=units,parameters =pars,ndim=3)

    def _energy(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        r = np.sqrt(x**2 + y**2 + z**2)
        return A*Pot(r)
   
    def _gradient(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        r = np.sqrt(x**2 + y**2 + z**2)
        grad = np.zeros_like(xyz) 
        grad[:,0] = A*x*der(r)/r**3
        grad[:,1] = A*y*der(r)/r**3
        grad[:,2] = A*z*der(r)/r**3
        return grad
    
def principal(potdm, Rf, tiem, stps, y0a=[], plotting=True,
              direct = "/home/jordi/satellite/schrodinger_poisson/potpaco"):
    du = tiem/stps
    uev = np.arange(0., tiem, du)  
    x0, y0, z0, vx0, vy0, vz0 = y0a       
    w0 = gd.PhaseSpacePosition(pos=[x0,y0,z0], vel=[vx0,vy0,vz0])
    orbit = gp.Hamiltonian(potdm).integrate_orbit(w0, dt=du, n_steps=stps)  
    if plotting ==True:
        figs = orbit.plot(marker=',', linestyle='none')
#        plt.show()
#            1sixplots_orbits(orbit[0:1000])    
    w =  orbit.w()  
    t = uev
    x, y, z, vx,vy,vz = w
    R2 = np.sqrt( x**2 + y**2 + z**2)
    if np.any(R2>12)==True:
        print(i,'r max !!!!!')
        print('rmax=', np.amax(R2))
        print('r(0)=', R2[0])        
    return t, w

rut = "/home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens/potcore"
DU = {2:{'t':600, 'rf':8}}#para boson star shooting
###############################################################################
###############        Random positions & velocities        ###################
###############################################################################

nsol = 2
rf = DU[nsol]['rf']
tiemp= DU[nsol]['t'] * 4.

ncor = 1
nump= 1000
#
ri  = 0.1 


##vesc = np.sqrt(0.5) # = sqrt(2 N(8)/8) 
#da2 = float(input('k='))
#k = int(da2)
##carpeta = 'orbitas_random_vel_new'
##
###nump = 3
###print('seed '*10)
###np.random.seed(12345)   
##
##print(k)
#stps = 2000
# 
#
##c = 2.99e5 #km/s
##print('vescape/c= ', vesc)
########           Escala de tiempo       ######################################
##vr0 = vesc/2.
##vt0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
##vf0 = np.random.uniform(0.,2.*np.pi)    
##vrho0 = vr0*np.sin(vt0)*np.cos(vf0)
##vphi0 = vr0*np.sin(vt0)*np.sin(vf0)
##vz0 = vr0*np.cos(vt0)
##tiemp = tiemp/20.
##vphi0 = vesc/2.
##y0= [0., 4., 0., 0., 0., vphi0] #vrho, rho, vz, z, phi, vphi
##k = 1
##t, cord = principal(potdm, rf, tiemp,  stps, y0a=y0, plotting = True,
##                                       direct = rut)
################################################################################
#POS=[]
#for i in range(0,nump,1):
##########      esfera de posiciones aleatorias de radio rf 
#    r0 = np.random.uniform(ri, rf)
#    th0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#    phi0 = np.random.uniform(0.,2.*np.pi) 
#########      posiciones iniciales
#    z0 = r0*np.cos(th0)   
#    x0 = r0*np.sin(th0)*np.cos(phi0)
#    y0 = r0*np.sin(th0)*np.sin(phi0)   
#########      esfera de velocidades aleatorias de radio vrf  
#    vrf = vesc/2.
#    vr0 = np.random.uniform(0., vrf)
#    vt0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#    vf0 = np.random.uniform(0.,2.*np.pi)    
##    ########      velocidades iniciales
#    vx0 = vr0*np.sin(vt0)*np.cos(vf0)
#    vy0 = vr0*np.sin(vt0)*np.sin(vf0)
#    vz0 = vr0*np.cos(vt0)
#    pos = np.array([x0, y0,z0,vx0, vy0, vz0])
#    POS.append(pos)
#POS = np.array(POS)
#np.save("%s/%s/%d/POS.npy"%(rut,carpeta,k), POS)
#
#potdm = SFDM_Potential(A=1.)
#time0 = time()
#for i in range(1,nump+1,1):
#    POS = np.load("%s/%s/%d/POS.npy"%(rut,carpeta,k))##con1diciones  iniciales
#    y0 = POS[i-1,:]  
#    t, cord = principal(potdm, rf, tiemp, stps, y0a=y0, plotting = False,
#                        direct = rut)
#    np.save("%s/%s/%d/cords_%d.npy"%(rut,carpeta,k,ncor), cord)
##    np.save("%s/%s/%d/tiemp_%d.npy"%(rut,carpeta,k,ncor), t)
#    ncor += 1
#timef = time()
#print(timef-time0,'sec')         

###############################################################################
#############           Disco        ########################
###############################################################################
#carp = 'orbitas_disc_dist'
##nump= 6528
#nump = 10
#x0 = np.load("%s/%s/X_0.npy"%(rut,carp))
#y0 = np.load("%s/%s/Y_0.npy"%(rut,carp))
#z0 = np.load("%s/%s/Z_0.npy"%(rut,carp))
#r0 = np.sqrt(x0**2 + y0**2 + z0**2)
#phi0 = np.arctan2(y0,z0)
#vz0 = np.zeros(np.shape(x0))
#vphi0 = np.sqrt(der(r0)/r0)
#vx0 = -vphi0*np.sin(phi0)
#vy0 = vphi0*np.cos(phi0)
#potdm = SFDM_Potential(A=1.)
#
#for i in range(1,nump+1,1):
#    y0 = [x0[i-1], y0[i-1], z0[i-1], vx0[i-1], vy0[i-1], vz0[i-1]] 
#    t, cord = principal(potdm, rf, tiemp, stps, y0a=y0, plotting = True,
#                        direct = rut)
#    np.save("%s/%s/%d/cords_%d.npy"%(rut,carp,k,ncor), cord)
##    np.save("%s/%s/%d/tiemp_%d.npy"%(rut,carpeta,k,ncor), t)
#    ncor += 1
###############################################################################
#####################               Plots           ###########################
############################################################################
DE = {2: {24:{'mu' : 156.55, 'rlam' : 3.8e-3, 'limb' : 90},
            25:{'mu' : 15.655, 'rlam' : 1.7e-2, 'limb' : 501, 'ref': 3}}}

def mptodos(te=0, carp = 'orbitas_random_vel_new', kmax = 100):
    mue = 25
    mu = DE[2][mue]['mu']
    rlam = DE[2][mue]['rlam']
    lanp = 100./10.
    xo2= []
    yo2= []
    zo2= []
    t =  np.load("%s/%s/1/tiemp_1.npy"%(rut,carp))
    print(te, t[te])
    for k in range(1,kmax +1,1):
        for ncor in range(1,1001 ,1):
            xi,yi,zi,_,_,_ =  np.load("%s/%s/%d/cords_%d.npy"%(rut,carp,k, ncor))      
            xo2.append(lanp*xi[te]/(rlam*mu))
            yo2.append(lanp*yi[te]/(rlam*mu))
            zo2.append(lanp*zi[te]/(rlam*mu))
    X2, Y2, Z2 = np.array(xo2), np.array(yo2), np.array(zo2)
    np.save("%s/%s/X_%d.npy"%(rut, carp, t[te]), X2)
    np.save("%s/%s/Y_%d.npy"%(rut, carp, t[te]), Y2)
    np.save("%s/%s/Z_%d.npy"%(rut, carp, t[te]), Z2)

def mptodos_plot(te=0, carp = 'orbitas_random_vel_new', histo = False,
                 Fase = False, histo2D = False, astro = False, tresD =False,
                 cord = 'Galactocentric'):    
    print(te)
    t =  np.load("%s/%s/1/tiemp_1.npy"%(rut,carp))
    X = np.load("%s/%s/X_%d.npy"%(rut, carp, t[te]))
    Y = np.load("%s/%s/Y_%d.npy"%(rut, carp, t[te]))
    Z = np.load("%s/%s/Z_%d.npy"%(rut, carp, t[te])) 
    mue = 25
    mu = DE[2][mue]['mu']
    rlam = DE[2][mue]['rlam']
    lanp = 100./10.    

    r0 = np.sqrt(X[:]**2 + Y[:]**2 + Z[:]**2)
    tho = np.arccos(Z/r0)
    phio = np.arctan2(Y,X)
 
    r0, tho, phio = filtro(r0, tho, phio, 400.,'menores a')
#    r0, tho, phio = filtro(r0, tho, phio, [0., 30.],'intervalo')
#    r0, tho, phio = filtro(r0, tho, phio,  [30.,300.],'intervalo') 
    if histo == True: 
        pts.histo(tho, r'$\theta$', bins = 80, #rang=(2.,np.pi),
                  nom_archivo ="%s/%s/hist_core_tho_t%d"%(rut,carp,te),
    #                  fit = True, #dist = 'dweibull',
    #                normalized=True,
                  logx = False, xangular =True)
        pts.histo(r0, r'$r$(kpc)', bins = 80, normalized=False,
                  nom_archivo ="%s/%s/hist_core_r_t%d"%(rut,carp,te))

 
    if histo2D ==True:
        pts.histo2d(r0, tho, r'$r$(kpc)', r'$\theta$',
                    bins=[np.linspace(100,350, 20), np.linspace(0,np.pi, 20)],
                    density=True,                      
                    nom_archivo ="%s/%s/hist_core_r_th_t%d"%(rut,carp,te))
        pts.histo2d(tho,phio,  r'$\theta$',r"$\phi$",
                    bins=[np.linspace(0.,np.pi, 20),
                          np.linspace(-np.pi,np.pi, 20)],
                    cmax = 1000,
                    nom_archivo ="%s/%s/hist_core_phi_th_t%d"%(rut,carp,te))
    if Fase == True:
        pts.scater(tho,r0,r'$\theta$',r"$r$(kpc)",
                   r'$t=20 \tau_s$',
    #               '$t=$%.2f'%t[te],
                   xangular=True, ylim=(0,400),
                   name= "%s/%s/core_th_r_t%d"%(rut,carp,te))
        pts.scater(tho, phio,r'$\theta$', r"$\phi$", 
                   r'$t=20 \tau_s$',                   
#                   '$t=$%.2f'%t[te], 
                   xangular=True,# angular =True,
                   name= "%s/%s/core_th_phi_t%d"%(rut,carp,te))   
   
    if astro==True: 
        X = r0*np.sin(tho)*np.cos(phio)
        Y = r0*np.sin(tho)*np.sin(phio)
        Z = r0*np.cos(tho)        
        galact = SkyCoord(x=X, y=Y, z=Z, unit='kpc', frame= 'galactocentric',
                          representation_type='cartesian')
        if cord == 'Galactocentric':
            galact.representation_type = 'spherical'
        elif cord == 'Heliocentric':
            Hcrs = galact.transform_to(coord.HCRS)
            Hcrs.representation_type = 'spherical'       
        fig = plt.figure(figsize=(20,6))
        ax = fig.add_subplot(111, projection="aitoff")
        ax.set_title("Test Particles")
        cx = [];  cy = []; amplitudes=[]
        for i in range(0,np.shape(X)[0]):
            if cord == 'Galactocentric':
                cx.append(galact[i].lon.wrap_at(180*u.deg).radian)
                cy.append(galact[i].lat.radian)
            elif cord == 'Heliocentric':
                cx.append(Hcrs[i].ra.wrap_at(180*u.deg).radian)
                cy.append(Hcrs[i].dec.radian)
            amplitudes.append(r0[i])
        sc = ax.scatter(cx,cy, c=amplitudes, s=20)
        plt.colorbar(sc).ax.set_ylabel(r'$r$(kpc)')      
        ax.grid()
        ax.set_xlabel('lon $l$')
        ax.set_ylabel('lat $b$')
        if cord == 'Galactocentric':
            nombre = "%s/%s/particles_galcto_t%d"%(rut,carp,te)
        elif cord == 'Heliocentric':
            nombre = "%s/%s/particles_helio_t%d"%(rut,carp,te)
        plt.savefig(nombre, dpi=100, bbox_inches='tight')
        plt.show()  
    if tresD==True: 
        x00 = r0*np.sin(tho)*np.cos(phio)
        y00 = r0*np.sin(tho)*np.sin(phio)
        z00 = r0*np.cos(tho)   
        pts.scater(x00,y00,r'$x$(kpc)',r'$y$(kpc)',r'$10^{5}$ Particles',
                   z3D=True, z=z00,
                   zlab=r'$z$(kpc)', initialview=[0,45], R = 300,
                   name = '%s/%s/Particles_core_t%d'%(rut,carp,te)) #initialview=[45,-60])


for i in range(1975,1999, 50000):  
#    mptodos(te=i, kmax = 100)
    mptodos_plot(te=i, 
#                 histo = True,
#                 Fase = True, 
#                 histo2D = False,
                tresD=True,
#                 astro = True, cord = 'Heliocentric'
                 )

 

#import matplotlib.animation as animation
#from matplotlib.animation import FFMpegWriter
#def exportdata():
#    numpa = 99900
#    carp = 'orbitas_random_vel'
#    t =  np.load("%s/%s/1/tiemp_1.npy"%(rut,carp))    
#    data = []    
#    data2 = []
#    data3 = []
#    #
#    #for te in range(0,999,25):
#    for te in range(0,999,10):
#        tho = np.load("%s/%s/Th_%d.npy"%(rut, carp, t[te]))
#        phio = np.load("%s/%s/Ph_%d.npy"%(rut, carp, t[te]))
#        X = np.load("%s/%s/X_%d.npy"%(rut, carp, t[te]))
#        Y = np.load("%s/%s/Y_%d.npy"%(rut, carp, t[te]))
#        Z = np.load("%s/%s/Z_%d.npy"%(rut, carp, t[te])) 
#        r0 = np.sqrt(X[:numpa]**2 + Y[:numpa]**2 + Z[:numpa]**2)    
#        tho = tho[:numpa]
#        phio = phio[:numpa] 
#        data.append([tho,phio])
#        data2.append([tho,r0])
#        data3.append([phio,r0])    
#    data3 = np.array(data3)
#    data2 = np.array(data2)
#    data = np.array(data)
#    return data, data2, data3
##
#def update_plot(i, data, scat):
##    scat.set_array(data[i])
#    scat.set_offsets(data[i][:].T)
#    return scat,
#
#def video(data,xlabel, ylabel,title='prueba', fps = 3, dpi = 500, numframes = 39):
#    xy_data = data  
#    fig = plt.figure(figsize=(7,5))
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    scat = plt.scatter(data[0][0], data[0][1], marker=',', s=1)
#    ani = animation.FuncAnimation(fig, update_plot, frames= range(numframes),
#                                  fargs=(xy_data, scat))
#    writer = FFMpegWriter(fps = fps)
#    ani.save('%s.mp4'%title, writer=writer, dpi = dpi) 
#    plt.show()
#
##_, data2, data3 = exportdata()
##video(data,r'$\theta$',r'$\phi$', title="%s/orbitas_random_vel/tho_phio"%(rut),
##      numframes = 100, fps = 10)
##video(data2,r'$\theta$',r'$r$(kpc)', title="%s/orbitas_random_vel/tho_ro"%(rut),
##      numframes = 100, fps = 10)
##video(data3,r'$\phi$',r'$r$(kpc)', title="%s/orbitas_random_vel/phio_ro"%(rut),
##      numframes = 100, fps = 10)
