#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:36:24 2020

@author: jordi
"""
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pot_paco_orbitas_analisis import filtro
import plots_jordi as pts
import astropy.units as u
from constants_grav import G

def scape_vel(r, dens0, rs):
    vesc = np.sqrt(8.0*G*np.pi*rs**3*dens0*(-(r/(r + rs)) - np.log(rs) + np.log(r + rs))/r)
    return vesc

def circular_vel(r, rs, dens0):
    vc2 = 4.*np.pi*G*rs**3*dens0*(np.log((rs + r)/rs) - r/(rs + r))/r
    return np.sqrt(vc2)

def mptodos(te=0, carp = 'triaxial', de = 'NFW', kmax= 100,
            ncormin=1,ncormax = 1000, rut = '/home/jordi/satellite/schrodinger_poisson'):
    tho= []; phio= []
    xo2 = np.zeros(kmax*ncormax)
    yo2 = np.zeros(kmax*ncormax)
    zo2 = np.zeros(kmax*ncormax)

    t =  np.load("%s/%spot/%s/1/tiemp_1.npy"%(rut,de,carp))
    pbar = tqdm(total=kmax + 1)
    for k in range(1,kmax + 1,1):
        for ncor in range(ncormin,ncormax +1,1):
            xi,yi,zi,_,_,_ =  np.load("%s/%spot/%s/%d/cords_%d.npy"%(rut,de,carp,k,ncor))
            ind = (k-1)*ncormax + ncor - 1
            xo2[ind] = xi[te]
            yo2[ind] = yi[te]
            zo2[ind] = zi[te]
            thi = np.arccos(zi[te] / np.sqrt(xi[te]**2 + yi[te]**2 + zi[te]**2))
            phii = np.arctan2(yi[te],xi[te])
            tho.append(thi)
            phio.append(phii)
        pbar.update(1)
    pbar.close()
    X2, Y2, Z2 = xo2, yo2, zo2
    TH, Ph = np.array(tho), np.array(phio)   
    np.save("%s/%spot/%s/X_%d.npy"%(rut,de,carp,t[te]), X2)
    np.save("%s/%spot/%s/Y_%d.npy"%(rut,de,carp,t[te]), Y2)
    np.save("%s/%spot/%s/Z_%d.npy"%(rut,de,carp,t[te]), Z2)
    np.save("%s/%spot/%s/Th_%d.npy"%(rut,de,carp,t[te]), TH)
    np.save("%s/%spot/%s/Ph_%d.npy"%(rut,de,carp,t[te]), Ph)

def mptodos_plot(te=0,carp = 'triaxial', de = 'NFW',
                 rut = '/home/jordi/satellite/schrodinger_poisson', 
                 leg = [], histo = False, tresD=False,
                 Fase = False, histo2D = False,
                 astro = False, cord = 'Galactocentric'):    
    print(te)

    t =  np.load("%s/%spot/%s/1/tiemp_1.npy"%(rut,de,carp))
    X = np.load("%s/%spot/%s/X_%d.npy"%(rut,de,carp,t[te]))
    Y = np.load("%s/%spot/%s/Y_%d.npy"%(rut,de,carp,t[te]))
    Z = np.load("%s/%spot/%s/Z_%d.npy"%(rut,de,carp,t[te])) 

    tho = np.load("%s/%spot/%s/Th_%d.npy"%(rut,de,carp,t[te]))
    phio = np.load("%s/%spot/%s/Ph_%d.npy"%(rut,de,carp,t[te]))
    r0 = np.sqrt(X[:]**2 + Y[:]**2 + Z[:]**2)
#    r0, tho, phio = filtro(r0, tho, phio, 500.,'menores a')
    r0, tho, phio = filtro(r0, tho, phio, 300.,'menores a')
#    r0, tho, phio = filtro(r0, tho, phio, [0., 30.],'intervalo')
#    r0, tho, phio = filtro(r0, tho, phio,  [30.,300.],'intervalo')

    if histo == True:  
        pts.histo(tho, r'$\theta$', bins = 80, #rang=(2.,np.pi),
                  nom_archivo ="%s/%spot/%s/hist_th_t%d"%(rut,de,carp,te),
                  logx = False, xangular =True)
        pts.histo(r0, r'$r$(kpc)', bins = 80, logx = False,
                  nom_archivo ="%s/%spot/%s/hist_r_t%d"%(rut,de,carp,te))
    if histo2D ==True:
        pts.histo2d(r0, tho, r'$r$(kpc)', r'$\theta$',
                    bins=[np.linspace(100,350, 20), np.linspace(0,np.pi, 20)],
                    density=True,                      
                    nom_archivo ="%s/%spot/%s/hist_r_th_t%d"%(rut,de,carp,te))
        pts.histo2d(tho,phio,  r'$\theta$',r"$\phi$",
                    bins=[np.linspace(0.,np.pi, 20),
                          np.linspace(-np.pi,np.pi, 20)],
                    cmax = 1000,
                    nom_archivo ="%s/%spot/%s/hist_phi_th_t%d"%(rut,de,carp,te))
    if Fase == True:
        pts.scater(tho,r0,r'$\theta$',r"$r$(kpc)",
                   r'$t=20 \tau_s$',
#                   '$t=$%.2f'%t[te],
                   xangular=True,s=1, color = False, c = r0, zlab = r'$r$(kpc)',
                   ylim = (0, np.amax(r0)),
                   name= "%s/%spot/%s/th_r_t%d"%(rut,de,carp,te))
        pts.scater(tho,phio,r'$\theta$',r"$\phi$",
                   '$t=20 \\tau_s$',
                   xangular=True,s=1, color = False, c = r0, zlab = r'$r$(kpc)',
                   name= "%s/%spot/%s/th_phi_t%d"%(rut,de,carp,te))
        pts.scater(phio, r0, r"$\phi$", r'$r$(kpc)','$t=$%.2f'%t[te],
                   xangular=True, s=1,
#                   name= "%s/%spot_%d/%s/phio_ro_N%d_t%d"%(rut,de,nsol,carp,numpa,te)
                   )

    if tresD==True: 
        X = r0*np.sin(tho)*np.cos(phio)
        Y = r0*np.sin(tho)*np.sin(phio)
        Z = r0*np.cos(tho)
        pts.scater(X,Y,r'$x$(kpc)',r'$y$(kpc)',r'$t = 20\tau_s$', z3D=True, z=Z, zlab=r'$z$(kpc)', 
                   initialview=[0,45], 
                  name = "%s/%spot/%s/particles_%d_1"%(rut,de, carp, te),
                  R = np.amax(np.sqrt(X**2 + Y**2 +Z**2)),
                  s = 5)  
        pts.scater(X,Y,r'$x$(kpc)',r'$y$(kpc)',r'$t = 20\tau_s$', z3D=True,
                   z=Z, zlab=r'$z$(kpc)', 
                   initialview=[45,45],
                  name = "%s/%spot/%s/particles_%d_2"%(rut,de, carp,te),
                  R = np.amax(np.sqrt(X**2 + Y**2 +Z**2)),
                  s = 5)    
    if astro==True: 
        fig = plt.figure(figsize=(20,6))
        ax = fig.add_subplot(111, projection="mollweide")
        if cord == 'Galactocentric':
            lat = np.pi/2. - tho  
            lon = phio            
#            ax.scatter(lon_MW, lat_MW,c=r_MW,
#    #                   c = 'r',
#                       marker= 'X', s = 100 )            
        elif cord == 'Heliocentric':        
            X = r0*np.sin(tho)*np.cos(phio)
            Y = r0*np.sin(tho)*np.sin(phio)
            Z = r0*np.cos(tho)        
            galact = SkyCoord(x=X, y=Y, z=Z, unit='kpc', frame= 'galactocentric',
                              representation_type='cartesian')
            Hcrs = galact.transform_to(coord.HCRS)
            Hcrs.representation_type = 'spherical'             
            lon = Hcrs.ra.wrap_at(180*u.deg).radian
            lat = Hcrs.dec.radian
        
        ax.set_title("%s coords"%cord)    
        sc = ax.scatter(lon, lat, s=1, c=r0) 
        plt.colorbar(sc).ax.set_ylabel(r'$r$(kpc)')   
        ax.grid()
        ax.set_xlabel('lon $l$')
        ax.set_ylabel('lat $b$')
        if cord == 'Galactocentric':
            plt.savefig("%s/%spot/%s/particles_galcto_t%d.png"%(rut,de,carp,te), dpi=100, bbox_inches='tight')
        else:
            plt.savefig("%s/%spot/%s/particles_helio_t%d.png"%(rut,de,carp,te), dpi=100, bbox_inches='tight')        
        plt.show()            
def mptodos_l(te=0, carp = 'triaxial', de = 'NFW', kmax= 100,
            ncormin=1,ncormax = 1000, rut = '/home/jordi/satellite/schrodinger_poisson'):
    print(te)
    lt = []; lf = []
    Lx =[];Ly=[];Lz=[]
    t =  np.load("%s/%spot/%s/1/tiemp_1.npy"%(rut,de,carp))
    pbar = tqdm(total=kmax + 1)
    for k in range(1,kmax + 1,1):
        for ncor in range(ncormin,ncormax + 1,1):
            lx,ly,lz = np.load("%s/%spot/%s/%d/L_%d.npy"%(rut,de,carp,k,ncor))
            lth = np.arccos(lz/np.sqrt(lx**2 + ly**2 + lz**2))
            lphi = np.arctan2(ly,lx)
            lt.append(lth[te])
            lf.append(lphi[te])
            Lx.append(lx[te])
            Ly.append(ly[te])
            Lz.append(lz[te])
        pbar.update(1)
    pbar.close()
    LTH, LPh = np.array(lt), np.array(lf)  
    LX, LY,LZ = np.array(Lx), np.array(Ly), np.array(Lz)
    np.save("%s/%spot/%s/LTh_%d.npy"%(rut,de,carp,t[te]), LTH)
    np.save("%s/%spot/%s/LPh_%d.npy"%(rut,de,carp,t[te]), LPh)
    np.save("%s/%spot/%s/LX_%d.npy"%(rut,de,carp,t[te]), LX)
    np.save("%s/%spot/%s/LY_%d.npy"%(rut,de,carp,t[te]), LY)
    np.save("%s/%spot/%s/LZ_%d.npy"%(rut,de,carp,t[te]), LZ)        

def mptodos_plot_l(te=0, carp = 'triaxial', de ='NFW',
                  rut = '/home/jordi/satellite/schrodinger_poisson', 
                  histo = False, tresD=False,histo2D =False,
                  Fase = False, astro = False, cord = 'Heliocentric'):    
    print(te)
    t =  np.load("%s/%spot/%s/1/tiemp_1.npy"%(rut,de,carp))
    LTH = np.load("%s/%spot/%s/LTh_%d.npy"%(rut,de,carp,t[te]))
    LPh = np.load("%s/%spot/%s/LPh_%d.npy"%(rut,de,carp,t[te]))
#    LX = np.load("%s/%spot/%s/LX_%d.npy"%(rut,de,carp,t[te]))
#    LY = np.load("%s/%spot/%s/LY_%d.npy"%(rut,de,carp,t[te]))
#    LZ = np.load("%s/%spot/%s/LZ_%d.npy"%(rut,de,carp,t[te]))

    X = np.load("%s/%spot/%s/X_%d.npy"%(rut,de,carp,t[te]))
    Y = np.load("%s/%spot/%s/Y_%d.npy"%(rut,de,carp,t[te]))
    Z = np.load("%s/%spot/%s/Z_%d.npy"%(rut,de,carp,t[te])) 
    r0 = np.sqrt(X[:]**2 + Y[:]**2 + Z[:]**2)

#    r0, LTH, LPh = filtro(r0, LTH, LPh, 500.,'menores a')
#    r0, LTH, LPh = filtro(r0, LTH, LPh, [0., 30.],'intervalo')
    r0, LTH, LPh = filtro(r0, LTH, LPh,  [30.,300.],'intervalo')  
#    LX = np.sin(LTH)*np.cos(LPh)
#    LY = np.sin(LTH)*np.sin(LPh)
#    LZ = np.cos(LTH)
#        
#    if histo == True:  
#        pts.histo(LTH, r'$\theta_l$', bins = 80, #rang=(2.,np.pi),
#                  nom_archivo ="%s/%spot_%d/%s/hist_l_th_%d_t%d"%(rut,de,nsol,carp,nsol,te),
##                  fit = True, #dist = 'dweibull',
##                  normalized=False,
#                  logx = False, xangular =True)
#        pts.histo(LPh, r'$\phi_l$', bins = 80, 
##                  rang=(0,2),
#                  nom_archivo ="%s/%spot_%d/%s/hist_l_phi_%d_t%d"%(rut,de,nsol,carp,nsol,te),
##                  fit = True,
##                  normalized=False,
#                  logx = False, xangular =True)
#    if histo2D ==True:
#        pts.histo2d(LTH, LPh,  r'$\theta$',r"$\phi$",
#                    bins=[np.linspace(0.,np.pi, 20),
#                          np.linspace(-np.pi,np.pi, 20)],
#                    cmax = 10000,
#                    nom_archivo ="%s/%spot_%d/%s/hist_l_phi_th_%d_t%d"%(rut,de,nsol,carp,nsol,te))

    if Fase == True:
        pts.scater(LTH,LPh,r'$\theta_l$',r"$\phi_l$",
                   r'$t=20 \tau_s$',
#                   '$t=$%.2f'%t[te],
                   xangular=True,s=1, angular =True, ylim = (-np.pi, np.pi),
#                   color = True, c=r0, clab = r'$r$',
                   name= "%s/%spot/%s/lth_lph_t%d"%(rut,de,carp,te),
#                   extra_data = True, x_extra = lth, y_extra = lphi,
#                   extra_err = True, xerr_extra = siglth, yerr_extra = siglphi
                   )
#    if astro == True:
#        l_cart = SkyCoord(x=LX, y=LY, z=LZ, unit='kpc',frame= 'galactocentric',
#                          representation_type='cartesian')
#        if cord == 'Galactocentric':
#            l_cart.representation_type = 'spherical'
#        elif cord == 'Heliocentric':
#            Hcrs = l_cart.transform_to(coord.HCRS)
#            Hcrs.representation_type = 'spherical'       
#        fig = plt.figure(figsize=(20,6))
#        ax = fig.add_subplot(111, projection="aitoff")
#        ax.set_title("Test Particles")
#        cx = [];  cy = []; amplitudes=[]
#        for i in range(0,np.shape(X)[0]):
#            if cord == 'Galactocentric':
#                cx.append(l_cart[i].lon.wrap_at(180*u.deg).radian)
#                cy.append(l_cart[i].lat.radian)
#            elif cord == 'Heliocentric':
#                cx.append(Hcrs[i].ra.wrap_at(180*u.deg).radian)
#                cy.append(Hcrs[i].dec.radian)
#            amplitudes.append(r0[i])
#        sc = ax.scatter(cx,cy, c=amplitudes, s=20)
#        plt.colorbar(sc).ax.set_ylabel(r'$r$(kpc)')  
#        ax.grid()
#        ax.set_xlabel('lon $l$')
#        ax.set_ylabel('lat $b$')
#        plt.savefig("%s/%spot_%d/%s/l_%s_%d_t%d"%(rut,de,nsol,carp,cord,nsol,te), dpi=100, bbox_inches='tight')
#        plt.show() 
#        
#    if tresD==True: 
#        plt.close()
#        pts.scater(LX,LY,r'$x$',r'$y$',r'$t = 20\tau_s$', z3D=True, z=LZ, 
#                   zlab=r'$z$', initialview=[0,45], 
#                  name = "%s/%spot_%d/%s/l_pot%d_%d_1"%(rut,de,nsol, carp, nsol,te),
#                  R = 1, s = 1#, color = True, c=r0, clab =r'$r$'
#                  )  
#        pts.scater(LX,LY,r'$x$',r'$y$',r'$t = 20\tau_s$', z3D=True, z=LZ,
#                   zlab=r'$z$', initialview=[45,45],
#                  name = "%s/%spot_%d/%s/l_pot%d_%d_2"%(rut,de,nsol, carp,nsol,te),
#                  R = 1., s = 1#,  color = True, c=r0, clab = r'$r$'
#                  )    
def taus_t(te= 0,carp = 'triaxial', de ='NFW',kmax = 100, nump = 1000,
           rut = '/home/jordi/satellite/schrodinger_poisson'):
    T =[]
    pbar = tqdm(total=kmax + 1)
    for k in range(1,kmax + 1, 1):
        for ncor in range(1, nump+1):
            tau =np.load("%s/%spot/%s/%d/tau_%d.npy"%(rut,de,carp,k,ncor))
            T.append(tau[te])
        pbar.update(1)
    pbar.close()
    T = np.array(T)
    return T        

if __name__ == '__main__':
    print('Calcular taus')
    rut = "/home/jordi/satellite/schrodinger_poisson"
    di = 'NFW'
    carp = 'triaxial'##
    nump = 1000
###############################################################################
######################     TORSION          ###################################
###############################################################################
#    t =  np.load("%s/%spot/%s/1/tiemp_1.npy"%(rut,di,carp))
#    dt = t[1]-t[0]
#    dt2 = dt*dt
#    kmax = 100
#    pbar = tqdm(total=kmax + 1)
#    for k in range(1,kmax + 1,1):
#        for ncor in range(1, nump + 1,1):
#            xi,yi,zi,vxi, vyi, vzi =  np.load("%s/%spot/%s/%d/cords_%d.npy"%(rut,di,carp,k,ncor))
#            v = np.array([vxi[:-2], vyi[:-2], vzi[:-2]])
#            vxdoti = []
#            vydoti = []
#            vzdoti = []
#            for i in range(0, np.shape(vxi)[0] - 1):
#                vxdoti.append((vxi[i + 1] - vxi[i])/dt)
#                vydoti.append((vyi[i + 1] - vyi[i])/dt)
#                vzdoti.append((vzi[i + 1] - vzi[i])/dt)
#            vxdoti = np.array(vxdoti[:-1])
#            vydoti = np.array(vydoti[:-1])
#            vzdoti = np.array(vzdoti[:-1])          
#            vxdot2i = []
#            vydot2i = []
#            vzdot2i = []        
#            for i in range(0, np.shape(vxi)[0] - 2):
#                vxdot2i.append((vxi[i + 2] - 2.*vxi[i + 1] + vxi[i])/dt2)
#                vydot2i.append((vyi[i + 2] - 2.*vyi[i + 1] + vyi[i])/dt2)
#                vzdot2i.append((vzi[i + 2] - 2.*vzi[i + 1] + vzi[i])/dt2)
#            vxdot2i = np.array(vxdot2i)
#            vydot2i = np.array(vydot2i)
#            vzdot2i = np.array(vzdot2i)   
#            
#            vx, vy , vz = v
#            alpha = vy*vzdoti - vz*vydoti
#            beta = vz*vxdoti - vx*vzdoti
#            gama = vx*vydoti - vy*vxdoti
#            tau = (alpha*vxdot2i + beta*vydot2i + gama*vzdot2i)/(alpha**2 + beta**2 + gama**2)
#            np.save("%s/%spot/%s/%d/tau_%d.npy"%(rut,di,carp,k,ncor), tau)
#        pbar.update(1)
#    pbar.close()