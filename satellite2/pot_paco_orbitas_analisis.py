#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:14:34 2020

@author: jordi
"""
import numpy as np
from pot_paco import DE
import plots_jordi as pts
from plots_orbitas_schrodinger_poisson import scatter3d_animation
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from MW_class_satellite_position import MW_orbital_poles

nom = ("Sagittarius", 'LMC','SMC','Draco','Ursa Minor','Sculptor','Sextans','Carina','Fornax','Leo II','Leo I')
nomsatMW = ['Sagittarius', 'LMC','SMC','Draco','Ursa Minor','Sculptor','Sextans','Carina','Fornax','Leo II','Leo I']
a = np.loadtxt('/home/jordi/satellite/MW_sat_pawlowski.txt').T
x_MW_c, y_MW_c, z_MW_c, vx, sigvx, vy, sigvy, vz, sigvz, lpol, bpol, delpol, h, _ = a


r_MW =  np.sqrt(x_MW_c**2 + y_MW_c**2 + z_MW_c**2)
lon_MW = np.arctan2(y_MW_c, x_MW_c)
lat_MW = np.arcsin(z_MW_c/r_MW)

def mptodos(nsol,te=0, mue = 25, carp = 'orbitas_random_vel', de = '', kmax= 100,
            ncormin=1,ncormax = 1000, rut = ''):
#    mu = DE[nsol][mue]['mu']
#    rlam = DE[nsol][mue]['rlam']
#    lanp = 100./7.5
    
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/mixSFDM/'%dirdata
    popt = np.load('%spopt_nsol%d_%s.npy'%(dirfitsG, nsol, 'v5'))
    rlam, mu, Md, ad, Mb, bb = popt      
#    rlam, mu = 1., 1.
    lanp= 1.
 
    xo2= []
    yo2= []
    zo2= []
    tho= []
    phio= []
    t =  np.load("%s/%spot_%d/%s/1/tiemp_1.npy"%(rut,de,nsol,carp))
    pbar = tqdm(total=kmax + 1)
    for k in range(1,kmax + 1,1):
        for ncor in range(ncormin,ncormax ,1):
#            if nsol == 6:###     Pot 6
#                xi,yi,zi =  np.load("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,de,nsol,carp,k, ncor))      
#            elif nsol==2:###     Pot 2
#                xi,yi,zi,_,_,_ =  np.load("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,de,nsol,carp,k, ncor))
            
            xi,yi,zi,_,_,_ =  np.load("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,de,nsol,carp,k, ncor))
            
            xo2.append(lanp*xi[te]/(rlam*mu))
            yo2.append(lanp*yi[te]/(rlam*mu))
            zo2.append(lanp*zi[te]/(rlam*mu))
            thi = np.arccos(zi[te] / np.sqrt(xi[te]**2 + yi[te]**2 + zi[te]**2))
            phii = np.arctan2(yi[te],xi[te])
            tho.append(thi)
            phio.append(phii)
        pbar.update(1)
    pbar.close()     
    X2, Y2, Z2 = np.array(xo2), np.array(yo2), np.array(zo2)
    TH, Ph = np.array(tho), np.array(phio)   
    np.save("%s/%spot_%d/%s/X_%d.npy"%(rut,de,nsol,carp,t[te]), X2)
    np.save("%s/%spot_%d/%s/Y_%d.npy"%(rut,de,nsol,carp,t[te]), Y2)
    np.save("%s/%spot_%d/%s/Z_%d.npy"%(rut,de,nsol,carp,t[te]), Z2)
    np.save("%s/%spot_%d/%s/Th_%d.npy"%(rut,de,nsol,carp,t[te]), TH)
    np.save("%s/%spot_%d/%s/Ph_%d.npy"%(rut,de,nsol,carp,t[te]), Ph)

def filtro(r0, tho, phio, value, dejamos):
    mask = []
    if dejamos == 'menores a':
        for i in range(0, np.shape(r0)[0]):
            if r0[i]>value:
                mask.append(1) ## quitamos estos
            else:
                mask.append(0)## dejamos estos
    elif dejamos == 'mayores a':
        for i in range(0, np.shape(r0)[0]):
            if r0[i]<value:
                mask.append(1) ## quitamos estos
            else:
                mask.append(0)## dejamos estos  
    elif dejamos == 'intervalo':
        for i in range(0, np.shape(r0)[0]):
            if r0[i]<value[0]:
                mask.append(1) ## quitamos estos
            elif r0[i]>value[1]:
                mask.append(1) ## quitamos estos           
            else:
                mask.append(0)## dejamos estos  
    nr0 = np.ma.masked_array(r0, mask = mask, fill_value=np.nan)
    ntho = np.ma.masked_array(tho, mask = mask, fill_value=np.nan)
    nphio = np.ma.masked_array(phio, mask = mask, fill_value=np.nan)
    return nr0, ntho, nphio

def mptodos_plot(nsol,te=0,Rf = 8,mue= 25, carp = 'orbitas_random_vel', de = '',
                 rut = '', leg = [], histo = False, tresD=False,
                 Fase = False, histo2D = False, Animation = False, 
                 astro = False, cord = 'Galactocentric'):    
    print(te)
#    ro, zet, fie = np.load("%s/%spot_%d/rhozfi.npy"%(rut,de,nsol)) 
    t =  np.load("%s/%spot_%d/%s/1/tiemp_1.npy"%(rut,de,nsol,carp))
    X = np.load("%s/%spot_%d/%s/X_%d.npy"%(rut,de,nsol,carp,t[te]))
    Y = np.load("%s/%spot_%d/%s/Y_%d.npy"%(rut,de,nsol,carp,t[te]))
    Z = np.load("%s/%spot_%d/%s/Z_%d.npy"%(rut,de,nsol,carp,t[te])) 
    
#    mu = DE[nsol][mue]['mu']
#    rlam = DE[nsol][mue]['rlam']
#    lanp = 100./7.5    
#    ro, zet = lanp*ro/(rlam*mu), lanp*zet/(rlam*mu) 
#    cons = 1.65818e12
#    fie = cons*(mu/1000.)**2*rlam**2*fie/lanp**2

    tho = np.load("%s/%spot_%d/%s/Th_%d.npy"%(rut,de,nsol,carp,t[te]))
    phio = np.load("%s/%spot_%d/%s/Ph_%d.npy"%(rut,de,nsol,carp,t[te]))
    r0 = np.sqrt(X[:]**2 + Y[:]**2 + Z[:]**2)

#    r0, tho, phio = filtro(r0, tho, phio, 750.,'menores a')
#    r0, tho, phio = filtro(r0, tho, phio, 500.,'menores a')
#    r0, tho, phio = filtro(r0, tho, phio, 400.,'menores a')
    r0, tho, phio = filtro(r0, tho, phio, [0., 30.],'intervalo')
#    r0, tho, phio = filtro(r0, tho, phio,  [30.,300.],'intervalo')
#    r0, tho, phio = filtro(r0, tho, phio,  [30.,280.],'intervalo')

    if histo == True:  
        pts.histo(tho, r'$\theta$', bins = 80, #rang=(2.,np.pi),
                  nom_archivo ="%s/%spot_%d/%s/hist_th_%d_t%d"%(rut,de,nsol,carp,nsol,te),
#                  fit = True, #dist = 'dweibull',
#                  normalized=False,
                  logx = False, xangular =True)
        pts.histo(r0, r'$r$(kpc)', bins = 80, 
#                  rang=(0,2),
                  nom_archivo ="%s/%spot_%d/%s/hist_r_%d_t%d"%(rut,de,nsol,carp,nsol,te),
#                  fit = True,
#                  normalized=False,
                  logx = False)
    if histo2D ==True:
        pts.histo2d(r0, tho, r'$r$(kpc)', r'$\theta$',
                    bins=[np.linspace(100,350, 20), np.linspace(0,np.pi, 20)],
                    density=True,                      
                    nom_archivo ="%s/%spot_%d/%s/hist_r_th_%d_t%d"%(rut,de,nsol,carp,nsol,te))
        pts.histo2d(tho,phio,  r'$\theta$',r"$\phi$",
                    bins=[np.linspace(0.,np.pi, 20),
                          np.linspace(-np.pi,np.pi, 20)],
                    cmax = 1000,
                    nom_archivo ="%s/%spot_%d/%s/hist_phi_th_%d_t%d"%(rut,de,nsol,carp,nsol,te))
    if Fase == True:
        pts.scater(tho,r0,r'$\theta$',r"$r$(kpc)",
                   r'$t=20 \tau_s$',
#                   '$t=$%.2f'%t[te],
                   xangular=True,s=1, color = False, c = r0, zlab = r'$r$(kpc)',
                   ylim = (0, 30),
#                   ylim=(0, lanp*Rf/(rlam*mu)),
                   name= "%s/%spot_%d/%s/th_r_%d_t%d"%(rut,de,nsol,carp,nsol,te))
        pts.scater(tho,phio,r'$\theta$',r"$\phi$",
#                   '$t=$%.2f'%t[te],
                   '$t=20 \\tau_s$',
                   xangular=True,s=1, color = False, c = r0, zlab = r'$r$(kpc)',
                   name= "%s/%spot_%d/%s/th_phi_%d_t%d"%(rut,de,nsol,carp,nsol,te))
        pts.scater(phio, r0, r"$\phi$", r'$r$(kpc)','$t=$%.2f'%t[te],
                   xangular=True, s=1,
#                   ylim=(0, lanp*Rf/(rlam*mu)),
#                   name= "%s/%spot_%d/%s/phio_ro_N%d_t%d"%(rut,de,nsol,carp,numpa,te)
                   )

    if tresD==True: 
        X = r0*np.sin(tho)*np.cos(phio)
        Y = r0*np.sin(tho)*np.sin(phio)
        Z = r0*np.cos(tho)
        print('Rmax = ',np.amax(np.sqrt(X**2 + Y**2 +Z**2)))
#        plt.close()
        pts.scater(X,Y,r'$x$(kpc)',r'$y$(kpc)',
                   '$t=$%.2f'%t[te],
#                   r'$t = 20\tau_s$', 
                   z3D=True, z=Z, zlab=r'$z$(kpc)', initialview=[0,45], 
                  name = "%s/%spot_%d/%s/particles_pot%d_%d_1"%(rut,de,nsol, carp, nsol,te),
#                  R = lanp*Rf/(rlam*mu),
                  R = np.amax(np.sqrt(X**2 + Y**2 +Z**2)),
                  s = 1)  
        pts.scater(X,Y,r'$x$(kpc)',r'$y$(kpc)',
                   r'$t = 20\tau_s$',
                   z3D=True, z=Z, zlab=r'$z$(kpc)', initialview=[45,45],
                  name = "%s/%spot_%d/%s/particles_pot%d_%d_2"%(rut,de,nsol, carp,nsol,te),
#                  R = lanp*Rf/(rlam*mu),
                  R = np.amax(np.sqrt(X**2 + Y**2 +Z**2)),
                  s = 1)    
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
#        xedg = [-np.pi, -5.*np.pi/6, -2.*np.pi/3., -np.pi/2., -np.pi/3., -np.pi/6., 0., np.pi/6., np.pi/3., np.pi/2., 2.*np.pi/3., 5.*np.pi/6, np.pi]
#        yedg = [-np.pi/2., -5.*np.pi/12., -np.pi/3., -np.pi/4., -np.pi/6., -np.pi/12., 0.,
#                np.pi/12., np.pi/6., np.pi/4., np.pi/3., 5.*np.pi/12., np.pi/2.]
#        h, xedges, yedges, ax = plt.hist2d(lon,lat, bins = [xedg, yedg],
#                                           density = True)
#        print(xedges/np.pi)
#        print(yedges*2./np.pi)
#        print(h)
        ax.grid()
        ax.set_xlabel('lon $l$')
        ax.set_ylabel('lat $b$')
        if cord == 'Galactocentric':
            plt.savefig("%s/%spot_%d/%s/particles_galcto_t%d.png"%(rut,de,nsol,carp,te), dpi=100, bbox_inches='tight')
        else:
            plt.savefig("%s/%spot_%d/%s/particles_helio_t%d.png"%(rut,de,nsol,carp,te), dpi=100, bbox_inches='tight')        
        plt.show()            


        
    if Animation == True:   
        scatter3d_animation(X,Y,Z,'$x$','$y$','$z$','$t=$%.2f'%t[te],
                            "%s/%spot_%d/%s/haz_%d.mp4"%(rut,de,nsol,carp,te),
                            galaxy=False, 
#                            R = lanp*Rf/(rlam*mu), 
                            R=600,
                            elevado=False)  
def mptodos_l(nsol,te=0, carp = 'orbitas_random_vel', de = '', kmax= 100,
            ncormin=1,ncormax = 1000, rut = ''):
    lt = []
    lf = []
    Lx =[];Ly=[];Lz=[]
    t =  np.load("%s/%spot_%d/%s/1/tiemp_1.npy"%(rut,de,nsol,carp))
    pbar = tqdm(total=kmax + 1)
    for k in range(1,kmax + 1,1):
        for ncor in range(ncormin,ncormax ,1):
            lth = np.load("%s/%spot_%d/%s/%d/ang_mom_th_%d.npy"%(rut,de,nsol,carp,k,ncor))
            lphi = np.load("%s/%spot_%d/%s/%d/ang_mom_ph_%d.npy"%(rut,de,nsol,carp,k,ncor))
            lx,ly,lz = np.load("%s/%spot_%d/%s/%d/ang_mom_%d.npy"%(rut,de,nsol,carp,k,ncor))
            lt.append(lth[te])
            lf.append(lphi[te])
            Lx.append(lx[te])
            Ly.append(ly[te])
            Lz.append(lz[te])
        pbar.update(1)
    pbar.close()
    LTH, LPh = np.array(lt), np.array(lf)  
    LX, LY,LZ = np.array(Lx), np.array(Ly), np.array(Lz)
    np.save("%s/%spot_%d/%s/LTh_%d.npy"%(rut,de,nsol,carp,t[te]), LTH)
    np.save("%s/%spot_%d/%s/LPh_%d.npy"%(rut,de,nsol,carp,t[te]), LPh)
    np.save("%s/%spot_%d/%s/LX_%d.npy"%(rut,de,nsol,carp,t[te]), LX)
    np.save("%s/%spot_%d/%s/LY_%d.npy"%(rut,de,nsol,carp,t[te]), LY)
    np.save("%s/%spot_%d/%s/LZ_%d.npy"%(rut,de,nsol,carp,t[te]), LZ)
    
def mptodos_plot_l(nsol,te=0, carp = 'orbitas_random_vel', de = '',
                  rut = '', histo = False, tresD=False,histo2D =False,
                  Fase = False, astro = False, cord = 'Heliocentric'):    
    print(te)
    t =  np.load("%s/%spot_%d/%s/1/tiemp_1.npy"%(rut,de,nsol,carp))
    LTH = np.load("%s/%spot_%d/%s/LTh_%d.npy"%(rut,de,nsol,carp,t[te]))
    LPh = np.load("%s/%spot_%d/%s/LPh_%d.npy"%(rut,de,nsol,carp,t[te]))
    LX = np.load("%s/%spot_%d/%s/LX_%d.npy"%(rut,de,nsol,carp,t[te]))
    LY = np.load("%s/%spot_%d/%s/LY_%d.npy"%(rut,de,nsol,carp,t[te]))
    LZ = np.load("%s/%spot_%d/%s/LZ_%d.npy"%(rut,de,nsol,carp,t[te]))

    X = np.load("%s/%spot_%d/%s/X_%d.npy"%(rut,de,nsol,carp,t[te]))
    Y = np.load("%s/%spot_%d/%s/Y_%d.npy"%(rut,de,nsol,carp,t[te]))
    Z = np.load("%s/%spot_%d/%s/Z_%d.npy"%(rut,de,nsol,carp,t[te])) 
    r0 = np.sqrt(X[:]**2 + Y[:]**2 + Z[:]**2)
    print(np.shape(r0))
    print(np.shape(LTH))
#    r0, LTH, LPh = filtro(r0, LTH, LPh, 500.,'menores a')
#    r0, LTH, LPh = filtro(r0, LTH, LPh, [0., 30.],'intervalo')
#    r0, LTH, LPh = filtro(r0, LTH, LPh,  [30.,300.],'intervalo')  
    LX = np.sin(LTH)*np.cos(LPh)
    LY = np.sin(LTH)*np.sin(LPh)
    LZ = np.cos(LTH)
# 
    lx, siglx, ly, sigly, lz,siglz, lth, siglth, lphi, siglphi, b, sigb = MW_orbital_poles()

    if histo == True:  
        pts.histo(LTH, r'$\theta_l$', bins = 80, #rang=(2.,np.pi),
                  nom_archivo ="%s/%spot_%d/%s/hist_l_th_%d_t%d"%(rut,de,nsol,carp,nsol,te),
#                  fit = True, #dist = 'dweibull',
#                  normalized=False,
                  logx = False, xangular =True)
        pts.histo(LPh, r'$\phi_l$', bins = 80, 
#                  rang=(0,2),
                  nom_archivo ="%s/%spot_%d/%s/hist_l_phi_%d_t%d"%(rut,de,nsol,carp,nsol,te),
#                  fit = True,
#                  normalized=False,
                  logx = False, xangular =True)
    if histo2D ==True:
        pts.histo2d(LTH, LPh,  r'$\theta$',r"$\phi$",
                    bins=[np.linspace(0.,np.pi, 20),
                          np.linspace(-np.pi,np.pi, 20)],
                    cmax = 10000,
                    nom_archivo ="%s/%spot_%d/%s/hist_l_phi_th_%d_t%d"%(rut,de,nsol,carp,nsol,te))

    if Fase == True:
        pts.scater(LTH,LPh,r'$\theta_l$',r"$\phi_l$",
#                   r'$t=20 \tau_s$',
                   '$t=$%.2f'%t[te],
                   xangular=True,s=1, angular =True, ylim = (-np.pi, np.pi),
#                   color = True, c=r0, clab = r'$r$',
                   name= "%s/%spot_%d/%s/lth_lph_%d_t%d"%(rut,de,nsol,carp,nsol,te),
                   extra_data = True, x_extra = lth, y_extra = lphi,
                   extra_err = True, xerr_extra = siglth, yerr_extra = siglphi,
                   extratext = True, texts = nomsatMW)
    if astro == True:
        l_cart = SkyCoord(x=LX, y=LY, z=LZ, unit='kpc',frame= 'galactocentric',
                          representation_type='cartesian')
        l_MW = SkyCoord(lphi, b, frame= 'galactocentric',
                          representation_type='spherical', unit = 'rad')
        sigl_MW = SkyCoord(siglphi, sigb, frame= 'galactocentric',
                          representation_type='spherical', unit = 'rad')  

        if cord == 'Galactocentric':
            l_cart.representation_type = 'spherical'
        elif cord == 'Heliocentric':
            Hcrs = l_cart.transform_to(coord.HCRS)
            Hcrs.representation_type = 'spherical'
            
        fig = plt.figure(figsize=(20,6))
        ax = fig.add_subplot(111, projection="aitoff")
        ax.set_title("Orbital Poles")
        cx = [];  cy = []
        for i in range(0,np.shape(X)[0]):
            if cord == 'Galactocentric':
                cx.append(l_cart[i].lon.wrap_at(180*u.deg).radian)
                cy.append(l_cart[i].lat.radian)
            elif cord == 'Heliocentric':
                cx.append(Hcrs[i].ra.wrap_at(180*u.deg).radian)
                cy.append(Hcrs[i].dec.radian)

        sc = ax.scatter(cx,cy, c=r0, s=20)
        plt.colorbar(sc).ax.set_ylabel(r'$r$(kpc)')  
#        ax.scatter(cx,cy, s=20)
        if cord == 'Galactocentric':
            font = {'family': 'serif', 'color':  'red',
                    'weight': 'bold','size': 18}
            for i in range(0, np.shape(lx)[0]):
                ax.text(l_MW[i].lon.wrap_at(180*u.deg).radian,
                        l_MW[i].lat.radian, nomsatMW[i],fontdict=font)
            ax.errorbar(l_MW.lon.wrap_at(180*u.deg).radian,
                       l_MW.lat.radian,
                       xerr = sigl_MW.lon.wrap_at(180*u.deg).radian,
                       yerr = sigl_MW.lat.radian, 
                        fmt='o',alpha = 0.95,c = 'r',
                         ms = 2.5, capsize = 2.5, elinewidth = 2.5)
        ax.grid()
        ax.set_xlabel('lon $l$')
        ax.set_ylabel('lat $b$')
        plt.savefig("%s/%spot_%d/%s/l_%s_%d_t%d"%(rut,de,nsol,carp,cord,nsol,te), dpi=100, bbox_inches='tight')
        plt.show() 
        
    if tresD==True: 
        plt.close()
        pts.scater(LX,LY,r'$x$',r'$y$',r'$t = 20\tau_s$', z3D=True, z=LZ, 
                   zlab=r'$z$', initialview=[0,45], 
                  name = "%s/%spot_%d/%s/l_pot%d_%d_1"%(rut,de,nsol, carp, nsol,te),
                  R = 1, s = 1#, color = True, c=r0, clab =r'$r$'
                  )  
        pts.scater(LX,LY,r'$x$',r'$y$',r'$t = 20\tau_s$', z3D=True, z=LZ,
                   zlab=r'$z$', initialview=[45,45],
                  name = "%s/%spot_%d/%s/l_pot%d_%d_2"%(rut,de,nsol, carp,nsol,te),
                  R = 1., s = 1#,  color = True, c=r0, clab = r'$r$'
                  )    
        