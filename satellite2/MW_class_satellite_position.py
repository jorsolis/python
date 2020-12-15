#!/usr/bin/env python
# coding: utf-8
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
import numpy as np
#from plots_orbitas_schrodinger_poisson import scatter3d_animation
import plots_jordi as pts
#import cartopy.crs as ccrs
"""          Galactic cartesian coordinates
       The Milky Way’s Disk of Classical Satellite Galaxies in Light of Gaia DR2
        M. S. Pawlowski & P. Kroupa
    o de From  Marcel S. Pawlowski and Pavel Kroupa MNRAS 435, 2116–2131 (2013)
"""

def MW_orbital_poles():
    from uncertainties import unumpy
    a = np.loadtxt('/home/jordi/satellite/MW_sat_pawlowski.txt').T
    x_MW_c, y_MW_c, z_MW_c, vx, sigvx, vy, sigvy, vz, sigvz, lpol, bpol, delpol, h, _ = a
    VX = unumpy.uarray(vx, sigvx)
    VY = unumpy.uarray(vy, sigvy)
    VZ = unumpy.uarray(vz, sigvz)
    X = unumpy.uarray(x_MW_c, 0.)
    Y = unumpy.uarray(y_MW_c, 0.)
    Z = unumpy.uarray(z_MW_c, 0.)    
    LX = Y*VZ - Z*VY
    LY = Z*VX - X*VZ
    LZ = X*VY - Y*VX
    norm = unumpy.sqrt(LX**2 + LY**2 + LZ**2)
    Lxn = LX/norm
    Lyn = LY/norm
    Lzn = LZ/norm
    lthn = unumpy.arccos(Lzn)
    lphin = unumpy.arctan2(Lyn,Lxn)
    b = np.pi/2. - lthn
    SigLth = []; SigLphi=[]; LTH=[]; LPH=[]; LXN =[]; LYN =[]; LZN =[];
    SigLXN = []; SigLYN=[]; SigLZN = []; B= []; SigB = []
    for i in range(0, np.shape(vx)[0]):
        LTH.append(lthn[i].n)
        LPH.append(lphin[i].n)
        SigLth.append(lthn[i].s)
        SigLphi.append(lphin[i].s)      
        LXN.append(Lxn[i].n)
        LYN.append(Lyn[i].n)
        LZN.append(Lzn[i].n)
        SigLXN.append(Lxn[i].s)
        SigLYN.append(Lyn[i].s)
        SigLZN.append(Lzn[i].s)
        B.append(b[i].n)
        SigB.append(b[i].s)
    return np.array(LXN), np.array(SigLXN), np.array(LYN), np.array(SigLYN), np.array(LZN), np.array(SigLZN), np.array(LTH), np.array(SigLth), np.array(LPH), np.array(SigLphi), np.array(B), np.array(SigB)


if __name__ == '__main__':
    t = np.ones((11))
    nom = ("Sagittarius", 'LMC','SMC','Draco','Ursa Minor','Sculptor','Sextans','Carina','Fornax','Leo II','Leo I')
    a = np.loadtxt('/home/jordi/satellite/MW_sat_pawlowski.txt').T
    x_MW_c, y_MW_c, z_MW_c, vx, sigvx, vy, sigvy, vz, sigvz, lpol, bpol, delpol, h, _ = a
    
    t = np.ones((11))

##############################################################################    
###########                 PLOTS POSITIONS            #####################    
##############################################################################       
#    pts.plot3d(x_MW_c, y_MW_c, z_MW_c, t, r'$x$[kpc]', r'$y$[kpc]', r'$z$[kpc]', 0, 45,
#           'Milky Way Classical Satellites',
#           '/home/jordi/satellite/mw_class_satellites_1',
#           galaxy= True,R=192)
#    pts.plotmultiple2(x_MW_c, y_MW_c, z_MW_c,'Milky Way Classical Satellites',
#                  '/home/jordi/satellite/mw_class_satellites_2', R=250, units =True)
#    
#    
#    r_MW_c = np.sqrt(x_MW_c**2 + y_MW_c**2 + z_MW_c**2)
#    th_MW = np.arccos(z_MW_c/r_MW_c)
#    phi_MW = np.arctan2(y_MW_c,x_MW_c)
#    pts.scater(th_MW,phi_MW,r'$\theta$',r'$\phi$','MW Class sat positions',ylim=(0,0),xlim=(0,0), xangular = True, 
#               z3D=False, z=[],angular =False, color = False, c=[], clab = '',
#               errorbar= False, yerr= [],
#               name = '', dpi = 250, R = 200, save = True, s = 1., extra_data=False,
#               x_extra = [], y_extra= [], extra_err=False, yerr_extra=None,
#               xerr_extra= None, extratext = False, texts = [])
#    pts.histo(th_MW, r'$\theta$', bins = 16, rang=(0,0),nom_archivo ='', fit = False,
#              dist = 'norm',normalized = False,title='MW Class sat',
#              logx = False, dens = False, xangular = True)
    
    
##############################################################################    
###########         PLOTS POSITIONS AITOFF GALACTOCENTRIC       #####################    
##############################################################################    
#    c_MW_c = SkyCoord(x_MW_c, y_MW_c, z_MW_c, unit='kpc', 
#                      frame= 'galactocentric',representation_type='cartesian')
#    c_MW_c.representation_type = 'spherical'
    
#    fig = plt.figure(figsize=(20,6))
#    ax = fig.add_subplot(1,2,1,projection='aitoff')
#    ax.set_title("Milky Way Classical Satellites")
#    for i in range(0,11):
#        ax.plot(c_MW_c[i].lon.wrap_at(180*u.deg).radian, c_MW_c[i].lat.radian, marker='*', markersize= 10)
#        ax.text(c_MW_c[i].lon.wrap_at(180*u.deg).radian, c_MW_c[i].lat.radian, nom[i], fontsize=8)
#    ax.grid()
#    #plt.legend(nom, loc= 'lower right', fontsize= 12)
#    ax.set_xlabel('lon $l$')
#    ax.set_ylabel('lat $b$')
#    plt.savefig('/home/jordi/satellite/mw_class_satellites_2', dpi=100, bbox_inches='tight')
#    plt.show()
    
##############################################################################    
###########         PLOTS POSITIONS AITOFF HELIOCENTRIC       #####################    
##############################################################################    
#    c_MW_HCRS = c_MW_c.transform_to(coord.HCRS)
#    c_MW_HCRS.representation_type = 'spherical'
#    
#    
#    fig = plt.figure(figsize=(20,6))
#    ax = fig.add_subplot(1,2,1,projection='aitoff')
#    ax.set_title("Milky Way Classical Satellites heliocentric coords")
#    cx = [];  cy = []; amplitudes=[]
#    for i in range(0,11):
#        cx.append(c_MW_HCRS[i].ra.wrap_at(180*u.deg).radian)
#        cy.append(c_MW_HCRS[i].dec.radian)
#        amplitudes.append(c_MW_HCRS[i].distance.kpc)
#        ax.text(c_MW_HCRS[i].ra.wrap_at(180*u.deg).radian,c_MW_HCRS[i].dec.radian, nom[i], fontsize=8)
#    sc = ax.scatter(cx,cy, c=amplitudes, s=20)
#    plt.colorbar(sc).ax.set_ylabel(r'$r$(kpc)')     
#    ax.grid()
#    #plt.legend(nom, loc= 'lower left')
#    ax.set_xlabel('lon $l$')
#    ax.set_ylabel('lat $b$')
#    plt.savefig('/home/jordi/satellite/mw_class_satellites_heliocentric', dpi=100, bbox_inches='tight')
#    plt.show()
    
##############################################################################    
###########        CALCULO ORBITAL POLES             #####################    
##############################################################################    
#    lx = y_MW_c*vz - z_MW_c*vy
#    ly = z_MW_c*vx - x_MW_c*vz
#    lz = x_MW_c*vy - y_MW_c*vx
#    lrho = np.sqrt(lx**2 + ly**2)
#    lr = np.sqrt(lrho**2 + lz**2)
#    siglx = np.sqrt(y_MW_c**2*sigvz**2 + z_MW_c**2*sigvy**2)
#    sigly = np.sqrt(z_MW_c**2*sigvx**2 + x_MW_c**2*sigvz**2)
#    siglz = np.sqrt(x_MW_c**2*sigvy**2 + y_MW_c**2*sigvx**2)
#    siglrho = np.sqrt(lx**2*siglx**2/lrho**2 + ly**2*sigly**2/lrho**2)
#    siglr = np.sqrt(lrho**2*siglrho**2/lr**2 + lz**2*siglz**2/lr**2)
#    lth = np.arccos(lz/lr)
#    lphi = np.arctan2(ly,lx)
#    siglth = np.sqrt(siglr**2*lz**2/(lr**4*(1 - (lz**2/lr**2))) + siglz**2/(lr**2*(1 - (lz**2/lr**2))))
#    siglphi = np.sqrt(sigly**2/(lx**2*(1+ (ly**2/lx**2))**2) + siglx**2*ly**2/(lx**4*(1+ (ly**2/lx**2))**2))
#    for i in range(0, np.shape(lx)[0]):
#        print('----- Satelite %s ------'%(nom[i]))
#        print('lth=', lth[i], '+-', siglth[i] )
#        print('lphi=',lphi[i], '+-', siglphi[i] )
#    plt.figure(figsize=(5,3))
#    pts.plotmultiple([], [], [], r"$\theta_l$", r"$\phi_l$", 'title', '', ylim=(-np.pi,np.pi),
#                     save = False, loc_leg='best',angular=True, xangular=True,data=True, xd=lth, yd=lphi,
#                    err=True, yerr=siglphi, errx = True,xerr = siglth)

##############################################################################    
###########            PLOTS ORBITAL POLES           #####################    
##############################################################################        
    lx, SigLXN, ly, SigLYN, lz, SigLZN, LTH, SigLth, LPH, SigLphi, b, sigb = MW_orbital_poles()     
        
    pts.plotmultiple([], [], [], r"$\theta_l$", r"$\phi_l$",
                     'Orbital poles', '', ylim=(-np.pi,np.pi),
                     save = False, loc_leg='best',angular=True, xangular=True,
                     data=True, xd=LTH, yd=LPH,
                    err=True, yerr=SigLphi, errx = True, xerr = SigLth)
     
    l_MW_c = SkyCoord(lx, ly, lz, frame= 'galactocentric',
                      representation_type='cartesian')
    l_MW_c.representation_type = 'spherical'
    
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(1,2,1,projection='aitoff')
    ax.set_title("Orbital Poles MW Classical Satellites")
    for i in range(0,11):
        ax.plot(l_MW_c[i].lon.wrap_at(180*u.deg).radian, l_MW_c[i].lat.radian, marker='*', markersize= 10)
        ax.text(l_MW_c[i].lon.wrap_at(180*u.deg).radian, l_MW_c[i].lat.radian, nom[i], fontsize=8)
    ax.grid()
    ax1 = fig.add_subplot(1, 2, 2,projection="lambert")
    for i in range(0,11):
        ax1.plot(l_MW_c[i].lon.wrap_at(180*u.deg).radian, l_MW_c[i].lat.radian, marker='*', markersize= 10)
    ax1.grid()
    plt.legend(nom, loc= 'best', fontsize= 12)
    plt.show()
##############################################################################    
    l_MW_c = SkyCoord(LPH, b, frame= 'galactocentric',
                      representation_type='spherical', unit = 'rad')
    sigl_MW = SkyCoord(SigLphi, sigb, frame= 'galactocentric',
                      representation_type='spherical', unit = 'rad')    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(1, 1, 1,projection="aitoff")
    ax1.set_title("Orbital Poles MW Classical Satellites")
    ax1.errorbar(l_MW_c.lon.wrap_at(180*u.deg).radian,
               l_MW_c.lat.radian,
               xerr = sigl_MW.lon.wrap_at(180*u.deg).radian,
               yerr = sigl_MW.lat.radian, 
                fmt='o',alpha = 0.95,c = 'r',
                 ms = 2.5, capsize = 2.5, elinewidth = 2.5)
    font = {'family': 'serif', 'color':  'red','weight': 'bold','size': 18}
    for i in range(0,11):
        ax1.text(l_MW_c[i].lon.wrap_at(180*u.deg).radian, 
                 l_MW_c[i].lat.radian, nom[i], fontdict=font)
    ax1.grid()
    plt.show()


#    l_MW_c.representation_type = 'cartesian'
#    LX = np.array(l_MW_c.x)
#    LY = np.array(l_MW_c.y)
#    LZ = np.array(l_MW_c.z)
#
#    L= np.array([LX,LY,LZ])
#    np.savetxt('L.txt', L)#, fmt='%.18e', delimiter=' ', newline='n', header='', footer='', comments='# ', encoding=None)[source]¶
    
