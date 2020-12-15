#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:23:03 2019

@author: jordi
"""
import numpy as np
import plots_jordi as pts
from constants_grav import c, Gentc2
rut = "/home/jordi/satellite/schrodinger_poisson/potpaco"
    
RF = {1:{'Rf': 30},
      2:{'Rf': 15},
      3:{'Rf': 7.5}}

aka= {1:{"rhos_ex" : 0.5, "nsol" : 1},
      2:{"rhos_ex" : 0.7, "nsol" : 2},
      3:{"rhos_ex" : 0.9, "nsol" : 3},
      4:{"rhos_ex" : 0.59, "nsol" : 4},
      5:{"rhos_ex" : 0.62, "nsol" : 5},
      6:{"rhos_ex" : 0.64, "nsol" : 6},
      7:{"rhos_ex" : 0.66, "nsol" : 7},
      8:{"rhos_ex" : 0.68, "nsol" : 8}}

cons = 1.65818e12
    
def cordenadas(VXZ):
    z = []
    for i in range(0,101):
        z.append(VXZ[i,1])
    
    rho = []
    i = 5050
    while i<10201:
        rho.append(VXZ[i,0])
        i += 101
      
    V = []
    i= 5050
    while i<10201:
        V0=[]
        for j in range(0,101):
            V0.append(VXZ[i+j,2])
        V.append(V0)
        i += 101  

    return np.array(rho), np.array(z), np.transpose(np.array(V))


def cordenadas_dens(phi):
    z = []
    for i in range(0,101):
        z.append(phi[i,1])
    
    x = []
    i = 0
    while i<10201:
        x.append(phi[i,0])
        i += 101
      
    phi_cuad = []
    i= 0
    while i<10201:
        V0=[]
        for j in range(0,101):
            V0.append(phi[i+j,2])
        phi_cuad.append(V0)
        i += 101  

    return np.array(x), np.array(z), np.transpose(np.array(phi_cuad))

def fuerza(VXZ):
    rho, z, V = cordenadas(VXZ)
    V2 = []
    for i in range(0,101):
        V2p = []
        for j in range(0,50):
            V2p.append(-(V[i,j + 1] - V[i,j])/(rho[j + 1] - rho[j]))
        V2.append(V2p)
    dVdrho = np.array(V2)
    
    V2 = []
    for i in range(0,100):
        V2p = []
        for j in range(0,51):
            V2p.append(-(V[i + 1, j] - V[i,j])/(z[i + 1] - z[i]))
        V2.append(V2p)
    dVdz = np.array(V2)
    return dVdrho, dVdz  
  
def density_interpolada(dens):
    from scipy.interpolate import interp2d
    x, z, phic = cordenadas_dens(dens)
    phicu = interp2d(x, z, phic, kind='linear', copy=True, bounds_error=False)
    return phicu

def potential_interpolado(VXZ):
    from scipy.interpolate import interp2d
    x, z, potxz = cordenadas(VXZ)
    Vu = interp2d(x, z, potxz, kind='linear', copy=True, bounds_error=False)
    return Vu

def fuerza_interpolada(VXZ):
    from scipy.interpolate import interp2d
    dVdrho, dVdz = fuerza(VXZ)
    rho,z,_ = cordenadas(VXZ)
    rhor = rho[:50]
    derrho = interp2d(rhor, z, dVdrho, kind='linear', copy=True, bounds_error=False)
    zr = z[:100]
    derz = interp2d(rho, zr, dVdz, kind='linear', copy=True, bounds_error=False)    
    return derrho, derz

def plots_fuerza(VXZ, nsol, refi):
    rho, z, V = cordenadas(VXZ)
    dVdrho, dVdz = fuerza(VXZ)
    rut = "/home/jordi/satellite/schrodinger_poisson/potpaco"
    rhonew = np.linspace(0,rho[49], 1000)
    znew = np.linspace(z[0],z[99], 1000)
    derrho, derz = fuerza_interpolada(VXZ)
    Rho, Z = np.meshgrid(rhonew,znew)
    pts.densityplot(Rho, Z, derrho(rhonew, znew), r"$\rho$", r"$z$",
                    r"$-\frac{\partial V}{\partial \rho}(\rho,z)$", "", 
                    name="%s/pot_%d/dVdrho_%d.png"%(rut,nsol,refi))
    pts.densityplot(Rho, Z, derz(rhonew, znew), r"$\rho$", r"$z$",
                    r"$-\frac{\partial V}{\partial z}(\rho,z)$", "", 
                    name="%s/pot_%d/dVdz_%d.png"%(rut,nsol,refi))
   
def plots_SFDM_density(nsol, di = '', refina = 3):
    densb = np.loadtxt("%srho_b_%d_%d.txt"%(di,nsol,refina))
    densex = np.loadtxt("%srho_ex_%d_%d.txt"%(di,nsol,refina))
    rho, z, phib = cordenadas(densb)
    _, _, phiex = cordenadas(densex)
#    Rho, Z = np.meshgrid(rho,z)
#    pts.densityplot(Rho,Z,phib,r"$\rho$",r"$z$",r"$\Phi^2_0(x,0,z)$","DM",
#                    name="%s/%spot_%d/dens_b_%d.png"%(rut,di,nsol,refina))
#    pts.densityplot(Rho,Z,phiex,r"$\rho$",r"$z$",r"$\Phi^2_1(x,0,z)$","DM",
#                    name="%s/%spot_%d/dens_ex_%d.png"%(rut,di,nsol,refina))
#    pts.densityplot(Rho,Z,phiex + phib,r"$\rho$",r"$z$",r"$\Phi^2_0 + \Phi^2_1$",
#                    "DM $xz$-plane projection",
#                    name="%s/%spot_%d/dens_mix_%d.png"%(rut,di,nsol,refina))
    pts.plotmultiple([z, z], 
                     [phib[:,0],phiex[:,0]],
                     [r'$\Psi_{100}$',r'$\Psi_{210}$'],
                     r"$z$", "", "", "",save = False)
    print('corregir caso l= 1, m=1')
    np.save("%srho(r)_b_%d_%d.npy"%(di,nsol,refina), np.array([z[50:], phib[50:,0]]))
    np.save("%srho(r)_ex_%d_%d.npy"%(di,nsol,refina), np.array([z[50:], phiex[50:,0]/3.]))
    r, psib = np.load("%srho(r)_b_%d_%d.npy"%(di,nsol,refina))
    r, psiex = np.load("%srho(r)_ex_%d_%d.npy"%(di,nsol,refina))
    pts.plotmultiple([r,r], 
                     [psib,psiex],
                     [r'$\psi_{100}^2$',r'$r^2\psi_{210}^2$'],
                     r"$z$", "", "", "",save = False)

def plots_SFDM_density_interpolado(nsol, di = '', ref = 3, m=0,
                                   lam = 1., mu = 1):
    rf = RF[ref]['Rf']  
    densb = np.loadtxt("%srho_b_%d_%d.txt"%(di,nsol,ref))
    densex = np.loadtxt("%srho_ex_%d_%d.txt"%(di,nsol,ref))
    phib = density_interpolada(densb)
    phiex = density_interpolada(densex)
    x = np.linspace(-rf,rf,500)
    z = np.linspace(-rf,rf,500)
    X, Z = np.meshgrid(x,z)

    if mu!= 1:
        cons = mu**2/(4*np.pi*Gentc2)
        xlab = r'$x$(kpc)'
        zlab = r'$z$(kpc)'
        denslab = r'$|\Psi_{100}|^2 + |\Psi_{21%d}|^2 (10^{1}M_\odot/pc^3)$'%m
        uni = 'units'
        lab1d = '$(10^{1}M_\odot/pc^3)$'
    else :
        cons = 1.
        xlab = r'$\sqrt{\lambda} \hat\mu x$'
        zlab = r"$\sqrt{\lambda} \hat{\mu} z$"
        denslab = r'$|\Psi_{100}|^2 + |\Psi_{21%d}|^2$'%m
        uni = 'adim'
        lab1d = 'au'
#    pts.densityplot(X/(mu*np.sqrt(lam)),Z/(mu*np.sqrt(lam)),
#                    cons*lam**2*phib(x,z),r"$\hat{\mu} x$",
#                    r"$\hat{\mu} z$",r"$|\Psi_{100}|^2$","",
#                    name="%s/%spot_%d/dens_b_ref%d.png"%(rut,di,nsol,ref),
#                    aspect='1/1')
#    pts.densityplot(X/(mu*np.sqrt(lam)),Z/(mu*np.sqrt(lam)),
#                    cons*lam**2*phiex(x,z),r"$\hat{\mu} x$",
#                    r"$\hat{\mu} z$",r"$|\Psi_{21%d}|^2$"%m,"",
#                    name="%s/%spot_%d/dens_ex_ref%d.png"%(rut,di,nsol,ref), 
#                    aspect='1/1')
    if m==0:
        extraylab = zlab
        extray1 = cons*lam**2*phib(x,z)[:,250]
        extray2 = cons*lam**2*phiex(x,z)[:,250]
    elif m==1:
        extraylab = xlab
        extray1 = cons*lam**2**phib(x,z)[250,:]
        extray2 = cons*lam**2*phiex(x,z)[250,:]
        
    pts.densityplot2(X/(mu*np.sqrt(lam)),Z/(mu*np.sqrt(lam)),
                     cons*lam**2*phiex(x,z) + cons*lam**2*phib(x,z),
                     z/(mu*np.sqrt(lam)), extray1, extray2, xlab, zlab, 
                     denslab, extraylab, 
                     [r'$|\Psi_{100}|^2 $', r'$|\Psi_{21%d}|^2$'%m], "",
                    name="%s/%spot_%d/dens_mix_ref%d_%s.png"%(rut,di,nsol,ref,uni), 
                    aspect='1/1')
    pts.plotmultiple([z/(mu*np.sqrt(lam)), z/(mu*np.sqrt(lam)), z/(mu*np.sqrt(lam))], 
                      [extray1, extray2, extray1 + extray2], 
                      [r'$|\Psi_{100}|^2 $', r'$|\Psi_{21%d}|^2$'%m],
                      zlab, lab1d, 'density', 
                      "%s/%spot_%d/dens_mix_z_ref%d_%s.png"%(rut,di,nsol,ref,uni))

def plots_potential_interpolado(nsol, di = '', ref = 1):
    rf = RF[ref]['Rf']
    VXZ = np.loadtxt("%sVxz_%d_%d.txt"%(di,nsol,ref))
    Vi = potential_interpolado(VXZ)
    rho = np.linspace(0.,rf,1001)
    z = np.linspace(-rf,rf,2001)
    Rho, Z = np.meshgrid(rho,z)
#    pts.densityplot2(Rho, Z, Vi(rho,z),
#                     Rho[100,:], Vi(rho,z)[100,:], Vi(rho,z)[166,:],
#                     r"$\hat \mu \rho$", r"$\hat \mu z$", 
#                     r"$\hat V(\hat\rho,\hat z)$", 
#                     r"$\hat \mu \rho$",
#                     [r"$\hat V (\hat\rho, 0)$", r"$\hat V (\hat\rho, 2.5)$"],
#                     '', name="%s/%spot_%d/V_int.png"%(rut,di,nsol))
#    pts.plotfunc3d(Rho,Z,Vi(rho,z),r"$\rho$",r"$z$",r"$\psi(\rho,z)$",
#                   "Potential %.2f"%aka[nsol]["rhos_ex"],
#                    name="%s/%spot_%d/potencial3D.png"%(rut,di,nsol))
    V00 = (Vi(rho,z)[1000:, 500] + 2.*Vi(rho,z)[1000, :])/3.
    r2V20 = -(Vi(rho,z)[1000:, 500] - Vi(rho,z)[1000, :])*2./(3.*np.sqrt(5))
    r = rho
    np.save("%s/%spot_%d/V00_%d.npy"%(rut,di,nsol,nsol),np.array([r, V00]))
    np.save("%s/%spot_%d/r2V20_%d.npy"%(rut,di,nsol,nsol),np.array([r, r2V20]))
    r, V00 = np.load("%s/%spot_%d/V00_%d.npy"%(rut,di,nsol,nsol))
    _, r2V20 = np.load("%s/%spot_%d/r2V20_%d.npy"%(rut,di,nsol,nsol))
    from def_potenciales import POT_multi_SFDM, POT_multi_SFDM2  
    r3 = np.linspace(0, 100, 2000)
    v00, r2v20 = POT_multi_SFDM(r3, nsol, di = di, rut =rut)
    np.save("%s/%spot_%d/V00_ext_%d.npy"%(rut,di,nsol,nsol),np.array([r3, v00]))
    np.save("%s/%spot_%d/r2V20_ext_%d.npy"%(rut,di,nsol,nsol),np.array([r3, r2v20]))
    
    pts.plotmultiple([r, r, r3, r3], [V00, r2V20, v00, r2v20],
                     [r'$V_{00}(r)$', r'$r^2 V_{20}(r)$',
                      r'$V_{00}(r)$', r'$r^2 V_{20}(r)$'],
                     r"$r$", "", "", "", xlim=(0,0),save = False)
    
    from scipy.misc import derivative
    r3 = np.linspace(0, 100, 2000)    
    v000, r2v200 = POT_multi_SFDM2(r3, nsol, di = di, rut =rut)
    dV0dr = derivative(v000, r3)
    dV2dr = derivative(r2v200, r3)
    pts.plotmultiple([r3, r3], [dV0dr, dV2dr],
                     [r'$dV_{00}(r)$', r'$d(r^2 V_{20}(r))$'],
                     r"$r$", "", "", "", xlim=(0,0),save = False)
#    th = np.linspace(0, np.pi, 101)
#    R, TH = np.meshgrid(r, th)
#    def potencial_sph_coords(r, th, V00, r2V20):
#        return V00 + r2V20*np.sqrt(5)*(-1. + 3.*np.cos(th)**2)/2.
#    f = potencial_sph_coords(R, TH, V00, r2V20)
#    print(np.shape(R), np.shape(TH), np.shape(f))
#    pts.plotfunc3d(R,TH,f,r"$r$",r"$\theta$",r"$\psi(r,\theta)$",
#                   "Potential %.2f"%aka[nsol]["rhos_ex"],
#                    name="%s/%spot_%d/potencial3D_sph.png"%(rut,di,nsol))
#    pts.densityplot(R, TH, f, r"$r$",r"$\theta$",r"$\psi(r,\theta)$",
#                    "Potential %.2f"%aka[nsol]["rhos_ex"],
#                    name="%s/%spot_%d/potencialdens_sph.png"%(rut,di,nsol),
#                    aspect='1/1')    

def plot_RC_adim(nsol,ref=1, di = ''):
    rf = RF[ref]['Rf']
    VXZ = np.loadtxt("%sVxz_%d_%d.txt"%(di,nsol,ref))
    rho = np.linspace(0.,rf,1001)
    z = np.linspace(-rf,rf,2001)
    lim = 1000
    dV_dR, _ = fuerza_interpolada(VXZ)
    v_cuad = -rho*dV_dR(rho,z)[lim, :]
    pts.plotmultiple([rho[:-100]], [np.sqrt(v_cuad)[:-100]], [],
                     r"$\hat{\mu}\rho$",r'$v/c$', 'multiSFDM RC pot%d'%nsol,
                     "%s/%spot_%d/rot_curve_%d.png"%(rut,di,nsol,nsol))
    np.save("%s/%spot_%d/vdm_%d.npy"%(rut,di,nsol,nsol),np.array([rho[:-100], np.sqrt(v_cuad)[:-100]]))
    r3 = np.linspace(0, 90, 200)
    from def_potenciales import v_multi_SFDM
    pts.plotmultiple([rho[:-100], r3],
                     [np.sqrt(v_cuad)[:-100],
                      v_multi_SFDM(r3, 1, 1, nsol)], [],
                     r"$\hat{\mu}\rho$",r'$v/c$', 'multiSFDM RC pot%d'%nsol,
                     "%s/%spot_%d/rot_curve_%d.png"%(rut,di,nsol,nsol))
    
def plot_N(nsol,ref=1, di = ''):
    r2 = np.linspace(1, 30, 100)
    NN2 = []
    for i in range(0, 100):
        Nb, Ne = num_partic_rf(nsol, ref= ref, rf = r2[i], di= di)
        NN2.append(Nb + Ne)  
    pts.plotmultiple([r2], [NN2], [],
                     r"$\hat{\mu}\rho$",r'$N$', 'multiSFDM RC pot%d'%nsol,
                     "%s/%spot_%d/Num_part_%d.png"%(rut,di,nsol,nsol))
    
DE = {1:{23:{'mu': 1565.5, 'rlam': 4.0e-3, 'limb' : 1001, 'ref': 1},
        24:{'mu' : 156.55, 'rlam' : 1.5e-2, 'limb' : 400}, ####### para di
         25:{'mu' : 15.655, 'rlam' : 2.9e-2, 'limb' : 100}},
    2: {23:{'mu': 1565.5, 'rlam': 4.0e-3, 'limb' : 1001, 'ref': 1},
        24:{'mu' : 156.55, 'rlam' : 3.8e-3, 'limb' : 90},
#       25:{'mu' : 15.655, 'rlam' : 2.9e-2, 'limb' : 501, 'ref': 3}},
#        25:{'mu' : 15.655, 'rlam' : 9.5e-3, 'limb' : 501, 'ref': 3}}, ###para pot_paco_orbitas_new       
        25:{'mu' : 26.592, 'rlam' : 8.6e-3, 'limb' : 501, 'ref': 3}}, ###para pot_paco_orbitas_new y curvas de rotacion a la vez
    3: {23:{'mu': 1565.5, 'rlam': 4.0e-3, 'limb' : 1001, 'ref': 1},
        24:{'mu' : 156.55, 'rlam' : 3.8e-3, 'limb' : 90},
        25:{'mu' : 15.655, 'rlam' : 1.5e-2, 'limb' : 50}},
    4: {21:{'mu': 156550.,'rlam': 1.0e-3, 'limb' : 4001, 'ref': 1},
          22:{'mu': 15655.0,'rlam': 1.0e-2, 'limb' : 4001, 'ref': 1},
          23:{'mu': 1565.5, 'rlam': 4.0e-3, 'limb' : 1001, 'ref': 1},
          24:{'mu': 156.55, 'rlam': 1.5e-2, 'limb' : 1001, 'ref': 2},
          25:{'mu': 15.655, 'rlam': 5.5e-2, 'limb' : 1001, 'ref': 3}},
    5:{21:{'mu': 156550.,'rlam': 1.0e-3, 'limb' : 4001, 'ref': 1},
          22:{'mu': 15655.0,'rlam': 1.0e-2, 'limb' : 4001, 'ref': 1},
          23:{'mu': 1565.5, 'rlam': 4.0e-3, 'limb' : 1001, 'ref': 1},
          24:{'mu': 156.55, 'rlam': 1.5e-2, 'limb' : 1001, 'ref': 2},
          25:{'mu': 15.655, 'rlam': 5.5e-2, 'limb' : 1001, 'ref': 3}},
   6:{22:{'mu': 15655.0,'rlam': 1.0e-3, 'limb' : 4001, 'ref': 1},
      23:{'mu': 1565.5, 'rlam': 4.0e-3, 'limb' : 1001, 'ref': 1},
      24:{'mu' : 156.55, 'rlam' : 3.2e-3, 'limb' : 300},
#      25:{'mu' : 15.655, 'rlam' : 1.9e-2, 'limb' : 300, 'ref': 3}}}###para pot_paco_orbitas
#     25:{'mu' : 15.655, 'rlam' : 5.9e-1, 'limb' : 300, 'ref': 3}}}###para pot_paco_orbitas_new
     25:{'mu' : 99.34, 'rlam' : 1.0e-1, 'limb' : 300, 'ref': 3}}}###para pot_paco_orbitas_new y curvas de rotacion a la vez
  
def plot_RC_unidades(nsol,mue= 21, ref=1, di = ''):
    rf = RF[ref]['Rf']
    VXZ = np.loadtxt("%sVxz_%d_%d.txt"%(di,nsol,ref))   
    rho = np.linspace(0.,rf,4001)
    z = np.linspace(-rf,rf,2001)   
    dV_dR, _ = fuerza_interpolada(VXZ)
    v_cuad = -rho*dV_dR(rho,z)
    lim = 1000
    rlam = DE[nsol][mue]['rlam']
    mu = DE[nsol][mue]['mu']
    pts.plotmultiple([rho/(rlam*mu)],
                      [np.sqrt(rlam**2*v_cuad[lim, :]*c**2)],
                      [],r"$\rho$ (kpc)",r'$v$(km/s)',
                       r'$\mu = 10^{-%d}$ eV/$c^2$ multistate DM Rotation curve'%mue,
                       "%s/%spot_%d/rot_curve_nsol%d_m%d.png"%(rut,di,nsol,nsol,mue))
    return np.array(rho/(rlam*mu)), np.array(np.sqrt(rlam**2*v_cuad[lim, :]*c**2))
###############################################################################
###### Funcion para crear archivos density 3d para graficar en mathematica ####################
###############################################################################    
def density3d_files(nsol, ref = 3, de = ''):    
    def ficuad3D(phib,phiex, x, y, z, unedo=False):
        Phi = []
        for i in range(0,np.shape(x)[0]):
            Phii=[]
            for j in range(0,np.shape(y)[0]):
                Phiii=[]
                for k in range(0,np.shape(z)[0]):
                    R,Z = np.sqrt(x[i]**2 + y[j]**2), z[k]
                    if unedo==False:
                        Phiii.append(phiex(R,Z)+phib(R,Z))
                    else:
                        Phiii.append(phib(R,Z))
                Phii.append(Phiii)
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi)    
    rf = RF[ref]['Rf']
    densb = np.loadtxt("%srho_b_%d_%d.txt"%(de,nsol,ref))
    densex = np.loadtxt("%srho_ex_%d_%d.txt"%(de,nsol,ref))
    phib = density_interpolada(densb)
    phiex = density_interpolada(densex) 
    Rf = rf #- 4
    p = 80   
    pp= p**3
    x = np.linspace(-Rf , Rf , p)
    y = np.linspace(-Rf , Rf , p)
    z = np.linspace(-Rf , Rf , p)    
    a= ficuad3D(phib, phiex, x, y, z)
    b = a.reshape(1,pp)
    np.savetxt("%s/%smix_%d.CSV"%(rut,de,nsol),b,delimiter=',')    
    a= ficuad3D(phib, 0, x, y, z, unedo=True)
    b = a.reshape(1,pp)
    np.savetxt("%s/%sbase_%d.CSV"%(rut,de,nsol),b,delimiter=',')    
    a= ficuad3D(phiex,0, x, y, z, unedo=True)
    b = a.reshape(1,pp)
    np.savetxt("%s/%sexx_%d.CSV"%(rut,de,nsol),b,delimiter=',')
    
###############################################################################
###################           Numero de particulas        ########################
###############################################################################      
def num_partic(nsol, ref = 1, di= ''):
    rf = RF[ref]['Rf']
    Nb, Ne = num_partic_rf(nsol, ref= ref, rf = rf, di= di)
    print('Nb/Ne=', Nb/Ne)

def num_partic_rf(nsol, ref= 1, rf = 1., di= ''):
    r, psib = np.load("%srho(r)_b_%d_%d.npy"%(di,nsol,ref))## estos se generan con 
    r, psiex = np.load("%srho(r)_ex_%d_%d.npy"%(di,nsol,ref))## plots_SFDM_density()
    from scipy.interpolate import interp1d
    phib = interp1d(r, psib, kind='linear', copy=True, bounds_error=False)
    phiex = interp1d(r, psiex, kind='linear', copy=True, bounds_error=False)    
    if rf>r[-5]:
        r = np.linspace(0, r[-5], 201)
        Nb = integralesferica(phib(r), r, r[-5])
        Ne = integralesferica(phiex(r), r, r[-5])
        return Nb, Ne
    else:
        r = np.linspace(0, rf, 201)
        Nb = integralesferica(phib(r), r, rf)
        Ne = integralesferica(phiex(r), r, rf)
        return Nb, Ne

def integralesferica(f,r,R):
    "integral definida de r**2*f(r) dr r de 0 a R "
    A=0.
    dr = r[1]-r[0]
    for i in range(0,np.shape(r)[0],1):
        A+= dr*f[i]*r[i]**2
    return A
###############################################################################
###############################################################################
###############################################################################
def numero_particulas2(f,r,R,Z,dr,dz):
    "integral definida de r*f(r,z)**2 drdz  r de 0 a R y z de -R a R"
    A=0.
    elem = int(np.rint(R/dr))
    elemz = int(np.rint(Z/dz))
    for i in range(0,elemz,1):
        for j in range(0,elem,1):
            A+= dr*f[i,j]*dz*r[j]
    return A

def plot_Nrhoz(nsol,ref = 3, di = ''):
    rf = RF[ref]['Rf'] 
    densb = np.loadtxt("%srho_b_%d_%d.txt"%(di,nsol,ref))
    densex = np.loadtxt("%srho_ex_%d_%d.txt"%(di,nsol,ref))
    phib = density_interpolada(densb)
    phiex = density_interpolada(densex)
    tol = 0.
    rho = np.linspace(tol, rf - tol, 101)
    z = np.linspace(tol, rf - tol, 201)
    Nxz=[]        
    for i in range(1,100):
        aux = []
        for j in range(1,200):
            if rho[i]**2 + z[j]**2 > rf**2:
                aux.append(np.nan)
            else:
                z2 = np.linspace(tol, z[j] - tol, j+1)
                rho2 = np.linspace(tol, rho[i] - tol, i+1)        
                dr = rho2[1]-rho2[0]
                dz = z2[1]-z2[0]
                Nb = numero_particulas2(phib(rho2,z2), rho2, rho[i], z[j], dr,dz)
                Ne = numero_particulas2(phiex(rho2,z2), rho2, rho[i], z[j], dr,dz)  
                aux.append(Nb+Ne)
    #            aux.append(Nb)
        Nxz.append(aux)    
    Nxz = np.array(Nxz)
    Nxz = np.transpose(Nxz)    
    X, Z = np.meshgrid(rho[1:-1],z[1:-1])  
    pts.densityplot(X,Z,Nxz,r"$\rho$ (kpc)", r"$z$ (kpc)",r'$N(\rho,z)$',
                    "", aspect = '1/1', name="")
##############################################################################
###############################################################################
###############################################################################
if __name__ == '__main__':
    di = 'baja_dens/'
#    di = ''
    nsol = 2
    refina = 1
    VXZ = np.loadtxt("%sVxz_%d_%d.txt"%(di,nsol,refina))
    dVdrho, dVdz = fuerza(VXZ)
    rho,z, potxz = cordenadas(VXZ)
    np.save("%sdVdrho_%d_%d.npy"%(di,nsol,refina), dVdrho)
    np.save("%sdVdz_%d_%d.npy"%(di,nsol,refina), dVdz)
    np.save("%scoordrho_%d_%d.npy"%(di,nsol,refina), rho)
    np.save("%scoordz_%d_%d.npy"%(di,nsol,refina), z)    
    np.save("%spotxz_%d_%d.npy"%(di,nsol,refina), potxz) 
    
#    for mm in range(25,26,1):
#        nsol = 6 
#        ref = DE[nsol][mm]['ref']
#        ref = 1
#        plotdensitySFDM_units(nsol,mue=mm,ref = ref, de = di)
#        _, _ = plot_RC_unidades(nsol,mue= mm, ref=ref, di = di)    

#    plot_Nrhoz(nsol,ref = 3, di = di)

    for nsol in range(6,7):
#        print(nsol, "baja densidad")
#        num_partic(nsol, di = di)
#        plot_RC_adim(nsol,ref=1, di = di)
#        _, _ = plot_RC_unidades(nsol,mue= 23, ref=1, di = di)
#        plots_SFDM_density_interpolado(nsol, di = di, ref = 3, m=0)
#        plots_SFDM_density(nsol, di = di, refina = 1)
#        plot_N(nsol,ref=1, di = di)
        plots_potential_interpolado(nsol,ref=1, di = di)
#    x = []
#    y = []
#    legends = []
#    for nsol in range(1,7):
#        r, v = np.load('%s/%spot_%d/vdm_%d.npy'%(rut,di,nsol,nsol))
#        x.append(r)
#        y.append(v)
#        legends.append('nsol %d'%nsol)
#    pts.plotmultiple(x, y, legends, r'$\hat\mu r$', r'$v/c$', 'RCs', '%s/%sRCs.png'%(rut,di),
#                     text= '', xv=[])

    
    
#    density3d_files(nsol, ref = 3, de = di)