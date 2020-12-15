#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020
con hernquist y sin Rc
@author: jordi
"""
import pymc3 as pm3
import plots_jordi as pts
import numpy as np
from MW_M31_rotation_curve_padila import (v2_DM, M_Miyamoto,M_hernquist,
                                          dens_Miyamoto, M_Miyamoto2)
from MW_M31_rotation_curve import vHernquist, vnagai
from MW_M31_rotation_curve_pymc import todas_las_trazas
import matplotlib.pyplot as plt
import arviz as az
from rotation_curves_SPARC_padilla2 import f as ff
from rotation_curves_SPARC_padilla2 import g
from subprocess import call
from constants_grav import G, Gentc2, mu22
###############################################################################
np.random.seed(12345)
###############################################################################
galaxy = 'Milky Way'

if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/Gaussian/'%dirdata
    dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 
    rad, v, v_error = dataGrandMW.T
    rad, v, v_error =rad[:118], v[:118], v_error[:118]
elif galaxy == 'M31':
    dirdata = '/home/jordi/satellite/M31_rotation_curve_data'
    dirfitsG = '%s/Fits/Gaussian/'%dirdata
    dataM31 = np.loadtxt('%s/M31_rotation_curve_data.txt'%dirdata)
    _, rad, v, v_error = dataM31.T
   
def f(r, Mc):
    Rc = np.sqrt(2.*re**2*(1. + (re/rs))/(1. + 3.*(re/rs)))
    return Mc*(-2.*np.exp(-r**2/Rc**2)*r + np.sqrt(np.pi)*Rc*pm3.math.erf(r/Rc))/(Rc*np.sqrt(np.pi))  

def v2_DMatter(r, Mc, re, rs):## M es entre 10^10 M_sol
    Rc = re*np.sqrt(2.*(1. + (re/rs))/(1. + 3.*(re/rs)))
    rhos = Mc*re*np.exp(-re**2/Rc**2)*(1. + re/rs)**2/(rs*np.sqrt(np.pi)**3*Rc**3)
    Mh = pm3.math.switch(r>re, f(re, Mc) + 4.*np.pi*rs**3*rhos*(g(r, rs) - g(re, rs)),
                    f(r, Mc))
    ve2 = G*Mh/r
    return ve2
    
def M_CNFW(r, Mc, re, rs):
    Rc = np.sqrt(2.*re**2*(1. + (re/rs))/(1. + 3.*(re/rs)))
    rhos = Mc*re*np.exp(-re**2/Rc**2)*(1. + re/rs)**2/(rs*np.sqrt(np.pi)**3*Rc**3)
    Mh = ff(r, Mc, Rc)*np.heaviside(re - r, 0.5) + (ff(re, Mc, Rc)+ 4.*np.pi*rs**3*rhos*(g(r, rs) - g(re, rs)))*np.heaviside(r - re, 0.5)
    return Mh

def M_t(r, Mc, re, rs, Md, ad, bd, Mb, bb):
    Mhalo = M_CNFW(r, Mc, re, rs)
    Mdisk =  M_Miyamoto(r, Md, ad, bd)
    Mbulge = M_hernquist(r, Mb, bb)
    return Mhalo + Mdisk + Mbulge

def dens_halo(r, Mc, re, rs):
    Rc = np.sqrt(2.*re**2*(1. + (re/rs))/(1. + 3.*(re/rs)))
    rhos = Mc*re*np.exp(-re**2/Rc**2)*(1. + re/rs)**2/(rs*np.sqrt(np.pi)**3*Rc**3)  
    rhoG = Mc*np.exp(-r**2/Rc**2)/(np.sqrt(np.pi)**3*Rc**3)
    rhoNFW = rhos*rs**3/(r*(rs + r)**2)
    return np.heaviside(re - r, 0.5)*rhoG + np.heaviside(r - re, 0.5)*rhoNFW

############## 1
MO = {'Mc' : [1e-5, 1000.], 're' : [1e-3, 10.],'rs' : [1e-3, 500.], 
      'Md' : [1e-4, 100., 5.0], 'ad' : [10, 80., 20.],'bd' : [.8, 3., 1.15],
      'Mb' : [1e-4, 5., 0.3], 'bb' : [1e-3, 3., 1.6]}
nsamples = 1000
############# 2
#MO = {'Mc' : [1e-5, 1000.], 're' : [1e-3, 10.], 'rs' : [1, 100.], 
#      'Md' : [1e-4, 1000., 5.0], 'ad' : [10, 80., 20.],'bd' : [.8, 3., 1.15],
#      'Mb' : [1e-4, 5., 0.3], 'bb' : [1e-3, 3., 1.6]}
#nsamples = 1000
#
with pm3.Model() as model: 
    Mc = pm3.Uniform(r'$M_c$', MO['Mc'][0], MO['Mc'][1])
    re = pm3.Uniform(r'$r_e$', MO['re'][0], MO['re'][1])
    rs = pm3.Uniform(r'$r_s$', MO['rs'][0], MO['rs'][1])
    Md = pm3.Uniform(r'$M_d$', MO['Md'][0], MO['Md'][1])
    
    ad = pm3.Normal(r'$a_d$', mu=MO['ad'][2], sigma = 0.4)   
    bd = pm3.Normal(r'$b_d$', mu=MO['bd'][2], sigma = 0.4)
 
#    ad = pm3.Uniform(r'$a_d$', MO['ad'][0], MO['ad'][1]) 
#    bd = pm3.Uniform(r'$b_d$', MO['bd'][0], MO['bd'][1])

    Mb = pm3.Uniform(r'$M_b$', MO['Mb'][0], MO['Mb'][1])
    bb = pm3.Uniform(r'$b_b$', MO['bb'][0], MO['bb'][1])
    rot_vel = pm3.Deterministic('rot_vel', 
                                np.sqrt(v2_DMatter(rad, Mc, re, rs) + 
                                        vnagai(rad, G, Md, ad, bd)**2 +
                                        vHernquist(rad, G, Mb, bb)**2))
    y = pm3.Normal('y', mu = rot_vel, tau = 1.0/v_error**2, observed = v)
    start = pm3.find_MAP()
    step = pm3.NUTS()
    trace = pm3.sample(nsamples, start=start,
#                       cores = 4, tune = 2000, target_accept=.95
                       )
    
############         extract and plot results         ########################
pm3.summary(trace).to_csv('%s/summary_v3.csv' % dirfitsG)
#print(pm3.summary(trace))

y_min = np.percentile(trace.rot_vel,2.5,axis=0)
y_max = np.percentile(trace.rot_vel,97.5,axis=0)
y_fit = np.percentile(trace.rot_vel,50,axis=0)
#y_fit = trace.rot_vel[:]

name =   [r'$M_c$', r'$r_e$', r'$r_s$', r'$M_d$', r'$a_d$', r'$b_d$',
          r'$M_b$',r'$b_b$']
### summary keys ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'mcse_mean', 'mcse_sd', 'ess_mean', 'ess_sd', 'ess_bulk', 'ess_tail', 'r_hat']
popt = pm3.summary(trace, var_names=name)['mean'].values 
poptsd= pm3.summary(trace, var_names=name)['sd'].values 
popt97= pm3.summary(trace, var_names=name)['hdi_97%'].values 
np.savetxt('%s/params.txt'%(dirfitsG), np.array(popt))
Mc, re, rs, Md, ad, bd, Mb, bb = popt 
print(popt)
Rc = re*np.sqrt(2.*(1. + (re/rs))/(1. + 3.*(re/rs)))
print('Rc = ', Rc)



rmin, rmax = np.amin(rad), np.amax(rad)
r = np.linspace(rmin, rmax, 1000)
vdisk = vnagai(r, G=G, M = Md, a = ad, b = bd)
vbulge = vHernquist(r, G, Mb, bb)
pts.plotmultiple([rad,r, r, r, r], [y_fit, np.sqrt(vdisk**2 + vbulge**2),
                 vdisk, vbulge, np.sqrt(v2_DM(r, Rc, Mc, re, rs))],
                 [r'Disk+Bulge+SFDM', r'Disk+Bulge',
                  r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc, $b_d=%.2f$kpc'%(Md,ad,bd),
                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$kpc'%(Mb,bb), 
                  r'DM  $R_c = %.1f$ kpc, $M_c = %.1f \times 10^{10} M_\odot$, $r_e=%.2f$kpc, $r_s=%.1f$kpc'%(Rc, Mc, re, rs),
                  'Observed'],
                 r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_MCMC.png'%(dirfitsG), ylim = (10,1000),
                 xlim = (rmin,rmax), data = True, xd = rad, yd = v, err = True,
                 yerr = v_error, fill_between = True, logx = True,logy = True,
                 fbx = rad, fby1 = y_min, fby2 = y_max)


rmin, rmax = np.amin(rad), np.amax(rad)#20.
r = np.linspace(rmin, rmax, 1000)
vdisk = vnagai(r, G=G, M = Md, a = ad, b = bd)
vbulge = vHernquist(r, G, Mb, bb)
pts.plotmultiple([rad,r, r, r, r], [y_fit, np.sqrt(vdisk**2 + vbulge**2),
                 vdisk, vbulge, np.sqrt(v2_DM(r, Rc, Mc, re, rs))],
                 [r'Disk+Bulge+SFDM', r'Disk+Bulge',
                  r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc, $b_d=%.2f$kpc'%(Md,ad,bd),
                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$kpc'%(Mb,bb), 
                  r'DM  $R_c = %.1f$ kpc, $M_c = %.1f \times 10^{10} M_\odot$, $r_e=%.2f$kpc, $r_s=%.1f$kpc'%(Rc, Mc, re, rs),
                  'Observed'],
                 r'$r$(kpc)', r'$v$(km/s)', '%s'%galaxy,
                 '%sDM_fit_MCMC_v3.png'%(dirfitsG), ylim = (0,360),
                 xlim = (rmin,rmax), data = True, xd = rad, yd = v, err = True,
                 yerr = v_error, fill_between = True,
                 fbx = rad, fby1 = y_min, fby2 = y_max)

fil= open("%s/cantidades_pymc3_v3.txt"%dirfitsG,"w+")

print('M_tot(Rc)=',M_t(Rc, Mc, re, rs, Md, ad, bd, Mb, bb))
cons = M_t(Rc, *popt)**2*Mc*1.0e9/1.45e4
Rcsd = poptsd[1]*np.sqrt(2.*(1. + (poptsd[1]/poptsd[2]))/(1. + 3.*(poptsd[1]/poptsd[2])))
cons1sigma = M_t(Rcsd, *poptsd)**2*Mc*1.0e9/1.45e4
mu2 = 1./np.sqrt(np.sqrt(cons))
mu21sigma = 1./np.sqrt(np.sqrt(cons1sigma))
mu = 1./(Rc*Gentc2*M_t(Rc, Mc, re, rs, Md, ad, bd, Mb, bb))
fil.write('Nsamples = %d \r\n'%nsamples)
fil.write('Rc = %.2f kpc, \r\n'%Rc)
fil.write('M(Rc) = %f x10^{10} M_sun \r\n'%(M_t(Rc, *popt)))
fil.write('r = 2.00 kpc,  M(20kpc) = %f x10^{10} M_sun \r\n'%(M_t(20., *popt)))
fil.write('mu = %f x10^{-22} eV/c^2,  mu = %f  x10^{-22} eV/c^2 \r\n'%(mu2, mu/mu22))
fil.write('Mhalo(300kpc) = %f x10^{12} M_sun \r\n'%(M_CNFW(300., Mc, re, rs)*1e-2))
fil.close()

print('Rc =',Rc,'kpc', 'M(Rc) =',M_t(Rc, *popt), 'x10^{10} M_sun')
print('M(20kpc) =',M_t(20., *popt), 'x10^{10} M_sun')
print('mu1 =',mu/mu22, 'x10^{-22} eV/c^2')
print('mu2 =',mu2, 'x10^{-22} eV/c^2', '+-', mu21sigma,'x10^{-22} eV/c^2')
print('rho_halo(0) = ',Mc/(np.sqrt(np.pi)**3*Rc**3)*10., 'M_sun/pc^3')
print('rho_halo(Rc) = ', dens_halo(Rc, Mc, re, rs)*10., 'M_sun/pc^3')
print('rho_halo(re) = ', dens_halo(re, Mc, re, rs)*10., 'M_sun/pc^3')
print('rho_halo(20kpc) = ', dens_halo(20., Mc, re, rs)*10., 'M_sun/pc^3')
todas_las_trazas(trace,name, '%sDM_fit_MCMC_pars_v3.png'%(dirfitsG), 
                 point_estimate="mean")

#pm3.traceplot(trace, varnames=name)
#pm3.plot_posterior(trace, name[2],figsize=(5, 4))

az.plot_density(trace, var_names=name, shade=0.1, point_estimate='mean', 
                figsize = [3,5] )
plt.savefig('%sDM_fit_MCMC_post_v3.png'%(dirfitsG), bbox_inches='tight')
#az.plot_density(trace, var_names=name, shade=0.1, point_estimate='mean', 
#                group='prior', figsize = [3,5] )

#print('wolfram script....')
#call(["wolframscript", "-file", "Plot_galaxy_densities_mathematica.wls"])
print('Md=', Md)
print('Masa disco=', M_Miyamoto(rmax, Md, ad, bd))
print('Masa disco=', M_Miyamoto2(rmax, rmax, Md, ad, bd))