#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

CON BH y con datos de radios <1kpc

@author: jordi
"""
import pymc3 as pm3
import plots_jordi as pts
import numpy as np
from MW_M31_rotation_curve_padila import v2_DM, M_CNFW
from def_potenciales import (vHernquist, vnagai, g, vBH,
                             M_Miyamoto2,M_hernquist)
import matplotlib.pyplot as plt
import arviz as az
from constants_grav import G, Gentc2, mu22
from MW_M31_rotation_curve_pymc import todas_las_trazas
import galpy.potential as galpot
import theano.tensor as tt
###############################################################################
np.random.seed(12345)
galaxy = 'Milky Way'

if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/Gaussian/'%dirdata
    datalogRCMW= np.loadtxt("%s/LogRC_data.dat.txt"%dirdata) 
    rad,_, v, v_error = datalogRCMW.T
    
def M_t(r, Rc, Mc, re, rs, Md, ad, bd, Mb, bb, MBH):
    Mhalo = M_CNFW(r, Rc, Mc, re, rs)
    Mdisk =  M_Miyamoto2(r,r, Md, ad, bd)
    Mbulge = M_hernquist(r, Mb, bb)
    return Mhalo + Mdisk + Mbulge + MBH

def f(r, Mc, Rc):
    return Mc*(-2.*np.exp(-r**2/Rc**2)*r + np.sqrt(np.pi)*Rc*pm3.math.erf(r/Rc))/(Rc*np.sqrt(np.pi))  

def v2_DMatter(r, Rc, Mc, re, rs):## M es entre 10^10 M_sol
    rhos = Mc*re*np.exp(-re**2/Rc**2)*(1. + re/rs)**2/(rs*np.sqrt(np.pi)**3*Rc**3)
    Mh = pm3.math.switch(r>re, f(re, Mc, Rc) + 4.*np.pi*rs**3*rhos*(g(r, rs) - g(re, rs)),
                    f(r, Mc, Rc))
    ve2 = G*Mh/r
    return ve2

############## 1
MO = {'Rc' : [1e-3, 1.], 'Mc' : [1e-3, 10.], 
      're' : [1e-3, 15., 8.5],'rs' : [1., 50., 4.], 

      'Md' : [1e-2, 1e-1, 5.0], 'ad' : [0., 80., 20.],'bd' : [.8, 3., 1.15],

#      'Md' : [1e-1, 30., 5.0], 'ad' : [10, 80., 20.],'bd' : [.8, 3., 1.15],
      'Mb' : [1e-4, 5., 0.3], 'bb' : [1e-3, 5e-1, 1.6],
      'BH' : [1e-4, 1e-2]}
nsamples = 1000
#
with pm3.Model() as model: 
    Rc = pm3.Uniform(r'$R_c$', MO['Rc'][0], MO['Rc'][1])
    Mc = pm3.Uniform(r'$M_c$', MO['Mc'][0], MO['Mc'][1])
    re = pm3.Uniform(r'$r_e$', MO['re'][0], MO['re'][1])
    rs = pm3.Uniform(r'$r_s$', MO['rs'][0], MO['rs'][1])  
    Md = pm3.Uniform(r'$M_d$', MO['Md'][0], MO['Md'][1])
#    ad = pm3.Normal(r'$a_d$', mu=MO['ad'][2], sigma = 0.4)  
    ad = pm3.Uniform(r'$a_d$', MO['ad'][0], MO['ad'][1])  
    bd = pm3.Normal(r'$b_d$', mu=MO['bd'][2], sigma = 0.4)   
    Mb = pm3.Uniform(r'$M_b$', MO['Mb'][0], MO['Mb'][1])
    bb = pm3.Uniform(r'$b_b$', MO['bb'][0], MO['bb'][1])
    MBH = pm3.Uniform(r'$M_{BH}$', MO['BH'][0], MO['BH'][1])
    
    rot_vel = pm3.Deterministic('rot_vel', 
                                np.sqrt(v2_DMatter(rad,Rc, Mc, re, rs) +
                                        vnagai(rad, G, Md, ad, bd)**2 +
                                        vHernquist(rad, G, Mb, bb)**2 + 
                                        vBH(rad, G, MBH)**2))
    y = pm3.Normal('y', mu = rot_vel, tau = 1.0/v_error**2, observed = v)
    start = pm3.find_MAP()
    step = pm3.NUTS()
    trace = pm3.sample(nsamples, start=start,
#                       cores = 4, tune = 2000, target_accept=.90
                       )

#############         extract and plot results         ########################
pm3.summary(trace,round_to=10).to_csv('%s/summary_v5.csv' % dirfitsG)
y_min = np.percentile(trace.rot_vel,2.5,axis=0)
y_max = np.percentile(trace.rot_vel,97.5,axis=0)
y_fit = np.percentile(trace.rot_vel,50,axis=0)

name =   [r'$R_c$', r'$M_c$', r'$r_e$', r'$r_s$', r'$M_d$', r'$a_d$', r'$b_d$',
          r'$M_b$',r'$b_b$',r'$M_{BH}$']

popt = pm3.summary(trace, var_names=name,round_to=10)['mean'].values ### summary keys ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'mcse_mean', 'mcse_sd', 'ess_mean', 'ess_sd', 'ess_bulk', 'ess_tail', 'r_hat']
poptsd= pm3.summary(trace, var_names=name,round_to=10)['sd'].values 
Rc, Mc, re, rs, Md, ad, bd, Mb, bb, MBH = popt 

print(popt)
rmin, rmax = np.amin(rad), np.amax(rad)
r = np.linspace(rmin, rmax, 1000)
vdisk = vnagai(r, G=G, M = Md, a = ad, b = bd)

vbulge = vHernquist(r, G, Mb, bb)
vBH = vBH(r, G, MBH)
pts.plotmultiple([rad,r, r, r, r,r], [y_fit, np.sqrt(vdisk**2 + vbulge**2),
                 vdisk, vbulge, np.sqrt(v2_DM(r, Rc, Mc, re, rs)),
                 vBH],
                 [r'Disk+Bulge+SFDM', r'Disk+Bulge',
                  r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc, $b_d=%.2f$kpc'%(Md,ad,bd),
                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$kpc'%(Mb,bb), 
                  r'DM  $R_c = %.1f$ kpc, $M_c = %.1f \times 10^{10} M_\odot$, $r_e=%.2f$kpc, $r_s=%.1f$kpc'%(Rc, Mc, re, rs),
                  r'BH',
#                  r'BH, $M_{BH}=%.4f \times 10^{6} M_\odot$'%MBH*1.0e4,
                  'Observed'],
                 r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_MCMC_v5.png'%(dirfitsG),
                 xlim = (rmin,rmax), data = True, xd = rad, yd = v, err = True,
                 yerr = v_error, fill_between = True, logx = True,logy = True,
                 fbx = rad, fby1 = y_min, fby2 = y_max)

rmin, rmax = np.amin(rad), 20.
r = np.linspace(rmin, rmax, 1000)

vdisk = vnagai(r, G=G, M = Md, a = ad, b = bd)

vbulge = vHernquist(r, G, Mb, bb)
vBH = vBH(r, G, MBH)
pts.plotmultiple([rad,r, r, r, r, r], [y_fit, np.sqrt(vdisk**2 + vbulge**2),
                 vdisk, vbulge, np.sqrt(v2_DM(r, Rc, Mc, re, rs)),vBH],
                 [r'Disk+Bulge+SFDM', r'Disk+Bulge',
                  r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc, $b_d=%.2f$kpc'%(Md,ad,bd),
                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$kpc'%(Mb,bb), 
                  r'DM  $R_c = %.1f$ kpc, $M_c = %.1f \times 10^{10} M_\odot$, $r_e=%.2f$kpc, $r_s=%.1f$kpc'%(Rc, Mc, re, rs),
                  r'BH',
#                  r'BH, $M_{BH}=%.4f\times 10^{6} M_\odot$'%MBH*1e4,
                  'Observed'],
                 r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_MCMC_v5.png'%(dirfitsG), ylim = (0,360),
                 xlim = (rmin,rmax), data = True, xd = rad, yd = v, err = True,
                 yerr = v_error, fill_between = True,
                 fbx = rad, fby1 = y_min, fby2 = y_max)

pts.residual(v, y_fit, lowess=False)

fil= open("%s/cantidades_pymc3_v5.txt"%dirfitsG,"w+")
cons = M_t(Rc, *popt)**2*Mc*1.0e9/1.45e4
Rcsd = poptsd[0]
cons1sigma = M_t(Rcsd, *poptsd)**2*poptsd[1]*1.0e9/1.45e4
mu2 = 1./np.sqrt(np.sqrt(cons))
mu21sigma = 1./np.sqrt(np.sqrt(cons1sigma))
mu = 1./(Rc*Gentc2*M_t(Rc, *popt))
mu1sigma = 1./(Rcsd*Gentc2*M_t(Rcsd, *poptsd))

R99 = 2.38167*Rc
cons2 = M_t(R99, *popt)**2*Mc*1.0e9/2.536e5
mu3 = 1./np.sqrt(np.sqrt(cons2))
fil.write('Nsamples = %d \r\n'%nsamples)
fil.write('Rc = %.2f kpc, \r\n'%Rc)
fil.write('M(Rc) = %f x10^{10} M_sun \r\n'%(M_t(Rc, *popt)))
fil.write('r = 2.00 kpc,  M(20kpc) = %f x10^{10} M_sun \r\n'%(M_t(20., *popt)))
fil.write('mu = %f x10^{-22} eV/c^2,  mu = %f  x10^{-22} eV/c^2 \r\n'%(mu2, mu/mu22))
fil.close()

print('muDM =',Mc/(np.sqrt(np.pi)**3*Rc**2), 'x10^{10} M_sun/kpc^2')
print('Rc =',Rc,'kpc', 'M(Rc) =',M_t(Rc, *popt), 'x10^{10} M_sun')
print('M(20kpc) =',M_t(20., *popt), 'x10^{10} M_sun')
print('mu1 =',mu/mu22, 'x10^{-22} eV/c^2', '+-', mu1sigma/mu22,'x10^{-22} eV/c^2')
print('mu2 =',mu2, 'x10^{-22} eV/c^2', '+-', mu21sigma,'x10^{-22} eV/c^2')
print('rho_halo(0) = ',Mc/(np.sqrt(np.pi)**3*Rc**3)*10., 'M_sun/pc^3')
print('mu3 =',mu3, 'x10^{-22} eV/c^2')

todas_las_trazas(trace,name, '%sDM_fit_MCMC_pars_v5.png'%(dirfitsG), 
                 point_estimate="mode")

##pm3.traceplot(trace, varnames=name)
#pm3.plot_posterior(trace, name[1],figsize=(5, 4))

az.plot_density(trace, var_names=name, shade=0.1, point_estimate='mean', 
                figsize = [3,5] )