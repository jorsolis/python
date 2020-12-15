#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

con miyamoto y sin Rc

@author: jordi
"""
import pymc3 as pm3
import plots_jordi as pts
import numpy as np
from MW_M31_rotation_curve_padila import v2_DM, M_Miyamoto
from MW_M31_rotation_curve import RC_miyamoto
import matplotlib.pyplot as plt
import arviz as az
from rotation_curves_SPARC_padilla2 import f as ff
from rotation_curves_SPARC_padilla2 import g
from subprocess import call
from constants_grav import c, G, Gentc2
#plt.style.use('seaborn')
###############################################################################
np.random.seed(12345)
sat = '/home/jordi/satellite'
dirfitsG = '/home/jordi/satellite/MW_rotation_curve_data/Fits/Gaussian/'
dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
dataM31 = np.loadtxt('%s/M31_rotation_curve_data.txt'%sat)
dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 
###############################################################################
rad, v, v_error = dataGrandMW.T
rad, v, v_error =rad[:118], v[:118], v_error[:118]

#def g(r, rs):
#    return  1./(1. + r/rs) + np.log(1. + r/rs)

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
    Mbulge = M_Miyamoto(r, Mb, 0., bb)
    return Mhalo + Mdisk + Mbulge
############### 1
#MO = {'Mc' : [0.50, 24., 14.], 're' : [0.10, 40., 22.], 'rs' : [0.0, 230., 180.], 
#      'Md' : [0.00, 60., 10.0], 'ad' : [0.00, 10., 1.],'bd' : [0.01, 10.90, 0.15],
#      'Mb' : [0.00, 3.5, 1.3], 'bb' : [0.00, 1., 0.35]}
############# 2
#MO = {'Mc' : [0.0001, 1000., 8.], 're' : [0.0001, 1., 0.001], 'rs' : [0.0001, 600., 2.], 
#      'Md' : [0.0001, 20., 10.0], 'ad' : [0.0001, 10., 1.],'bd' : [0.0001,5.0, 0.15],
#      'Mb' : [0.0001, 3.5, 1.3], 'bb' : [0.0001, 1., 0.35]}
#nsamples = 1000
############# 3
#MO = {'Mc' : [1e-4, 1e-3, 5.0e-4], 're' : [1e-4, 1., 1e-3],
#      'rs' : [0.0001, 1000., 2.], 
#      'Md' : [0.0001, 20., 10.0], 'ad' : [0.0001, 10., 1.],'bd' : [0.0001,5.0, 0.15],
#      'Mb' : [0.0001, 3.5, 1.3], 'bb' : [0.0001, 1., 0.35]}
############# 4
MO = {'Mc' : [1e-5, 1000., 5.0e-3], 're' : [1e-3, 10., 1e-2],
      'rs' : [1e-3, 500., 1.0e-2], 
      'Md' : [1e-4, 10., 5.0], 'ad' : [10., 50., 20.],'bd' : [1., 5.0, 1.15],
      'Mb' : [1e-4, 1., 0.3], 'bb' : [1e-1, 2, 0.8]}
nsamples = 1000
#
#with pm3.Model() as model: 
#    Mc = pm3.Uniform(r'$M_c$', MO['Mc'][0], MO['Mc'][1], testval = MO['Mc'][2])
#    re = pm3.Uniform(r'$r_e$', MO['re'][0], MO['re'][1], testval = MO['re'][2])
#    rs = pm3.Uniform(r'$r_s$', MO['rs'][0], MO['rs'][1], testval = MO['rs'][2])  
#    Md = pm3.Uniform(r'$M_d$', MO['Md'][0], MO['Md'][1], testval = MO['Md'][2])
#    ad = pm3.Uniform(r'$a_d$', MO['ad'][0], MO['ad'][1], testval = MO['ad'][2])
#    bd = pm3.Uniform(r'$b_d$', MO['bd'][0], MO['bd'][1], testval = MO['bd'][2])
#    Mb = pm3.Uniform(r'$M_b$', MO['Mb'][0], MO['Mb'][1], testval = MO['Mb'][2])
#    bb = pm3.Uniform(r'$b_b$', MO['bb'][0], MO['bb'][1], testval = MO['bb'][2])
#    rot_vel = pm3.Deterministic('rot_vel', np.sqrt(v2_DMatter(rad, Mc, re, rs) + 
#                                        RC_miyamoto(rad, G, Md, ad, bd)**2 +
#                                        RC_miyamoto(rad, G, Mb, 0., bb)**2))
#    y = pm3.Normal('y', mu = rot_vel, tau = 1.0/v_error**2, observed = v)
#    start = pm3.find_MAP()
#    step = pm3.NUTS()
#    trace = pm3.sample(nsamples, start=start, progressbar=True,
##                       cores = 4,
##                       tune = 2000, target_accept=.95
#                       ) #antes MDL

############         extract and plot results         ########################
pm3.summary(trace).to_csv('%s/summary_v2.csv' % dirfitsG)

y_min = np.percentile(trace.rot_vel,2.5,axis=0)
y_max = np.percentile(trace.rot_vel,97.5,axis=0)
y_fit = np.percentile(trace.rot_vel,50,axis=0)
rmin, rmax = np.amin(rad), np.amax(rad)#20.
name =   [r'$M_c$', r'$r_e$', r'$r_s$', r'$M_d$', r'$a_d$', r'$b_d$',
          r'$M_b$',r'$b_b$']

for i in name:
    az.plot_density(trace, group='prior', var_names=i, shade=0.1, point_estimate='mean',figsize = [3,5] )
        
popt = pm3.summary(trace, var_names=name)['mean'].values ### summary keys ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'mcse_mean', 'mcse_sd', 'ess_mean', 'ess_sd', 'ess_bulk', 'ess_tail', 'r_hat']
np.savetxt('%s/params.txt'%(dirfitsG), np.array(popt))
Mc, re, rs, Md, ad, bd, Mb, bb = popt 
print(popt)
Rc = re*np.sqrt(2.*(1. + (re/rs))/(1. + 3.*(re/rs)))
print('Rc = ', Rc)
r = np.linspace(rmin, rmax, 1000)
pts.plotmultiple([rad,r, r, r, r], [y_fit, 
                 np.sqrt(RC_miyamoto(r, G,  Md, ad, bd)**2 + RC_miyamoto(r, G, Mb, 0., bb)**2),
                 RC_miyamoto(r, G,  Md, ad, bd),
                 RC_miyamoto(r, G, Mb, 0., bb),
                  np.sqrt(v2_DM(r, Rc, Mc, re, rs))],
                 [r'Disk+Bulge+SFDM',
                  r'Disk+Bulge',
                  r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc, $b_d=%.2f$kpc'%(Md,ad,bd),
                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$kpc'%(Mb,bb), 
                  r'DM  $R_c = %.1f$ kpc, $M_c = %.1f \times 10^{10} M_\odot$, $r_e=%.2f$kpc, $r_s=%.1f$kpc'%(Rc, Mc, re, rs),
                  'Observed'],
                 r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_MCMC_v2.png'%(dirfitsG), ylim = (0,360),
                 xlim = (rmin,rmax), data = True, xd = rad, yd = v, err = True,
                 yerr = v_error, fill_between = True,
                 fbx = rad, fby1 = y_min, fby2 = y_max)

pts.plotmultiple([ r], [np.sqrt(v2_DM(r, Rc, Mc, re, rs))],
                 [r'DM  $R_c = %.1f$ kpc, $M_c = %.1f \times 10^{10} M_\odot$, $r_e=%.2f$kpc, $r_s=%.1f$kpc'%(Rc, Mc, re, rs),
                  'Observed'],
                 r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                 '',xlim = (rmin,rmax))

fil= open("%s/cantidades_pymc3_v2.txt"%dirfitsG,"w+")

cons = M_t(Rc, Mc, re, rs, Md, ad, bd, Mb, bb)**2*Mc*1.0e9/1.45e4
mu2 = 1./np.sqrt(np.sqrt(cons))
mu = 1./(Rc*Gentc2*M_t(Rc, Mc, re, rs, Md, ad, bd, Mb, bb))
fil.write('Nsamples = %d \r\n'%nsamples)
fil.write('Rc = %.2f kpc, \r\n'%Rc)
fil.write('M(Rc) = %f x10^{10} M_sun \r\n'%(M_t(Rc, *popt)))
fil.write('r = 2.00 kpc,  M(20kpc) = %f x10^{10} M_sun \r\n'%(M_t(20., *popt)))
fil.write('mu = %f x10^{-22} eV/c^2,  mu = %f  x10^{-22} eV/c^2 \r\n'%(mu2, mu/15655.0))
fil.write('Mhalo(300kpc) = %f x10^{12} M_sun \r\n'%(M_CNFW(300., Mc, re, rs)*1e-2))
fil.close()

print('Rc =',Rc,'kpc', 'M(Rc) =',M_t(Rc, *popt), 'x10^{10} M_sun')
print('M(20kpc) =',M_t(20., *popt), 'x10^{10} M_sun')
print('mu =',mu2, 'x10^{-22} eV/c^2')
print('mu =',mu/15655.0, 'x10^{-22} eV/c^2')
#
import matplotlib as mpl
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['lines.linewidth'] = 1
az.rcParams['plot.max_subplots']= 80

ax = az.plot_pair(trace, var_names = name, kind = 'kde',# kind = ["scatter", "kde"],
                  kde_kwargs={"fill_last": False}, marginals=True,
                  point_estimate="mode", figsize=(15, 12))
plt.savefig('%sDM_fit_MCMC_pars_v2.png'%(dirfitsG), bbox_inches='tight')
plt.show()

#print('wolfram script....')
#call(["wolframscript", "-file", "Plot_galaxy_densities_mathematica.wls"])

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#mpl.rcParams.update(mpl.rcParamsDefault)
##pm3.traceplot(trace, varnames=name)
#pm3.plot_posterior(trace, name[1],figsize=(5, 4))
#for i in(0, 1,2,3,6):
#    az.plot_density(trace, var_names=name[i], shade=0.1, point_estimate='mean' )
#    plt.savefig('%sDM_fit_MCMC_%s.png'%(dirfitsG, name[i]), bbox_inches='tight')