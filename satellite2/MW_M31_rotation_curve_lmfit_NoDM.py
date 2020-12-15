#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

Fit MCMC de la via lactea 

DM              mixSFDM                 pars      mu, lambda
Disco           Razor exponential       pars      Md, ad
Bulge           Exponential             pars      Mb, bb

Data            Table_GrandRC

@author: jordi
"""
import plots_jordi as pts
import numpy as np
from def_potenciales import exp_bulge, RC_exponential
import lmfit as lm
from pymc_tools import (todas_las_trazas, sigmas, popt_up_dw, gelman_rubin,
                        autocorr_time, save_flatchain, reports,
                        walkers_plot, plot_aceptance, plot_tau_estimates) 
from constants_grav import G
import pandas as pd
###############################################################################
np.random.seed(12345)
galaxy = 'Milky Way'
#galaxy = 'M31'
nsteps = 1e3
ncor = 1

if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/noDM/'%(dirdata)

    dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 
    rad, v, v_error = dataGrandMW.T
    rad, v, v_error = np.array(rad),np.array(v),np.array(v_error)
    
MO = {'Md' : [1e-2, 30., 8.15], 'ad' : [0.1, 5., 3.86],
          'Mb': [.1, 1.5, 0.86], 'bb' : [0.05, 0.5, 0.15]} #### no extrapolar DM

def func(r, Md, ad, Mb, bb):
    vdisk = RC_exponential(r, G, Md, ad)
    vbul = exp_bulge(r, G, Mb, bb)
    model =  np.sqrt(vdisk**2 + vbul**2) 
    return model  
    
def residual(params, r, data, eps_data):
    Md = params['Md']
    ad = params['ad']
    Mb = params['Mb']
    bb = params['bb'] 
    model = func(r, Md, ad, Mb, bb)
    return (data-model) / eps_data
def modelo(params, r):
    Md = params['Md']
    ad = params['ad']
    Mb = params['Mb']
    bb = params['bb']
    return func(r, Md, ad, Mb, bb) 

name =[r'$M_d$', r'$a_d$', r'$M_b$',r'$b_b$']

ID = 'v5'
params = lm.Parameters()
params.add('Md', min = MO['Md'][0], max = MO['Md'][1])
params.add('ad', min = MO['ad'][0], max = MO['ad'][1])
params.add('Mb', min = MO['Mb'][0], max = MO['Mb'][1])
params.add('bb', min = MO['bb'][0], max = MO['bb'][1])

params['Mb'].value = MO['Mb'][2]
params['bb'].value = MO['bb'][2]
params['Md'].value = MO['Md'][2]
params['ad'].value = MO['ad'][2]

#out = lm.minimize(residual, params, args=(rad, v, v_error),
#                  method='nelder',
##                 method = 'powell',
#                  nan_policy='omit')
#################################################################################
##############################        MCMC        ##############################3
################################################################################
res = lm.minimize(residual, args=(rad, v, v_error), method='emcee',
                  nan_policy='omit', burn = int(.3*nsteps),
                  steps = nsteps, nwalkers = 200, thin = 10,
#                  params=out.params,
                  params = params, 
                  is_weighted = True)

save_flatchain(res, dirfitsG, ncor, ID= ID)
reports(res, dirfit = dirfitsG, ID = 'nsol%d_%s'%(ncor, ID)) 
unsigma, dossigma = sigmas(res, dirfitsG, ncor, ID=ID)
_, _ = popt_up_dw(res, dirfitsG, ncor, ID=ID)
autocorr_time(res)
print('Gelman-Rubin', gelman_rubin(res.chain))  
Chain = res.chain[:,:,0].T
plot_tau_estimates(Chain, namefile ='nsol_%d'%ncor, dirfit = dirfitsG)
plot_aceptance(res, namefile='%saceptance_nsol_%d_%s'%(dirfitsG, ncor, ID))
walkers_plot(res, name, MO, namefile='%swalker_nsol%d_%s'%(dirfitsG, ncor, ID))
#################################################################################
##############################        PLOTS       ##############################3
#################################################################################
popt = np.load('%spopt_nsol%d_%s.npy'%(dirfitsG, ncor, ID))
popt_up = np.load('%spoptup_nsol%d_%s.npy'%(dirfitsG, ncor, ID))
popt_dw = np.load('%spoptdw_nsol%d_%s.npy'%(dirfitsG, ncor, ID))

Md, ad, Mb, bb = popt
rmin, rmax = 1e-1, np.amax(rad)

r = np.linspace(rmin, rmax, 100000)
vdisk = RC_exponential(r, G, Md, ad)
vbulge = exp_bulge(r, G, Mb, bb)

Mdu, adu, Mbu, bbu = popt_up
Mdd, add, Mbd, bbd = popt_dw

y_min= func(r, Mdd, add, Mbd, bbd)
y_max= func(r, Mdu, adu, Mbu, bbu)

#DMlab =  r'DM  $\sqrt{\lambda} = %.3f \times 10^{-3}$, $\mu = %.2f \times 10^{-24}$ eV/$c^2$'%(rlam*1e3,mu*hc*1e24)
dsclab = r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad)
blglab= r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3)

pts.plotmultiple([r, r, r, r, r, r, r], #[modelo(res.params,r),
                  [func(r,Md, ad, Mb, bb) ,
#                  v_multi_SFDM(r, rlam, mu, ncor),
                  vdisk, vbulge],
                 [r'Disc+Bulge',
#                  DMlab, 
                  dsclab, blglab], r'$r$(kpc)',
                  r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_emcee_nsol%d_%s.png'%(dirfitsG, ncor, ID),
                 xlim = (rmin,rmax), #ylim = (20,500),
                 data = True, xd = rad, yd = v, err = True, yerr = v_error,
                 fill_between = True,fbx = r, fby1 = y_min, fby2 = y_max,
                 logx = True, logy = True, xv = [bb, ad])
pts.plotmultiple([r, r, r, r, r, r, r], #[modelo(res.params,r),
                  [func(r, Md, ad, Mb, bb), vdisk, vbulge],
                 [r'Disc+Bulge',dsclab, blglab],
                 r'$r$(kpc)', r'$v$(km/s)', '',
                 '%sDM_fit_emcee2_nsol%d_%s.png'%(dirfitsG, ncor, ID),
                 xlim = (rmin, 27), ylim = (0,300), 
                 data = True, xd = rad, yd = v, err = True, 
                 yerr = v_error, xv = [ad],
                 fill_between = True, fbx = r, fby1 = y_min, fby2 = y_max)

flatchain = pd.read_pickle("%sflatchain_nsol%d_%s.pkl"%(dirfitsG,ncor, ID))

traces= {r'$M_d$':np.array(flatchain['Md']),
         r'$a_d$':np.array(flatchain['ad']), 
         r'$M_b$':np.array(flatchain['Mb']),
         r'$b_b$':np.array(flatchain['bb'])}
todas_las_trazas(traces, name, 
                 '%sDM_fit_MCMC_emcee_pars_nsol%d_%s.png'%(dirfitsG, ncor, ID),
                 point_estimate="mean")
