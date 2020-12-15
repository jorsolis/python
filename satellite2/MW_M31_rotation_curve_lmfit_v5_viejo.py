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
from scipy.interpolate import interp1d
import numpy as np
from def_potenciales import exp_bulge, RC_exponential
import lmfit as lm

from pymc_tools import (todas_las_trazas, sigmas, popt_up_dw, gelman_rubin,
                        autocorr_time, save_flatchain, reports,
                        walkers_plot, plot_aceptance, plot_tau_estimates) 
from constants_grav import G, hc, c
###############################################################################
np.random.seed(12345)
galaxy = 'Milky Way'
nsamples = 1e4

def v_multi_SFDM(r, rlam, mu, ncor):
    paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d_viejo.npy'%(paco,ncor,ncor))    
    xn = x2/(mu*rlam)
    vdmn = rlam*vdm*c
    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate"
                  )
    return ve(r)

#nsamples = 5e5
paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/mixSFDM/'%dirdata
    dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 
    rad, v, v_error = dataGrandMW.T
    
ncor = 2  ## 4 y 5 m = 1
print(ncor)
    
MO = {'rlam' : [0.001, 0.1, 0.02], 'mu' : [1.556, 155.6,  15.5655], 
      'Md' : [1e-2, 30., 8.15], 'ad' : [0.1, 5., 3.86],
      'Mb': [.1, 1.5, 0.86], 'bb' : [0.05, 0.5, 0.15]} #### no extrapolar DM

def func(r,rlam,mu, Md, ad, Mb, bb):
    vh = v_multi_SFDM(r, rlam, mu, ncor)
    vdisk = RC_exponential(r, G, Md, ad)
    vbul = exp_bulge(r, G, Mb, bb)
    model =  np.sqrt(vh**2 + vdisk**2 + vbul**2) 
    return model 

def residual(params, r, data, eps_data):
    rlam = params['rlam']
    mu = params['mu']
    Md = params['Md']
    ad = params['ad']
    Mb = params['Mb']
    bb = params['bb'] 
    model = func(r,rlam,mu, Md, ad, Mb, bb)
    return (data-model) / eps_data
def modelo(params, r):
    rlam = params['rlam']
    mu = params['mu']
    Md = params['Md']
    ad = params['ad']
    Mb = params['Mb']
    bb = params['bb']
    return func(r,rlam,mu, Md, ad, Mb, bb) 
    
params = lm.Parameters()
params.add('rlam', value = MO['rlam'][2], min = MO['rlam'][0], max = MO['rlam'][1])
params.add('mu', value = MO['mu'][2], min = MO['mu'][0], max = MO['mu'][1])
params.add('Md', value = MO['Md'][2], min = MO['Md'][0], max = MO['Md'][1])
params.add('ad', value = MO['ad'][2], min = MO['ad'][0], max = MO['ad'][1])
params.add('Mb', value = MO['Mb'][2], min = MO['Mb'][0], max = MO['Mb'][1])
params.add('bb', value = MO['bb'][2], min = MO['bb'][0], max = MO['bb'][1])

################################################################################
#############################        MCMC        ##############################3
###############################################################################
res = lm.minimize(residual, args=(rad, v, v_error), method='emcee',
                  nan_policy='omit', burn = int(.3*nsamples),
                  steps = nsamples, nwalkers = 200,
                  params = params, is_weighted = True )
ID = 'v5'
name =[r'$\sqrt{\lambda}$', r'$\mu$', r'$M_d$', r'$a_d$',r'$M_b$',r'$b_b$']

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
popt_up = np.load('%spoptup_nsol%d_v5.npy'%(dirfitsG, ncor))
popt_dw = np.load('%spoptdw_nsol%d_v5.npy'%(dirfitsG, ncor))
popt = np.load('%spopt_nsol%d_v5.npy'%(dirfitsG, ncor))
rlam, mu, Md, ad, Mb, bb = popt

rmin, rmax = 1e-1, np.amax(rad)

r = np.linspace(rmin, rmax, 100000)
vdisk = RC_exponential(r, G, Md, ad)
vbulge = exp_bulge(r, G, Mb, bb)

rlamu, muu, Mdu, adu, Mbu, bbu = popt_up
rlamd, mud, Mdd, add, Mbd, bbd = popt_dw

y_min= func(r, rlamd, mud, Mdd, add, Mbd, bbd)
y_max= func(r, rlamu, muu, Mdu, adu, Mbu, bbu)

DMlab =  r'DM  $\sqrt{\lambda} = %.5f$, $\mu = %.2f \times 10^{-25}$ eV/$c^2$'%(rlam,mu*hc*1e25)
dsclab = r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad)
blglab= r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3)

pts.plotmultiple([r, r, r, r, r, r, r], 
#                 [modelo(res.params,r),
                  [func(r,rlam,mu, Md, ad, Mb, bb) ,
                  np.sqrt(vdisk**2 + vbulge**2 ),
                  v_multi_SFDM(r, rlam, mu, ncor), vdisk, vbulge],
                 [r'Disc+Bulge+BH+SFDM', r'Disc+Bulge+ BH',
                  DMlab, dsclab, blglab], r'$r$(kpc)',
                  r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_emcee_nsol%d_v5.png'%(dirfitsG, ncor),
                 xlim = (rmin,rmax), #ylim = (20,500),
                 data = True, xd = rad, yd = v, err = True, yerr = v_error,
                 fill_between = True,fbx = r, fby1 = y_min, fby2 = y_max,
                 logx = True, logy = True, xv = [bb, ad])
pts.plotmultiple([r, r, r, r, r, r, r],
#                 [modelo(res.params,r),
                  [ func(r,rlam,mu, Md, ad, Mb, bb),
                  np.sqrt(vdisk**2 + vbulge**2),
                  v_multi_SFDM(r, rlam, mu, ncor)
                  , vdisk, vbulge],
                 [r'Disc+Bulge+BH+SFDM', r'Disc + Bulge + BH', DMlab, dsclab, 
                  blglab],
#                 [],
                 r'$r$(kpc)', r'$v$(km/s)', '',
                 '%sDM_fit_emcee2_nsol%d_v5.png'%(dirfitsG, ncor), xlim = (rmin, 27),
#                 ylim = (0,300),
                 data = True, xd = rad, yd = v, err = True, 
                 yerr = v_error,
                 fill_between = True, fbx = r, fby1 = y_min, fby2 = y_max)

#import pandas as pd
#flatchain = pd.read_pickle("%sflatchain_nsol%d_%s.pkl"%(dirfitsG,ncor, 'v5'))
#lanp = 100./7.5
#name =[r'$\lambda$', r'$\hat{\mu}$', r'$M_d$', r'$a_d$',r'$M_b$',r'$a_b$']
#traces= {r'$\lambda$':np.array(flatchain['rlam']*1e3/lanp),
#         r'$\hat{\mu}$':np.array(flatchain['mu']),
#           r'$M_d$':np.array(flatchain['Md']),
#           r'$a_d$':np.array(flatchain['ad']),
#           r'$M_b$':np.array(flatchain['Mb']),
#           r'$a_b$':np.array(flatchain['bb'])}
##
#todas_las_trazas(traces, name, 
#                 '%sDM_fit_MCMC_emcee_pars_nsol%d_v5.png'%(dirfitsG, ncor),
#                 point_estimate="mean")