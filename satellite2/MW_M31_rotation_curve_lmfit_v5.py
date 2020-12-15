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
from def_potenciales import exp_bulge, RC_exponential, v_multi_SFDM
import lmfit as lm
from pymc_tools import (todas_las_trazas, sigmas, popt_up_dw, gelman_rubin,
                        autocorr_time, save_flatchain, reports,
                        walkers_plot, plot_aceptance, plot_tau_estimates) 
from constants_grav import G, hc, mukpc, c
import pandas as pd
from pot_paco import plots_SFDM_density_interpolado
from scipy.interpolate import interp1d
###############################################################################
np.random.seed(12345)
galaxy = 'Milky Way'
#galaxy = 'M31'
nsteps = 1e4

mass = 25


def ord_data(dataMW, errMW):
    r = dataMW[:,0]
    vel = dataMW[:,3]
    err = errMW[0,:]    
    sortindex = np.argsort(r, axis = 0)
    rord = np.sort(r, axis= 0)    
    velord = vel[sortindex]
    errord = err[sortindex]
    return rord, velord, errord

if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/mixSFDMmu%d/'%(dirdata, mass)

    dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 
    rad, v, v_error = dataGrandMW.T
    rad, v, v_error = np.array(rad),np.array(v),np.array(v_error)
#    rad, v, v_error = np.array(rad[:118]), np.array(v[:118]), np.array(v_error[:118])
      
elif galaxy == 'M31':
    dirdata = '/home/jordi/satellite/M31_rotation_curve_data'
    dirfitsG = '%s/Fits/mixSFDMmu%d/'%(dirdata, mass)
    dataM31= np.loadtxt("%s/M31_rotation_curve_data.txt"%dirdata) 
    _, rad, v, v_error = dataM31.T
    rad, v, v_error = np.array(rad),np.array(v),np.array(v_error)    

################################################################################
####################            POT PACO        ################################
################################################################################
paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
soldic = {1:{'n':2, 'l':1, 'm':0}, 2:{'n':2, 'l':1, 'm':0}, 6:{'n':2, 'l':1, 'm':0},
          3:{'n':2, 'l':1, 'm':1}, 4:{'n':2, 'l':1, 'm':1}, 5:{'n':2, 'l':1, 'm':1}}
ncor = 2
n = soldic[ncor]['n']
l = soldic[ncor]['l']
m = soldic[ncor]['m']
##############################################################################
if mass == 25:
    MO = {'rlam' : [1e-4, 0.1, 0.02], 'mu' : [mukpc[26], mukpc[24], mukpc[25]], 
          'Md' : [1e-2, 30., 8.15], 'ad' : [0.1, 5., 3.86],
          'Mb': [.1, 1.5, 0.86], 'bb' : [0.05, 0.5, 0.15]} #### no extrapolar DM
elif mass == 24:
    MO = {'rlam' : [1e-5, 0.001, 0.0002], 'mu' : [mukpc[24], mukpc[23], 500.], 
          'Md' : [1e-2, 30., 8.15], 'ad' : [1., 5., 3.86],#### no extrapolar DM
          'Mb': [.1, 1.5, 0.86], 'bb' : [0.05, 0.5, .15]}
  
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

name =[r'$\sqrt{\lambda}$', r'$\mu$',
       r'$M_d$', r'$a_d$',
       r'$M_b$',r'$b_b$'
       ]
ID = 'v5'
params = lm.Parameters()
params.add('rlam', min = MO['rlam'][0], max = MO['rlam'][1])
params.add('mu', min = MO['mu'][0], max = MO['mu'][1])#prueba 10**-24
params.add('Md', min = MO['Md'][0], max = MO['Md'][1])
params.add('ad', min = MO['ad'][0], max = MO['ad'][1])
params.add('Mb', min = MO['Mb'][0], max = MO['Mb'][1])
params.add('bb', min = MO['bb'][0], max = MO['bb'][1])

params['rlam'].value = MO['rlam'][2]
params['mu'].value = MO['mu'][2]
params['Mb'].value = MO['Mb'][2]
params['bb'].value = MO['bb'][2]
#params['Mb'].vary = False 
#params['bb'].vary = False
params['Md'].value = MO['Md'][2]
params['ad'].value = MO['ad'][2]

out = lm.minimize(residual, params, args=(rad, v, v_error),
                  method='nelder',
#                 method = 'powell',
                  nan_policy='omit')
#################################################################################
##############################        MCMC        ##############################3
################################################################################
#res = lm.minimize(residual, args=(rad, v, v_error), method='emcee',
#                  nan_policy='omit', burn = int(.3*nsteps),
#                  steps = nsteps, nwalkers = 200, thin = 10,
#                  params=out.params,
##                  params = params, 
#                  is_weighted = True)
#
#save_flatchain(res, dirfitsG, ncor, ID= ID)
#reports(res, dirfit = dirfitsG, ID = 'nsol%d_%s'%(ncor, ID)) 
#unsigma, dossigma = sigmas(res, dirfitsG, ncor, ID=ID)
#_, _ = popt_up_dw(res, dirfitsG, ncor, ID=ID)
#autocorr_time(res)
#print('Gelman-Rubin', gelman_rubin(res.chain))  
#Chain = res.chain[:,:,0].T
#plot_tau_estimates(Chain, namefile ='nsol_%d'%ncor, dirfit = dirfitsG)
#plot_aceptance(res, namefile='%saceptance_nsol_%d_%s'%(dirfitsG, ncor, ID))
#walkers_plot(res, name, MO, namefile='%swalker_nsol%d_%s'%(dirfitsG, ncor, ID))
#################################################################################
##############################        PLOTS       ##############################3
#################################################################################
popt = np.load('%spopt_nsol%d_%s.npy'%(dirfitsG, ncor, ID))
popt_up = np.load('%spoptup_nsol%d_%s.npy'%(dirfitsG, ncor, ID))
popt_dw = np.load('%spoptdw_nsol%d_%s.npy'%(dirfitsG, ncor, ID))

rlam, mu, Md, ad, Mb, bb = popt
rmin, rmax = 1e-1, np.amax(rad)

r = np.linspace(rmin, rmax, 100000)
vdisk = RC_exponential(r, G, Md, ad)
vbulge = exp_bulge(r, G, Mb, bb)
vDM = v_multi_SFDM(r, rlam, mu, ncor)

rlamu, muu, Mdu, adu, Mbu, bbu = popt_up
rlamd, mud, Mdd, add, Mbd, bbd = popt_dw

#rlamu, muu, Mdu, adu = popt_up
#rlamd, mud, Mdd, add = popt_dw
#Mbu, bbu = Mb, bb
#Mbd, bbd = Mb, bb

y_min= func(r, rlamd, mud, Mdd, add, Mbd, bbd)
y_max= func(r, rlamu, muu, Mdu, adu, Mbu, bbu)

DMlab =  r'DM  $\sqrt{\lambda} = %.3f \times 10^{-3}$, $\mu = %.2f \times 10^{-24}$ eV/$c^2$'%(rlam*1e3,mu*hc*1e24)
dsclab = r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad)
blglab= r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3)

pts.plotmultiple([r, r, r, r, r, r, r], #[modelo(res.params,r),
                  [func(r,rlam,mu, Md, ad, Mb, bb) ,
                  np.sqrt(vdisk**2 + vbulge**2 ), vDM, vdisk, vbulge],
                 [r'Disc+Bulge+BH+SFDM', r'Disc+Bulge+ BH',
                  DMlab, dsclab, blglab], r'$r$(kpc)',
                  r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_emcee_nsol%d_%s.png'%(dirfitsG, ncor, ID),
                 xlim = (rmin,rmax), #ylim = (20,500),
                 data = True, xd = rad, yd = v, err = True, yerr = v_error,
                 fill_between = True,fbx = r, fby1 = y_min, fby2 = y_max,
                 logx = True, logy = True, xv = [bb, ad])
pts.plotmultiple([r, r, r, r, r, r, r], #[modelo(res.params,r),
                  [func(r,rlam,mu, Md, ad, Mb, bb),
                   np.sqrt(vdisk**2 + vbulge**2), vDM, vdisk, vbulge],
                 [r'Disc+Bulge+BH+SFDM', r'Disc + Bulge + BH', DMlab, dsclab, 
                  blglab],
                 r'$r$(kpc)', r'$v$(km/s)', '',
                 '%sDM_fit_emcee2_nsol%d_%s.png'%(dirfitsG, ncor, ID),
                 xlim = (rmin, 27), ylim = (0,300), 
                 data = True, xd = rad, yd = v, err = True, 
                 yerr = v_error, xv = [ad],
                 fill_between = True, fbx = r, fby1 = y_min, fby2 = y_max)

flatchain = pd.read_pickle("%sflatchain_nsol%d_%s.pkl"%(dirfitsG,ncor, ID))

traces= {r'$\sqrt{\lambda}$':np.array(flatchain['rlam']),
         r'$\mu$':np.array(flatchain['mu']),
         r'$M_d$':np.array(flatchain['Md']),
         r'$a_d$':np.array(flatchain['ad']), 
         r'$M_b$':np.array(flatchain['Mb']),
         r'$b_b$':np.array(flatchain['bb'])}
todas_las_trazas(traces, name, 
                 '%sDM_fit_MCMC_emcee_pars_nsol%d_%s.png'%(dirfitsG, ncor, ID),
                 point_estimate="mean")
#plots_SFDM_density_interpolado(ncor, di = 'baja_dens/', ref = 2, m=m,
#                               lam = rlam**2, mu = mu)
#print('rlam*mu = ', rlam*mu, 'rlam = ', rlam, 'mu = ', mu)