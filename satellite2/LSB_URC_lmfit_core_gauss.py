#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:20:13 2020

LSB URC  (5 velocity bins)

DM              core            pars       Rc, Mc
Stellar disk    exponential     pars       MD
HI disk         exponential     pars       none

@author: jordi
"""
import numpy as np
import plots_jordi as pts
from scipy.interpolate import interp1d
from constants_grav import G,Gentc2, c, hc, mukpc, letr
from pymc_tools import (todas_las_trazas, sigmas, popt_up_dw, walkers_plot,
                        plot_aceptance, plot_tau_estimates, save_flatchain,
                        reports,autocorr_time, gelman_rubin) 
import lmfit as lm
import pandas as pd
from def_potenciales import RC_exponential,v2_GaussDM, f
from URC_LSB import plot_data

dirdata = '/home/jordi/LSB_galaxies/URC'

def func(r, Rc, Mc, Md, ad):
    ve2 = v2_GaussDM(r, G, Rc, Mc) 
    vdisk = RC_exponential(r, G, Md, ad)   
    model =  np.sqrt(ve2 + vdisk**2) 
    return model

Vin = 127.
rin = 1.6
def bul(r, alpha):
    return np.sqrt(alpha*Vin**2*rin/r)
    
def func2(r, Rc, Mc, Md, ad, alpha):
    ve2 = v2_GaussDM(r, G, Rc, Mc) 
    vdisk = RC_exponential(r, G, Md, ad)   
    vbulge = bul(r,alpha)
    model =  np.sqrt(ve2 + vdisk**2 + vbulge**2)   
    return model

def residual(params, r, data, eps_data):
    Rc = params['Rc']
    Mc = params['Mc']
    Md = params['Md']
    ad = params['ad']   
    model = func(r, Rc, Mc, Md, ad)        
    return (data-model) / eps_data

def residual2(params, r, data, eps_data):
    Rc = params['Rc']
    Mc = params['Mc']
    Md = params['Md']
    ad = params['ad']  
    alpha = params['alpha']
    model = func2(r, Rc, Mc, Md, ad, alpha)        
    return (data-model) / eps_data

###############################################################################
###############################################################################   
def fitting(residual, params, args, i, MCMC = False, print_res = False,
            nsteps = 1000, nwalkers = 100, thin = 1):
    out = lm.minimize(residual, params, args=args,
#                      method='nelder',
                      method = 'powell'
                      )
    if print_res==True:
#        print(lm.printfuncs.report_fit(out.params, min_correl=0.5))
#        print(out.params.pretty_print())
        print(lm.fit_report(out))
    popt = [list(out.params.valuesdict().values())]
    Chi2red = out.redchi
    if MCMC==True:
        res = lm.minimize(residual, args=args, method='emcee',
                          nan_policy='omit', burn=int(.3*nsteps),
                          nwalkers = nwalkers, steps=nsteps, thin=thin,
                          params=out.params,
#                          params = params,
                          is_weighted=True)
        save_flatchain(res, dirfits, i, ID=ID)
        reports(res, dirfit = dirfits, ID =ID)      
        flatchain = pd.read_pickle("%sflatchain_nsol%d_%s.pkl"%(dirfits ,i, ID))

        if i==5:
            nam = [r'$R_c$',  r'$M_c$', r'$M_d$', r'$\alpha$']
            traces= {nam[0]:np.array(flatchain['Rc']),
                     nam[1]:np.array(flatchain['Mc']), 
                     nam[2]:np.array(flatchain['Md']),
                     nam[3]:np.array(flatchain['alpha'])}             
        else:
            nam = [r'$R_c$',  r'$M_c$', r'$M_d$']
            traces= {nam[0]:np.array(flatchain['Rc']),
                     nam[1]:np.array(flatchain['Mc']), 
                     nam[2]:np.array(flatchain['Md'])} 
        
        todas_las_trazas(traces, nam, '%sDM_fit_MCMC_emcee_pars_vbin%d.png'%(dirfits,i),
                         point_estimate="mode")
     
        unsig = sigmas(res, dirfits, i, ID=ID) 
        popt_dw, popt_up = popt_up_dw(res, dirfits, i, ID=ID)
        
        walkers_plot(res, nam, MO, namefile='')     
        plot_aceptance(res, namefile='')  
        autocorr_time(res)
        Chain = res.chain[:,:,0].T
        plot_tau_estimates(Chain, namefile =ID, dirfit = dirfits)
        popt = np.load('%spopt_nsol%d_%s.npy'%(dirfits, i, ID))
        print('Gelman-Rubin', gelman_rubin(Chain))
        
    else:                
        unsig = np.sqrt(np.diag(out.covar)) 
        nstd = 1. # to draw 1-sigma intervals
        popt2 = [popt[0],popt[1],popt[2]]
        popt_up = popt2 + nstd * unsig  ##Fitting parameters at 1 sigma
        popt_dw = popt2 - nstd * unsig  ##Fitting parameters at 1 sigma
        np.save('%spars_dw_nsol%d_%d.npy'%(dirfits, i, i), popt_dw)
        np.save('%spars_up_nsol%d_%d.npy'%(dirfits, i,i), popt_up)
    return popt, Chi2red, popt_dw, popt_up, unsig 

if __name__ == '__main__':
#    plot_data()    
    RD = np.array([1.7, 2.2, 3.7, 4.5, 7.9])
#    for i in range(2,,1):
    
    i = 4
    
    ID = 'MCMC_core_vbin%d'%(i)
   
    dirfits = '/home/jordi/LSB_galaxies/URC/Fits_gauss/bin%d/'%i
    _, rad,_, vobs, err = np.loadtxt('%s/bin%d.txt'%(dirdata, i)).T        
    if i ==1:
        MO = {'Rc' : [0.1, 10.], 'Mc' : [0.01, 1.], 
              'Md' : [1e-3, 1.]}
    elif i ==2:
        MO = {'Rc' : [0.1, 30.], 'Mc' : [1., 10.], 
              'Md' : [1e-3, 1.]}
    elif i == 3:
        MO = {'Rc' : [0.1, 30.], 'Mc' : [1., 20.], 
              'Md' : [1e-3, 1e1]}       
    else:
        MO = {'Rc' : [0.1, 1000.], 'Mc' : [1., 1e3], 
              'Md' : [1e-3, 1e2]} 

    params = lm.Parameters()
    params.add('Rc', min = MO['Rc'][0], max = MO['Rc'][1])
    params.add('Mc', min = MO['Mc'][0], max = MO['Mc'][1])
    params.add('Md', min =  MO['Md'][0], max = MO['Md'][1])
    params.add('ad', vary = False)
    if i == 5 :
        params.add('alpha', vary = True, min = 0.2, max = 1.)
        resid = residual2
    else:
        resid = residual          
    params['ad'].value = RD[i - 1]  
    popt, Chi2red, popt_dw, popt_up, unsig = fitting(resid, params, 
            (rad, vobs, err), i, MCMC = True,
            nsteps = 10000, nwalkers = 50, thin = 20)
#        
    popt = np.load('%spopt_nsol%d_%s.npy'%(dirfits, i, ID)).T
    popt_dw = np.load('%spoptdw_nsol%d_%s.npy'%(dirfits, i, ID))
    popt_up = np.load('%spoptup_nsol%d_%s.npy'%(dirfits, i, ID))        

    if i == 5 :
        Rc, Mc, Md, ad, alpha = popt
        Rc_mi, Mc_mi, Md_mi, alpha_mi = popt_dw
        Rc_ma, Mc_ma, Md_ma, alpha_ma =  popt_up
    else:
        Rc, Mc, Md, ad = popt
        Rc_mi, Mc_mi, Md_mi = popt_dw
        Rc_ma, Mc_ma, Md_ma = popt_up
        
    v_disco = RC_exponential(rad, G, Md, ad)         
    v_DM_halo = np.sqrt(v2_GaussDM(rad, G, Rc, Mc))
    fit = np.sqrt(v_disco**2 + v_DM_halo**2)        
    labbar = r'Disc $M_d =%.4f \times 10^{7} M_\odot$, $a_d =%.2f$kpc'%(Md*1e3, ad)           
    mu = np.sqrt(1/(Gentc2*Rc*f(Rc, Mc, Rc)))  
    parss = r'DM $\mu = %.1f \times 10^{-24}$eV/$c^2$, $M_c =%.1f \times 10^{8} M_\odot$'%(mu*hc*1e24, Mc*1e2)
    
    if i == 5 :
        Y = [np.sqrt(v_disco**2 + v_DM_halo**2 + bul(rad,alpha)**2), v_disco, v_DM_halo, bul(rad,alpha)]
        L = ['fit', labbar, parss, r'bulge $\alpha = %.2f$'%alpha]
        y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 + v2_GaussDM(rad, G, Rc_mi, Mc_mi) + bul(rad, alpha_mi)**2)
        y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 + v2_GaussDM(rad, G, Rc_ma, Mc_ma) + bul(rad, alpha_ma)**2)             
    else:
        Y = [fit, v_disco, v_DM_halo]
        L = ['fit', labbar, parss]
        y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 + v2_GaussDM(rad, G, Rc_mi, Mc_mi))
        y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 + v2_GaussDM(rad, G, Rc_ma, Mc_ma)) 

    pts.plotmultiple([rad, rad, rad, rad], Y, L, r'$r$(kpc)',r'$v$(km/s)',
                     '',
                     '%sSynthetic_RC_%s.png'%(dirfits, ID), data=True, xd=rad,
                     yd=vobs, err=True, yerr=err, xv=[ad, Rc],
                     fill_between = True,fbx = rad, fby1 = y_min,
                     fby2 = y_max)
    R0 = Rc
    print(letr['rho'], '_halo(0) = ',Mc/(np.sqrt(np.pi)**3*Rc**3)*10., 'M_sun/pc^3')  
    print('log10(', letr['rho'],'_halo(0) )= ',np.log10(Mc/(np.sqrt(np.pi)**3*Rc**3)*10.0**10), 'M_sun/kpc^3') 
    print(letr['rho'], '_halo(0)r0(0) = ',Mc/(np.sqrt(np.pi)**3*Rc**3)*10.*R0*1e3, 'M_sun/pc^2') 
    print('log10( Md )= ', np.log10(Md*10.0**10), 'M_sun') 
    print(letr['mu'], '=', mu ,'1/kpc' )
    print(letr['mu'], '=', mu*hc ,'eV/c^2')