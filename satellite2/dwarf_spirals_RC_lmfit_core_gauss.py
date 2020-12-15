#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:20:13 2020

Fit MINIMOS CUADRADOS de Dwarf Spirals

DM              core      pars       Rc, Mc
Stellar disk    exponential     pars       MD
HI disk         exponential     pars        none

@author: jordi
"""
import numpy as np
import plots_jordi as pts
from constants_grav import G,Gentc2, hc
from pymc_tools import (todas_las_trazas, sigmas, popt_up_dw,
                        autocorr_time, save_flatchain, reports,
                        walkers_plot, plot_aceptance)
import lmfit as lm
import pandas as pd
from def_potenciales import RC_exponential,v2_GaussDM, f
#from dwarf_spirals import plot_data

dirdata = '/home/jordi/The_universal_rotation_curve_of_dwarf_disk_galaxies'
dat = pd.read_csv('%s/Galaxy_RCs_vopt.txt'%(dirdata), sep = "\t",
                skipinitialspace = True, header = None, skip_blank_lines=False)
dat.columns = ['Name','D','RD','Vopt', 'MK']

dat2 = pd.read_csv('%s/Galaxy_Best_parameters_ref_orig.txt'%(dirdata),
                  sep = "\t", skipinitialspace = True, header = 3, skip_blank_lines=False)
dat2.columns = ['Name','MD','MDKS','MHI', 'MHIK13', 'rc', 'logrho0', 'Mh', 'C']
    
dirfits = '/home/jordi/The_universal_rotation_curve_of_dwarf_disk_galaxies/Fits/gauss/'

def func(r, Rc, Mc, Md, ad, MHI):
    ve2 = v2_GaussDM(r, G, Rc, Mc)
    vdisk = RC_exponential(r, G, Md, ad)   
    vHIdisk = RC_exponential(r, G, MHI, 3.*ad)
    model =  np.sqrt(ve2 + vdisk**2 + vHIdisk**2)    
    return model

def residual(params, r, data, eps_data):
    Rc = params['Rc']
    Mc = params['Mc']
    Md = params['Md']
    ad = params['ad']   
    MHI = params['MHI']
    model = func(r, Rc, Mc, Md, ad, MHI)         
    return (data-model) / eps_data
###############################################################################
###############################################################################
ncor = 1000
ID = 'core'    
def fitting(residual, params, args, i, MCMC = False, print_res = False,
            nsamples = 1000, nwalkers = 100, thin = 1):
    out = lm.minimize(residual, params, args=args,
                      method='nelder',
#                      method = 'powell'
                      )
    if print_res==True:
#        print(lm.printfuncs.report_fit(out.params, min_correl=0.5))
#        print(out.params.pretty_print())
        print(lm.fit_report(out))
    Rc, Mc, Md, ad, MHI = list(out.params.valuesdict().values())
    popt = [Rc, Mc, Md, ad, MHI]  
    Chi2red = out.redchi
    if MCMC==True:
        res = lm.minimize(residual, args=args, method='emcee',
                          params=out.params,
                          nan_policy='omit', 
                          burn=int(.3*nsamples),
                          nwalkers = nwalkers,
                          thin=thin,
                          steps=nsamples, is_weighted=True)
        
        save_flatchain(res, dirfits , ncor, ID=ID)
        reports(res, dirfit = dirfits, ID = ID)      
        flatchain = pd.read_pickle("%sflatchain_nsol%d_%s.pkl"%(dirfits ,ncor, ID))
        
        nam = [r'$R_c$', r'$M_c$', r'$M_d$']
        traces= {r'$R_c$':np.array(flatchain['Rc']),
                 r'$M_c$':np.array(flatchain['Mc']),
                 r'$M_d$':np.array(flatchain['Md'])} 
        todas_las_trazas(traces, nam, 
                         '%sDM_fit_MCMC_emcee_pars_%d_%s.png'%(dirfits,i, ID),
                         point_estimate="mode")         
        Rc, Mc, Md, ad, MHI = list(res.params.valuesdict().values())
        popt = [Rc, Mc, Md, ad, MHI]
        unsig, _ = sigmas(res, dirfits, ncor, ID=ID)   
        popt_dw, popt_up = popt_up_dw(res, dirfits, ncor, ID=ID)
        walkers_plot(res, nam, MO, namefile='')     
        plot_aceptance(res, namefile='')  
        autocorr_time(res)

    else:                
        unsig = np.sqrt(np.diag(out.covar)) 
        nstd = 1. # to draw 1-sigma intervals
        popt2 = [popt[0],popt[1],popt[2],popt[3],popt[4]]
        popt_up = popt2 + nstd * unsig  ##Fitting parameters at 1 sigma
        popt_dw = popt2 - nstd * unsig  ##Fitting parameters at 1 sigma

        np.save('%spars_dw_%d.npy'%(dirfits,i), popt_dw)
        np.save('%spars_up_%d.npy'%(dirfits,i), popt_up)
        np.save('%spars_%d.npy'%(dirfits,i), popt)
    return popt, Chi2red, popt_dw, popt_up, unsig 

if __name__ == '__main__':
########################        SYNTHETIC RC        ###########################    
    dat = pd.read_csv('%s/URC_data.txt'%(dirdata), sep = "\t",
                    skipinitialspace = True, skip_blank_lines=False)                 
    rad, vobs, err = np.array(dat.Ri), np.array(dat.Vi), np.array(dat.dVi)   
#    Ropt = 2.5
#    MO = {'Rc' : [0.1, 5., 1.9], 'Mc' : [0.01, 0.30, 0.18], 
#          'Md' : [1e-4, 1.8e-2, 9.3e-3]}#synthetic  
#    params = lm.Parameters()    
#    params.add('Rc', min = MO['Rc'][0], max = MO['Rc'][1])
#    params.add('Mc', min = MO['Mc'][0], max = MO['Mc'][1])
#    params.add('Md', min = MO['Md'][0], max = MO['Md'][1])
#    params.add('ad', vary = False)
#    params.add('MHI', vary = False)
#    params['ad'].value = Ropt/3.2 
#    params['MHI'].value = 1.7e-2
#    popt, Chi2red, popt_dw, popt_up, _ = fitting(residual, params,
#                                                 (rad, vobs, err), 1000,
#                                                 print_res = False,
#                                                 nsamples = 10000,
#                                                 nwalkers = 50,
#                                                 MCMC = True,
#                                                 thin = 10
#                                                 )
    popt = np.load('%spopt_nsol%d_%s.npy'%(dirfits, ncor, ID)).T
    popt_dw = np.load('%spoptdw_nsol%d_%s.npy'%(dirfits, ncor, ID))
    popt_up = np.load('%spoptup_nsol%d_%s.npy'%(dirfits, ncor, ID))
    Rc, Mc, Md, ad, MHI = popt            
    v_disco = RC_exponential(rad, G, Md, ad)
    v_discoHI = RC_exponential(rad, G, MHI, 3.0*ad)            
    v_DM_halo = np.sqrt(v2_GaussDM(rad, G, Rc, Mc))
    labbar = r'Disc $M_d =%.1f \times 10^{7} M_\odot$, $a_d =%.2f$kpc'%(Md*1e3, ad)
    mu = np.sqrt(1/(Gentc2*Rc*f(Rc, Mc, Rc)))           
    labDM = r'DM $\mu = %.1f \times 10^{-24}$eV/$c^2$, $M_c =%.1f \times 10^{8} M_\odot$'%(mu*hc*1e24, Mc*1e2)
    labHI = r'HI disc, $M_{HI}= %.1f \times 10^{8} M_\odot$'%(MHI*1e2)
    pts.jordi_style()
    pts.plotmultiple([rad, rad, rad, rad],
                     [func(rad, *popt), v_disco, v_discoHI, v_DM_halo],
                     ['fit', labbar, labHI, labDM], r'$r$(kpc)',r'$v$(km/s)','',
                     '%sSynthetic_RC_%s.png'%(dirfits, ID), data=True, xd=rad,
                     yd=vobs, err=True, yerr=err, xv=[ad, Rc],
                     fill_between = True,fbx = rad,
                     fby1 = func(rad, *popt_dw, ad, MHI),
                     fby2 = func(rad, *popt_up, ad, MHI))
    R0 = Rc
    print('rho_halo(0) = ',Mc/(np.sqrt(np.pi)**3*Rc**3)*10., 'M_sun/pc^3')  
    print('log10( rho_halo(0) )= ',np.log10(Mc/(np.sqrt(np.pi)**3*Rc**3)*10.0**10), 'M_sun/kpc^3') 
    print('rho_halo(0)r0(0) = ',Mc/(np.sqrt(np.pi)**3*Rc**3)*10.*R0*1e3, 'M_sun/pc^2') 
    print('log10( Md )= ', np.log10(Md*10.0**10), 'M_sun') 
    print('mu = ',mu ,'1/kpc' )
    print('mu = ',mu*hc ,'ev/c^2' )