#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:20:13 2020

LSB URC  (5 velocity bins)

DM              multi-SFDM      pars       mu, rlam
Stellar disk    exponential     pars       MD
HI disk         exponential     pars        none

@author: jordi
"""
import numpy as np
import plots_jordi as pts
from constants_grav import G, hc, mukpc
from pymc_tools import (todas_las_trazas, sigmas, popt_up_dw, walkers_plot,
                        plot_aceptance, plot_tau_estimates, save_flatchain,
                        reports,autocorr_time, gelman_rubin) 
import lmfit as lm
import pandas as pd
from def_potenciales import RC_exponential
#from URC_LSB import plot_data
from Os_tools import check_path
dirdata = '/home/jordi/LSB_galaxies/URC'

###############################################################################
###################       Multi 100, exc       ################################
###############################################################################
n = 2; l = 1; m = 1
state = 211
ID = 'multi_j'
from def_potenciales import multi_SFDM, multistate
###############################################################################
###############################################################################
###################            POT PACO        ################################
###############################################################################
#from scipy.interpolate import interp1d
#paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
#dirmixshoot = '/home/jordi/satellite/mix_shooting'
#soldic = {1:{'n':2, 'l':1, 'm':0}, 2:{'n':2, 'l':1, 'm':0}, 6:{'n':2, 'l':1, 'm':0},
#          3:{'n':2, 'l':1, 'm':1}, 4:{'n':2, 'l':1, 'm':1}, 5:{'n':2, 'l':1, 'm':1}}
#ncor = 3
#n = soldic[ncor]['n']
#l = soldic[ncor]['l']
#m = soldic[ncor]['m']
#
#def v_muliSFDM(r, rlam, mu):
#    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))    
#    xn = x2[:-150]/(mu*rlam)
#    vdmn = rlam*vdm[:-150]*c
#    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate",
#                  assume_sorted=True)
#    return ve(r)
################################################################################
################          MULTISTATE 32m        #############################
##############################################################################
#from scipy.interpolate import interp1d
#ncor = 103
#n = 3
#l = 2
#m = 0
#dirmixshoot = '/home/jordi/satellite/mix_shooting'
#
#def v_muliSFDM(r, rlam, mu):
#    x2, vdm = np.load('%s/ncor%d/vdm_%d.npy'%(dirmixshoot, ncor-100, ncor-100))    
#    xn = x2/(mu*rlam)
#    vdmn = rlam*vdm*c
#    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate" )
#    return ve(r)
###############################################################################
    
def func(r, rlam, mu, Md, ad):
#    ve2 = v_muliSFDM(r, rlam, mu)**2  
    
#    ve2 = v_multi_SFDM(r, rlam, mu, ncor)**2 
    
    DM = multi_SFDM(state, r, rlam, mu, ncor)
    ve2 = DM.circ_vel()**2
    
    vdisk = RC_exponential(r, G, Md, ad)   
    model =  np.sqrt(ve2 + vdisk**2) 
    return model

Vin = 127.
rin = 1.6
def bul(r, alpha):
    return np.sqrt(alpha*Vin**2*rin/r)
    
def func2(r, rlam, mu, Md, ad, alpha):
#    ve2 = v_muliSFDM(r, rlam, mu)**2  
#    ve2 = v_multi_SFDM(r, rlam, mu, ncor)**2 
    DM = multi_SFDM(state, r, rlam, mu, ncor)
    ve2 = DM.circ_vel()**2
    vdisk = RC_exponential(r, G, Md, ad)   
    vbulge = bul(r,alpha)
    model =  np.sqrt(ve2 + vdisk**2 + vbulge**2)   
    return model

def residual(params, r, data, eps_data):
    mu = params['mu']
    rlam = params['rlam']
    Md = params['Md']
    ad = params['ad']   
    model = func(r, rlam, mu, Md, ad)        
    return (data-model) / eps_data

def residual2(params, r, data, eps_data):
    mu = params['mu']
    rlam = params['rlam']
    Md = params['Md']
    ad = params['ad']  
    alpha = params['alpha']
    model = func2(r, rlam, mu, Md, ad, alpha)        
    return (data-model) / eps_data

###############################################################################
###############################################################################   
def fitting(residual, params, args, i, MO, MCMC = False, print_res = False,
            nsteps = 1000, nwalkers = 100, thin = 1, dirfits = '', ID = '', ID2 = ''):
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
#                          params=out.params,
                          params = params,
                          is_weighted=True)
        save_flatchain(res, dirfits, ncor, ID=ID)
        reports(res, dirfit = dirfits, ID =ID2)      
        flatchain = pd.read_pickle("%sflatchain_nsol%d_%s.pkl"%(dirfits ,ncor, ID))

        if i==5:
            nam = [r'$\mu$',  r'$\sqrt{\lambda}$', r'$M_d$', r'$\alpha$']
            traces= {nam[0]:np.array(flatchain['mu']),
                     nam[1]:np.array(flatchain['rlam']), 
                     nam[2]:np.array(flatchain['Md']),
                     nam[3]:np.array(flatchain['alpha'])}             
        else:
            nam = [r'$\mu$',  r'$\sqrt{\lambda}$', r'$M_d$']
            traces= {nam[0]:np.array(flatchain['mu']),
                     nam[1]:np.array(flatchain['rlam']), 
                     nam[2]:np.array(flatchain['Md'])} 
        
        todas_las_trazas(traces, nam, '%sDM_fit_MCMC_emcee_pars_nsol%d_vbin%d.png'%(dirfits, ncor, i),
                         point_estimate="mode")
     
        unsig = sigmas(res, dirfits, ncor, ID=ID) 
        popt_dw, popt_up = popt_up_dw(res, dirfits, ncor, ID=ID)
        
        walkers_plot(res, nam, MO, namefile='')     
        plot_aceptance(res, namefile='')  
        autocorr_time(res)
        Chain = res.chain[:,:,0].T
        plot_tau_estimates(Chain, namefile =ID2, dirfit = dirfits)
        popt = np.load('%spopt_nsol%d_%s.npy'%(dirfits, ncor, ID))
        print('Gelman-Rubin', gelman_rubin(res.chain))
        
    else:                
        unsig = np.sqrt(np.diag(out.covar)) 
        nstd = 1. # to draw 1-sigma intervals
        popt2 = [popt[0],popt[1],popt[2]]
        popt_up = popt2 + nstd * unsig  ##Fitting parameters at 1 sigma
        popt_dw = popt2 - nstd * unsig  ##Fitting parameters at 1 sigma
        np.save('%spars_dw_nsol%d_%d.npy'%(dirfits, ncor, i), popt_dw)
        np.save('%spars_up_nsol%d_%d.npy'%(dirfits, ncor,i), popt_up)
    return popt, Chi2red, popt_dw, popt_up, unsig 

    

def main(i, ncor):  
    RD = np.array([1.7, 2.2, 3.7, 4.5, 7.9])
        
    ID = 'MCMC_vbin%d'%(i)
    ID2 = 'MCMC_vbin%d_nsol%d'%(i, ncor)
    dirgen = '/home/jordi/LSB_galaxies/URC/Fitssolj%d/'%state
    check_path(dirgen)
    dirfits = '/home/jordi/LSB_galaxies/URC/Fitssolj%d/bin%d/'%(state,i)
    check_path(dirfits)
    _, rad,_, vobs, err = np.loadtxt('%s/bin%d.txt'%(dirdata, i)).T        
#    MO = {'rlam' : [0.0001, 1.], 'mu' : [mukpc[24], mukpc[22]], 
#          'Md' : [1e-4, 1.]}  ## i = 1,2; ncor = 2,3,6, 101 
###                                      i = 1; ncor = 102, 103  
#    MO = {'rlam' : [0.0001, 1.], 'mu' : [mukpc[25], mukpc[23]], 
#          'Md' : [1e-4, 1.]}  ## i = 2; ncor = 102????
#
#    MO = {'rlam' : [0.0001, 1.], 'mu' : [mukpc[24], mukpc[23]], 
#          'Md' : [1e-1, 1e2]}  ## i = 3; ncor = 2,3,6, 101, 102,103     
#    MO = {'rlam' : [0.0001, 1.], 'mu' : [mukpc[24], mukpc[23]], 
#          'Md' : [1e-4, 1e2]}  ## i = 4; ncor = 101  
#    
#    MO = {'rlam' : [0.0001, 1.], 'mu' : [mukpc[24], mukpc[23]], 
#          'Md' : [1e-4, 1e2]}  ## i = 5; ncor = 101         
#
#    MO = {'rlam' : [0.0001, 1.], 'mu' : [10., mukpc[23]], 
#          'Md' : [1e-4, 1e2]}  ## i = 5; ncor = 102, 103???   
    if i == 1:
        MO = {'rlam' : [1e-5, 1.], 'mu' : [mukpc[24], mukpc[22]], 
              'Md' : [1e-4, 1.5e-1]} #### pot jordi bin1
    elif i==2:
        MO = {'rlam' : [1e-5, 1.], 'mu' : [mukpc[24], mukpc[22]], 
              'Md' : [1e-1, 1.]} #### pot jordi bin2    
    elif i ==3:
        MO = {'rlam' : [1e-5, 1.], 'mu' : [mukpc[25], mukpc[22]], 
              'Md' : [1., 10.]} #### pot jordi bin3
    elif i ==4:
        MO = {'rlam' : [1e-5, 1.], 'mu' : [mukpc[25], mukpc[23]], 
              'Md' : [1., 10.]} #### pot jordi bin4        
    else:
        MO = {'rlam' : [1e-5, 1.], 'mu' : [mukpc[26], mukpc[24]], 
              'Md' : [1., 30.]} #### pot jordi bin5        
    params = lm.Parameters()
    params.add('mu',  min = MO['mu'][0], max = MO['mu'][1])
    params.add('rlam',min = MO['rlam'][0],max= MO['rlam'][1])
    params.add('Md', min =  MO['Md'][0], max = MO['Md'][1])
    params.add('ad', vary = False)
    if i == 5 :
        params.add('alpha', vary = True, min = 0.2, max = 1.)
        resid = residual2
    else:
        resid = residual
        
    params['ad'].value = RD[i - 1]
    
    nsteps = 1000
    popt, Chi2red, popt_dw, popt_up, unsig = fitting(resid, params, 
            (rad, vobs, err), i,MO, MCMC = True,
            nsteps = nsteps, nwalkers = 50, thin = 10,
            dirfits = dirfits, ID = ID, ID2 = ID2)
#        
    popt = np.load('%spopt_nsol%d_%s.npy'%(dirfits, ncor, ID)).T
    popt_dw = np.load('%spoptdw_nsol%d_%s.npy'%(dirfits, ncor, ID))
    popt_up = np.load('%spoptup_nsol%d_%s.npy'%(dirfits, ncor, ID))        

    if i == 5 :
        mu, rlam, Md, ad, alpha = popt
        mu_mi, rlam_mi, Md_mi, alpha_mi = popt_dw
        mu_ma, rlam_ma, Md_ma, alpha_ma =  popt_up
    else:
        mu, rlam, Md, ad = popt
        mu_mi, rlam_mi, Md_mi = popt_dw
        mu_ma, rlam_ma, Md_ma = popt_up
        
    v_disco = RC_exponential(rad, G, Md, ad)         
#    v_DM_halo = v_muliSFDM(rad, rlam, mu)
#    v_DM_halo = v_multi_SFDM(rad, rlam, mu, ncor)
    
    DM = multi_SFDM(state, rad, rlam, mu, ncor)
    v_DM_halo = DM.circ_vel()
    
    fit = np.sqrt(v_disco**2 + v_DM_halo**2)        
    labbar = r'Disc $M_d =%.4f \times 10^{7} M_\odot$, $a_d =%.2f$kpc'%(Md*1e3, ad)           
    parss = r'DM $\Phi_{%d%d%d}$: $\mu = %.2f \times 10^{-23} \rm{eV}/c^2$, $\sqrt{\lambda} = %.2f \times 10^{-3}$'%(n,l,m,mu*hc*1e23, rlam*1e3)
    
    if i == 5 :
        Y = [np.sqrt(v_disco**2 + v_DM_halo**2 + bul(rad,alpha)**2), v_disco, v_DM_halo, bul(rad,alpha)]
        L = ['fit', labbar, parss, r'bulge $\alpha = %.2f$'%alpha]
#        y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 + v_muliSFDM(rad, rlam_mi, mu_mi)**2 + bul(rad, alpha_mi)**2)
#        y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 + v_muliSFDM(rad, rlam_ma, mu_ma)**2 + bul(rad, alpha_ma)**2)             
#        y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 + v_multi_SFDM(rad, rlam_mi, mu_mi, ncor)**2 + bul(rad, alpha_mi)**2)
#        y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 + v_multi_SFDM(rad, rlam_ma, mu_ma, ncor)**2 + bul(rad, alpha_ma)**2)

        y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 + multi_SFDM(state, rad, rlam_mi, mu_mi, ncor).circ_vel()**2 + bul(rad, alpha_mi)**2)
        y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 + multi_SFDM(state, rad, rlam_ma, mu_ma, ncor).circ_vel()**2 + bul(rad, alpha_ma)**2)

    else:
        Y = [fit, v_disco, v_DM_halo]
        L = ['fit', labbar, parss]
#        y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 + v_muliSFDM(rad, rlam_mi, mu_mi)**2)
#        y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 + v_muliSFDM(rad, rlam_ma, mu_ma)**2) 
        y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 + multi_SFDM(state, rad, rlam_mi, mu_mi, ncor).circ_vel()**2)
        y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 + multi_SFDM(state, rad, rlam_ma, mu_ma, ncor).circ_vel()**2) 

    pts.plotmultiple([rad, rad, rad, rad], Y, L, r'$r$(kpc)',r'$v$(km/s)',
                     '',
                     '%sSynthetic_RC_%s.png'%(dirfits, ID2), data=True, xd=rad,
                     yd=vobs, err=True, yerr=err, xv=[ad],
                     fill_between = True,fbx = rad, fby1 = y_min,
                     fby2 = y_max)
    
if __name__ == '__main__':
#    plot_data()
    nbin = 5
    SDM = multistate(state)
    for ncor in SDM.nsols:
        print(ncor)
        main(nbin, ncor)