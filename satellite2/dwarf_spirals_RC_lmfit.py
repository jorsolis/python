#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:20:13 2020

Fit de Dwarf Spirals

DM              multi-SFDM      pars       mu, rlam
Stellar disk    exponential     pars       MD
HI disk         exponential     pars        none

@author: jordi
"""
import numpy as np
import plots_jordi as pts
from scipy.interpolate import interp1d
from constants_grav import G, c, hc, mukpc
from pymc_tools import (todas_las_trazas, sigmas, popt_up_dw,gelman_rubin,
                        autocorr_time, save_flatchain, reports,
                        walkers_plot, plot_aceptance, plot_tau_estimates) 
import lmfit as lm
import pandas as pd
from def_potenciales import RC_exponential, multistate
from Os_tools import check_path
#from dwarf_spirals import plot_data

dirdata = '/home/jordi/The_universal_rotation_curve_of_dwarf_disk_galaxies'
dat = pd.read_csv('%s/Galaxy_RCs_vopt.txt'%(dirdata), sep = "\t",
                skipinitialspace = True, header = None, skip_blank_lines=False)
dat.columns = ['Name','D','RD','Vopt', 'MK']

dat2 = pd.read_csv('%s/Galaxy_Best_parameters_ref_orig.txt'%(dirdata),
                  sep = "\t", skipinitialspace = True, header = 3, skip_blank_lines=False)
dat2.columns = ['Name','MD','MDKS','MHI', 'MHIK13', 'rc', 'logrho0', 'Mh', 'C']

###############################################################################
###################       Multi 100, exc       ################################
###############################################################################
n = 2; l = 1; m = 1
state = 211
ID = 'multi_j'
from def_potenciales import multi_SFDM
################################################################################

    
#dirfits = '/home/jordi/The_universal_rotation_curve_of_dwarf_disk_galaxies/Fits/'
#dirfits = '/home/jordi/The_universal_rotation_curve_of_dwarf_disk_galaxies/Fits_solj/'

dirfits = '/home/jordi/The_universal_rotation_curve_of_dwarf_disk_galaxies/Fitssolj%d/'%state

check_path(dirfits)

################################################################################
####################            POT PACO 2       ################################
################################################################################
#soldic = {1:{'n':2, 'l':1, 'm':0}, 2:{'n':2, 'l':1, 'm':0}, 6:{'n':2, 'l':1, 'm':0},
#          3:{'n':2, 'l':1, 'm':1}, 4:{'n':2, 'l':1, 'm':1}, 5:{'n':2, 'l':1, 'm':1}}
#ncor = 6
#n = soldic[ncor]['n']
#l = soldic[ncor]['l']
#m = soldic[ncor]['m']
#ID = 'multi'
#
#from def_potenciales import v_multi_SFDM
###############################################################################
###################            POT PACO        ################################
###############################################################################
#sat = '/home/jordi/satellite'
#paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
#dirmixshoot = '/home/jordi/satellite/mix_shooting'
#soldic = {1:{'n':2, 'l':1, 'm':0}, 2:{'n':2, 'l':1, 'm':0}, 6:{'n':2, 'l':1, 'm':0},
#          3:{'n':2, 'l':1, 'm':1}, 4:{'n':2, 'l':1, 'm':1}, 5:{'n':2, 'l':1, 'm':1}}
#ncor = 6
#n = soldic[ncor]['n']
#l = soldic[ncor]['l']
#m = soldic[ncor]['m']
#ID = 'multi'
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
###############################################################################
#ncor = 101
#n = 3
#l = 2
#m = 0
#dirmixshoot = '/home/jordi/satellite/mix_shooting'
#ID = 'multi'
#def v_muliSFDM(r, rlam, mu):
#    x2, vdm = np.load('%s/ncor%d/vdm_%d.npy'%(dirmixshoot, ncor-100, ncor-100))    
#    xn = x2/(mu*rlam)
#    vdmn = rlam*vdm*c
#    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate" )
#    return ve(r)
###############################################################################
 
def func(r, rlam, mu, Md, ad, MHI):
#    ve2 = v_muliSFDM(r, rlam, mu)**2  
    
#    ve2 = v_multi_SFDM(r, rlam, mu, ncor)**2  
    
    DM = multi_SFDM(state, r, rlam, mu, ncor)
    ve2 = DM.circ_vel()**2
    
    vdisk = RC_exponential(r, G, Md, ad)   
    vHIdisk = RC_exponential(r, G, MHI, 3.*ad)
    model =  np.sqrt(ve2 + vdisk**2 + vHIdisk**2)   
    return model

def residual(params, r, data, eps_data):
    mu = params['mu']
    rlam = params['rlam']
    Md = params['Md']
    ad = params['ad']   
    MHI = params['MHI']
    model = func(r, rlam, mu, Md, ad, MHI)        
    return (data-model) / eps_data
###############################################################################
###############################################################################   
def fitting(residual, params, args, i, MO, MCMC = False, print_res = False,
            nsteps = 1000, nwalkers = 100, thin = 1):
    out = lm.minimize(residual, params, args=args,
                      method='nelder',
#                      method = 'powell'
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
                          nwalkers = nwalkers,
                          steps=nsteps, 
                          thin=thin,
                          params=out.params, 
                          is_weighted=True)
        save_flatchain(res, dirfits , ncor, ID=ID)
        reports(res, dirfit = dirfits, ID = '%s_nsol%d'%(ID, ncor))      
        flatchain = pd.read_pickle("%sflatchain_nsol%d_%s.pkl"%(dirfits ,ncor, ID))
        
        nam = [r'$\mu$',  r'$\sqrt{\lambda}$', r'$M_d$']
        traces= {r'$\mu$':np.array(flatchain['mu']),
                 r'$\sqrt{\lambda}$':np.array(flatchain['rlam']),
                   r'$M_d$':np.array(flatchain['Md'])}        
        todas_las_trazas(traces, nam, 
                         '%sDM_fit_MCMC_emcee_pars_%d_%s.png'%(dirfits,i, ID),
                         point_estimate="mode")                

        unsig, _ = sigmas(res, dirfits, ncor, ID=ID)   
        popt_up, popt_dw = popt_up_dw(res, dirfits, ncor, ID=ID)
        walkers_plot(res, nam, MO, namefile='')     
        plot_aceptance(res, namefile='')  
        autocorr_time(res)
        Chain = res.chain[:,:,0].T
        plot_tau_estimates(Chain) 
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
        np.save('%spars_nsol%d_%d.npy'%(dirfits, ncor,i), popt)
    return popt, Chi2red, popt_dw, popt_up, unsig 
   
def main(ncor):
    dat = pd.read_csv('%s/URC_data.txt'%(dirdata), sep = "\t",
                      skipinitialspace = True, skip_blank_lines = False)   
    rad, vobs, err = np.array(dat.Ri), np.array(dat.Vi), np.array(dat.dVi)
#    
#    MO = {'rlam' : [0.0001, 1.], 'mu' : [mukpc[23], mukpc[21]], 
#          'Md' : [1e-3, 1.5e-1]} #### pot paco 2,6
#    MO = {'rlam' : [0.0001, 1.], 'mu' : [mukpc[24], mukpc[21]], 
#          'Md' : [1e-3, 1.5e-1]} #### pot (320)
    MO = {'rlam' : [1e-5, 1.], 'mu' : [mukpc[24], mukpc[22]], 
          'Md' : [1e-3, 1.5e-1]} #### pot jordi
     
    params = lm.Parameters()
    params.add('mu',  min = MO['mu'][0], max = MO['mu'][1])
    params.add('rlam',min = MO['rlam'][0],max= MO['rlam'][1])
    params.add('Md', min =  MO['Md'][0], max = MO['Md'][1])
    params.add('ad', vary = False)
    params.add('MHI', vary = False)    
              
    Ropt = 2.5
    params['ad'].value = Ropt/3.2 
    params['MHI'].value = 1.7e-2

    popt, Chi2red, popt_dw, popt_up, unsig = fitting(residual, params, 
            (rad, vobs, err), ncor, MO, MCMC = True,
            nsteps = 1000, nwalkers = 50, thin = 10)
    
    mu, rlam, Md, ad, MHI = np.load('%spopt_nsol%d_%s.npy'%(dirfits, ncor, ID)) 
    mu_mi, rlam_mi, Md_mi = np.load('%spoptdw_nsol%d_%s.npy'%(dirfits, ncor, ID))
    mu_ma, rlam_ma, Md_ma = np.load('%spoptup_nsol%d_%s.npy'%(dirfits, ncor, ID))        
    v_disco = RC_exponential(rad, G, Md, ad)
    v_discoHI = RC_exponential(rad, G, MHI, 3.0*ad)            
#    v_DM_halo = v_muliSFDM(rad, rlam, mu)
#    v_DM_halo = v_multi_SFDM(rad, rlam, mu, ncor)
    
    DM = multi_SFDM(state, rad, rlam, mu, ncor)
    v_DM_halo = DM.circ_vel()
    
    fit = np.sqrt(v_disco**2 + v_discoHI**2 + v_DM_halo**2)        
#    y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 + RC_exponential(rad, G, MHI, 3.0*ad)**2 + v_muliSFDM(rad, rlam_mi, mu_mi)**2)
#    y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 + RC_exponential(rad, G, MHI, 3.0*ad)**2 + v_muliSFDM(rad, rlam_ma, mu_ma)**2) 
    y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 + RC_exponential(rad, G, MHI, 3.0*ad)**2 + multi_SFDM(state, rad, rlam_mi, mu_mi, ncor).circ_vel()**2)
    y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 + RC_exponential(rad, G, MHI, 3.0*ad)**2 +  multi_SFDM(state, rad, rlam_ma, mu_ma, ncor).circ_vel()**2)
    labbar = r'Disc $M_d =%.4f \times 10^{7} M_\odot$, $a_d =%.2f$kpc'%(Md*1e3, ad)           
    labHI = r'HI disc, $M_{HI}= %.2f \times 10^{8} M_\odot$'%(MHI*1e2)
    labDM = r'$\mu = %.1f \times 10^{-24}$eV/$c^2$, $\sqrt{\lambda}$ = %.1f $\times 10^{-3}$'%(mu*hc*1e24, rlam*1e3)
    pts.jordi_style()
    pts.plotmultiple([rad, rad, rad, rad], [fit, v_disco, v_discoHI, v_DM_halo],
                     ['fit', labbar, labHI, labDM], r'$r$(kpc)',r'$v$(km/s)', '',
                     '%s/Synthetic_RC_nsol%d_%s.png'%(dirfits, ncor,ID),
                     data=True, xd=rad, yd=vobs, err=True, yerr=err, xv=[ad],
                     fill_between = True,fbx = rad, fby1 = y_min, fby2 = y_max)
    

    flatchain = pd.read_pickle('%sflatchain_nsol%d_%s.pkl'%(dirfits, ncor, ID))     

    fil= open("%s/cantidades_emcee_nsol%d_%s.txt"%(dirfits, ncor, ID),"w+")   
    fil.write('name \t mean \t 1sigma \t 2sigma  \r\n')
    proms = np.load('%spopt_nsol%d_%s.npy'%(dirfits, ncor, ID))
    names = ('mu', 'rlam', 'Md')
    for ii in range(1,4):
        quantiles = np.percentile(flatchain[names[ii - 1 ]], [2.28, 15.9, 50, 84.2, 97.7])      
        sigma1 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2 = 0.5 * (quantiles[4] - quantiles[0])
        prom = proms[ii - 1]
        print(ii, prom, sigma1, sigma2)
        fil.write('%s \t %.4f \t %.5f \t %.5f  \r\n'%(names[ii - 1], prom, sigma1, sigma2))

        if names[ii - 1]== 'rlam':
            print(ii, '(',prom*1e5,sigma1*1e5, sigma2*1e5, ') 10^-3')    
            fil.write('%s \t %.4f \t %.5f \t %.5f  \t 1e-5 \r\n'%(names[ii - 1], prom*1e5, sigma1*1e5, sigma2*1e5))
        if names[ii - 1] == 'mu':
            print(ii, '(',prom*hc*1e25,sigma1*hc*1e25, sigma2*hc*1e25, ') 10^-25 eV/c^2')
            fil.write('%s \t %.4f \t %.5f \t %.5f  \t 1e-25 \r\n'%(names[ii - 1], prom*hc*1e25, sigma1*hc*1e25, sigma2*hc*1e25))

    fil.close()
    
if __name__ == '__main__':
    SDM = multistate(state)
    for ncor in SDM.nsols:     
        main(ncor)