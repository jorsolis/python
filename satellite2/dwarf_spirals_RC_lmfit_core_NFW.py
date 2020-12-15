#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:20:13 2020

Fit MINIMOS CUADRADOS de Dwarf Spirals

DM              core+NFW      pars       Rc, Mc, re, rs
Stellar disk    exponential     pars       MD
HI disk         exponential     pars        none

@author: jordi
"""
import numpy as np
import plots_jordi as pts
from constants_grav import G, c, hc, mukpc, Gentc2
from pymc_tools import (todas_las_trazas, sigmas, popt_up_dw,
                        max_like_sol, autocorr_time, 
                        walkers_plot, plot_aceptance)
import lmfit as lm
import pandas as pd
from def_potenciales import RC_exponential,v2_DM
#from dwarf_spirals import plot_data

dirdata = '/home/jordi/The_universal_rotation_curve_of_dwarf_disk_galaxies'
dat = pd.read_csv('%s/Galaxy_RCs_vopt.txt'%(dirdata), sep = "\t",
                skipinitialspace = True, header = None, skip_blank_lines=False)
dat.columns = ['Name','D','RD','Vopt', 'MK']

dat2 = pd.read_csv('%s/Galaxy_Best_parameters_ref_orig.txt'%(dirdata),
                  sep = "\t", skipinitialspace = True, header = 3, skip_blank_lines=False)
dat2.columns = ['Name','MD','MDKS','MHI', 'MHIK13', 'rc', 'logrho0', 'Mh', 'C']
    
dirfits = '/home/jordi/The_universal_rotation_curve_of_dwarf_disk_galaxies/Fits/gauss'

def func(r, Rc, Mc, re, rs, Md, ad, MHI):
    ve2 = v2_DM(r, G, Rc, Mc, re, rs)
    vdisk = RC_exponential(r, G, Md, ad)   
    vHIdisk = RC_exponential(r, G, MHI, 3.*ad)
    model =  np.sqrt(ve2 + vdisk**2 + vHIdisk**2)    
    return model

def residual(params, r, data, eps_data):
    Rc = params['Rc']
    Mc = params['Mc']
    re = params['re']
    rs = params['rs']
    Md = params['Md']
    ad = params['ad']   
    MHI = params['MHI']
    model = func(r, Rc, Mc, re, rs, Md, ad, MHI)         
    return (data-model) / eps_data
###############################################################################
###############################################################################
    
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
    Rc, Mc, re, rs, Md, ad, MHI = list(out.params.valuesdict().values())
    popt = [Rc, Mc, re, rs, Md, ad, MHI]  
    Chi2red = out.redchi
    if MCMC==True:
        res = lm.minimize(residual, args=args, method='emcee',
                          params=out.params,
                          nan_policy='omit', 
                          burn=int(.3*nsamples),
                          nwalkers = nwalkers,
                          thin=thin,
                          steps=nsamples, is_weighted=True)
        print(lm.fit_report(res))
        nam = [r'$R_c$', r'$M_c$', r'$r_e$', r'$r_s$', r'$M_d$']
        traces= {r'$R_c$':np.array(res.flatchain['Rc']),
                 r'$M_c$':np.array(res.flatchain['Mc']),
                 r'$r_e$':np.array(res.flatchain['re']),
                 r'$r_s$':np.array(res.flatchain['rs']),
                 r'$M_d$':np.array(res.flatchain['Md'])} 
        todas_las_trazas(traces, nam, 
                         '%s/DM_fit_MCMC_emcee_pars_%d.png'%(dirfits,i),
                         point_estimate="mode")         
        Rc, Mc, re, rs, Md, ad, MHI = list(res.params.valuesdict().values())
        popt = [Rc, Mc, re, rs, Md, ad, MHI]
        unsig, _ = sigmas(res, dirfits, 1, ID='NFWcore')   
        popt_dw, popt_up = popt_up_dw(res, dirfits, 1, ID='NFWcore')
        walkers_plot(res, nam, MO, namefile='')     
        plot_aceptance(res, namefile='')  
        autocorr_time(res)

    else:                
        unsig = np.sqrt(np.diag(out.covar)) 
        nstd = 1. # to draw 1-sigma intervals
        popt2 = [popt[0],popt[1],popt[2],popt[3],popt[4]]
        popt_up = popt2 + nstd * unsig  ##Fitting parameters at 1 sigma
        popt_dw = popt2 - nstd * unsig  ##Fitting parameters at 1 sigma

    np.save('%s/pars_dw_%d.npy'%(dirfits,i), popt_dw)
    np.save('%s/pars_up_%d.npy'%(dirfits,i), popt_up)
    np.save('%s/pars_%d.npy'%(dirfits,i), popt)
    return popt, Chi2red, popt_dw, popt_up, unsig 

if __name__ == '__main__':
##    plot_data()    
#    MO = {'Rc' : [0.1, 1., 0.5], 'Mc' : [0.001, 0.050, 4.6e-3], 
#          're' : [0.01, 5.0, 0.1],'rs' : [1., 15., 5.630],
#          'Md' : [1e-6, 1.5e-3, 1e-3], 'ad' : [0.1, 2., 1.5]} 
   
#    RC = []; MC = []; RE=[]; RS=[]; MD = []; AD = []; X2 = [];MU=[]     
#    params = lm.Parameters()
#    params.add('Rc', value = MO['Rc'][2], min = MO['Rc'][0], max = MO['Rc'][1])
#    params.add('Mc', value = MO['Mc'][2], min = MO['Mc'][0], max = MO['Mc'][1])
#    params.add('re', value = MO['re'][2], min = MO['re'][0], max = MO['re'][1])
#    params.add('rs', value = MO['rs'][2], min = MO['rs'][0], max = MO['rs'][1])
#    params.add('Md', value = MO['Md'][2], min = MO['Md'][0], max = MO['Md'][1])    
#    fil = open("%s/cantidades.txt"%dirfits,"w+")
##    for i in range(1, 37,1):
#    for i in range(1, 2,1):
#        rad, vobs, err = np.load('%s/%d.npy'%(dirdata,i))
#        rmin, rmax = np.amin(rad), np.amax(rad)       
#        if np.shape(rad)[0]<7:
#            pass
##        elif i in (17, 21, 22, 24,25,33,36):
##            pass
#        else:
#            print(i)
#            params['ad'].value = dat['RD'][i-1]  
#            params['MHI'].value = dat2['MHI'][i-1]*1e-3    
#            popt, Chi2red, popt_dw, popt_up, unsig = fitting(residual, params,
#                                                      (rad, vobs, err),i,
##                                                      print_res = True,
#                                                      MCMC = True
#                                                      )
#            Rc, Mc, re, rs, Md, ad, MHI = popt 
#            Rc_mi, Mc_mi, re_mi, rs_mi, Md_mi = popt_dw
#            Rc_ma, Mc_ma, re_ma, rs_ma, Md_ma = popt_up         
#            v_disco = RC_exponential(rad, G, Md, ad)
#            v_discoHI = RC_exponential(rad, G, MHI, 3.0*ad)            
#            v_DM_halo = np.sqrt(v2_DM(rad, G, Rc, Mc, re, rs))
#            r = np.linspace(0.,8.)
#            fit = func(r, *popt)          
#            y_min = np.sqrt(RC_exponential(rad, G, Md_mi, ad)**2 +
#                            RC_exponential(rad, G, MHI, 3.0*ad)**2 +
#                            v2_DM(rad, G, Rc_mi, Mc_mi, re_mi, rs_mi))
#            y_max = np.sqrt(RC_exponential(rad, G, Md_ma, ad)**2 +
#                            RC_exponential(rad, G, MHI, 3.0*ad)**2 +
#                            v2_DM(rad, G, Rc_ma, Mc_ma, re_ma, rs_ma))
#
#            labbar = r'Disc $M_d =%.4f \times 10^{7} M_\odot$, $a_d =%.2f$kpc'%(Md*1e3, ad)           
#            parss = r'DM $R_c = %.1f$ pc, $M_c =%.1f \times 10^{7} M_\odot$,$r_e = %.1f$ pc, $r_s = %.2f$ kpc'%(Rc*1e3, Mc*1e3, re*1e3, rs)
#            X2.append(Chi2red)
#            fil.write('%d \t %f \t %f \t %f \t %f \t %f \r\n'%(i, Rc, Mc, re, rs, Md))                              
#            pts.plotmultiple([r, rad, rad, rad, rad],
#                             [fit, v_disco, v_discoHI,v_DM_halo],
#                             [r'fit: $\chi^2_r=$ %f'%Chi2red, labbar,
#                              r'HI disc, $M_{HI}= %.2f \times 10^{7} M_\odot$'%(dat2['MHI'][i-1]),
#                              parss],
#                             r'$r$(kpc)',r'$v$(km/s)', dat['Name'][i-1],
#                             '%s/%d.png'%(dirfits,i), data=True, xd=rad,
#                             yd=vobs, err=True, yerr=err, xv=[ad, Rc, re],
#                             fill_between = True,fbx = rad, fby1 = y_min, 
#                             fby2 = y_max)
#            R99= 2.38167*Rc
#            mu = np.sqrt(9.9* hc**2/(Gentc2*Mc*R99))
#            MD.append(Md*1e3);AD.append(ad); RC.append(Rc); RE.append(re)
#            RS.append(rs); MU.append(mu)
#    fil.close()     
#    
#    MD = np.array(MD); RC = np.array(RC); RE = np.array(RE); RS = np.array(RS) 
#    X2 = np.array(X2); MU = np.array(MU)
#    bins = 20
#    pts.histo(MU,r'eV/$c^2$', bins =bins,
##              normalized = False, rang = (.0001, .0003),
#              nom_archivo ='%s/mu.png'%(dirfits))
#
#    pts.histo(RC,r'$R_c$(kpc)', bins =bins,
#              nom_archivo ='%s/ad.png'%(dirfits)) 
#    pts.histo(RE,r'$r_e$(kpc)', bins =bins,
#              nom_archivo ='%s/ad.png'%(dirfits))     
#    pts.histo(RS,r'$r_s$(kpc)', bins =bins,
#              nom_archivo ='%s/ad.png'%(dirfits)) 
#    
#    pts.histo(MD,r'$M_d (10^{7} M_\odot)$', bins =bins, 
#              nom_archivo ='%s/Md.png'%(dirfits))
#  
#    pts.histo(X2,r'$\chi^2_{red}$', bins =bins,
#              nom_archivo ='%s/chi.png'%(dirfits)) 

###############################################################################
########################        SYNTHETIC RC        ###########################    
############################################################################### 
    dat = pd.read_csv('%s/URC_data.txt'%(dirdata), sep = "\t",
                    skipinitialspace = True, skip_blank_lines=False)                 
    Ropt = 2.5
    rad, vobs, err = np.array(dat.Ri), np.array(dat.Vi), np.array(dat.dVi)   
    MO = {'Rc' : [0.1, 5., 1.9], 'Mc' : [0.01, 0.30, 0.18], 
          're' : [0.1, 5.0, 2.8],'rs' : [1., 30., 9.0],
          'Md' : [1e-4, 1.8e-2, 9.3e-3]}#synthetic  
    params = lm.Parameters()    
    params.add('Rc', min = MO['Rc'][0], max = MO['Rc'][1])
    params.add('Mc', min = MO['Mc'][0], max = MO['Mc'][1])
    params.add('re', min = MO['re'][0], max = MO['re'][1])
    params.add('rs', min = MO['rs'][0], max = MO['rs'][1])
    params.add('Md', min = MO['Md'][0], max = MO['Md'][1])
    params.add('ad', vary = False)
    params.add('MHI', vary = False)
    params['ad'].value = Ropt/3.2 
    params['MHI'].value = 1.7e-2
    popt, Chi2red, popt_dw, popt_up, _ = fitting(residual, params,
                                                 (rad, vobs, err), 1000,
                                                 print_res = False,
                                                 nsamples = 1000,
                                                 nwalkers = 100,
                                                 MCMC = True,
                                                 thin = 100
                                                 )
    Rc, Mc, re, rs, Md, ad, MHI = popt            
    v_disco = RC_exponential(rad, G, Md, ad)
    v_discoHI = RC_exponential(rad, G, MHI, 3.0*ad)            
    v_DM_halo = np.sqrt(v2_DM(rad, G, Rc, Mc, re, rs))
    labbar = r'Disc $M_d =%.4f \times 10^{7} M_\odot$, $a_d =%.2f$kpc'%(Md*1e3, ad)           
    parss = r'DM $R_c = %.1f$ kpc, $M_c =%.1f \times 10^{8} M_\odot$,$r_e = %.1f$ kpc, $r_s = %.2f$ kpc'%(Rc, Mc*1e2, re, rs)
    pts.jordi_style()
    pts.plotmultiple([rad, rad, rad, rad],
                     [func(rad, *popt), v_disco, v_discoHI, v_DM_halo],
                     [r'fit: $\chi^2_r=$ %f'%Chi2red, labbar,
                      r'HI disc, $M_{HI}= %.2f \times 10^{8} M_\odot$'%(MHI*1e2),
                      parss], r'$r$(kpc)',r'$v$(km/s)', 'Synthetic RC',
                     '%s/Synthetic_RC.png'%(dirfits), data=True, xd=rad,
                     yd=vobs, err=True, yerr=err, xv=[ad, Rc, re],
                     fill_between = True,fbx = rad,
                     fby1 = func(rad, *popt_dw, ad, MHI),
                     fby2 = func(rad, *popt_up, ad, MHI))
    R0 = Rc
    print('rho_halo(0) = ',Mc/(np.sqrt(np.pi)**3*Rc**3)*10., 'M_sun/pc^3')  
    print('log10( rho_halo(0) )= ',np.log10(Mc/(np.sqrt(np.pi)**3*Rc**3)*10.0**10), 'M_sun/kpc^3') 
    print('rho_halo(0)r0(0) = ',Mc/(np.sqrt(np.pi)**3*Rc**3)*10.*R0*1e3, 'M_sun/pc^2') 
    print('log10( Md )= ', np.log10(Md*10.0**10), 'M_sun') 
