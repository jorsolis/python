#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

Fit MCMC de la via lactea 

DM      Core + NFW              pars      Rc, Mc, re, rs
Disco   Razor exponential       pars      Md, ad
Bulge   Exponential             pars      Mb, bb
BH      Newtonian               pars      MBH

@author: jordi
"""
import plots_jordi as pts
import numpy as np
from def_potenciales import(vHernquist, vnagai, Miyamoto_Nagai_3,
                            M_Miyamoto2,M_hernquist, exp_bulge,M_exp_bulge,
                            M_exp_disk, M_two_bulge, vBH, RC_exponential,
                            v2_DM, M_CNFW,dens_DM, dens_gaus)
import lmfit as lm
import matplotlib.pyplot as plt
from pymc_tools import todas_las_trazas
from constants_grav import G, Gentc2, mu22, hc
from MW_fit_pymc import proba_eq
from subprocess import call
###############################################################################
np.random.seed(12345)
galaxy = 'Milky Way'
nsamples = 1e4

if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/Gaussian/'%dirdata
    datalogRCMW= np.loadtxt("%s/LogRC_data.dat.txt"%dirdata) 
    rad,_, v, v_error = datalogRCMW.T
    rad,_, v, v_error = rad[:57],_, v[:57], v_error[:57]
#    
def M_t(r, Rc, Mc, re, rs, Md, ad, Mb, bb, MBH):
    Mhalo = M_CNFW(r, Rc, Mc, re, rs)
    Mdisk =  M_exp_disk(r, Md, ad)
    Mbulge = M_exp_bulge(r, Mb, bb)
    return Mhalo + Mdisk + Mbulge + MBH

MO = {'Rc' : [0.001, 0.100, 6.0e-3], 'Mc' : [0.001, 0.050], 
      're' : [0.01, 0.05, 8.5],'rs' : [1., 15., 4.], 
      'Md' : [1e-2, 30., 5.0], 'ad' : [0.1, 10., 5.],'bd' : [.8, 3., 1.15],
      'Mb': [.1, 1.5, 0.3], 'bb' : [0.05, 0.5, 1.6],
      'BH' : [1e-4, 1e-3]}

def residual(params, r, data, eps_data):
    Rc = params['Rc']
    Mc = params['Mc']
    re = params['re']
    rs = params['rs']
    Md = params['Md']
    ad = params['ad']
    Mb = params['Mb']
    bb = params['bb']
    MBH = params['MBH']
    ve2 = v2_DM(r, G, Rc, Mc, re, rs)
    vdisk = RC_exponential(r, G, Md, ad)
    vbul = exp_bulge(r, G, Mb, bb)
    model =  np.sqrt(ve2 + vdisk**2 + vbul**2+ vBH(r, G, MBH)**2) 
    return (data-model) / eps_data
def modelo(params, r):
    Rc = params['Rc']
    Mc = params['Mc']
    re = params['re']
    rs = params['rs']
    Md = params['Md']
    ad = params['ad']
    Mb = params['Mb']
    bb = params['bb']
    MBH = params['MBH']
    ve2 = v2_DM(r, G, Rc, Mc, re, rs)
    vdisk = RC_exponential(r, G, Md, ad)
    vbul = exp_bulge(r, G, Mb, bb)
    model =  np.sqrt(ve2 + vdisk**2 + vbul**2+ vBH(r, G, MBH)**2) 
    return model

def v_tot(r, Rc, Mc, re, rs, Md, ad, Mb, bb, MBH):
    ve2 = v2_DM(r, G, Rc, Mc, re, rs)
    vdisk = RC_exponential(r, G, Md, ad)
    vbul = exp_bulge(r, G, Mb, bb)
    model =  np.sqrt(ve2 + vdisk**2 + vbul**2+ vBH(r, G, MBH)**2) 
    return model
def func(r, Rc, Mc, re, rs, Md, ad, Mb, bb, MBH):
    ve2 = v2_DM(r, G, Rc, Mc, re, rs)
    vdisk = RC_exponential(r, G, Md, ad)
    vbul = exp_bulge(r, G, Mb, bb)
    model =  np.sqrt(ve2 + vdisk**2 + vbul**2+ vBH(r, G, MBH)**2) 
    return model    
    
    
params = lm.Parameters()
params.add('Rc', value = 76e-4, min = MO['Rc'][0], max = MO['Rc'][1])
params.add('Mc', value = 46e-4, min = MO['Mc'][0], max = MO['Mc'][1])
params.add('re', value = 0.020, min = MO['re'][0], max = MO['re'][1])
params.add('rs', value = 5.630, min = MO['rs'][0], max = MO['rs'][1])
params.add('Md', value = 5.15, min = MO['Md'][0], max = MO['Md'][1])
params.add('ad', value = 4.86, min = MO['ad'][0], max = MO['ad'][1])
params.add('Mb', value = 0.86, min = MO['Mb'][0], max = MO['Mb'][1])
params.add('bb', value = 0.12, min = MO['bb'][0], max = MO['bb'][1])
params.add('MBH', value = 0.00037, min = MO['BH'][0], max = MO['BH'][1])

##############################################################################
###################           FIT CHI^2            ###########################
##############################################################################
#out = lm.minimize(residual, params, args=(rad, v, v_error))
##print(lm.printfuncs.report_fit(out.params, min_correl=0.5))
##print(out.params.pretty_print())
#Rc, Mc, re, rs, Md, ad, Mb, bb, MBH = list(out.params.valuesdict().values())
#rmin, rmax = 1e-3, np.amax(rad)
#r = np.linspace(rmin, rmax, 100000)
#vdisk = RC_exponential(r, G, Md, ad)
#vbulge = exp_bulge(r, G, Mb, bb)
#pts.plotmultiple([r,r, r,r,r,r], 
#                 [modelo(out.params,r), np.sqrt(vdisk**2 + vbulge**2),
#                 vdisk, vbulge, np.sqrt(v2_DM(r, G, Rc, Mc, re, rs)),
#                 vBH(r, G, MBH)],
#                 [r'Disk+Bulge+SMBH+SFDM', r'Disk+Bulge',
#                  r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
#                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3), 
#                  r'SFDM  $R_c = %.3f$ pc, $M_c = %.1f \times 10^{7} M_\odot$, $r_e=%.2f$pc, $r_{sN}=%.1f$kpc'%(Rc*1e3, Mc*1e3, re*1e3, rs),
#                  r'SMBH, $M_{BH}=%.4f\times 10^{6} M_\odot$'%(MBH*1.0e4),'Observed'],                
#                 r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
#                 '%sDM_fit_chi2.png'%(dirfitsG), xlim = (rmin,rmax),
#                 ylim = (20,500), data = True, xd = rad, yd = v, err = True, 
#                 yerr = v_error, logx = True,logy = True)
###############################################################################
#############################        MCMC        ##############################3
###############################################################################
#out.params.add(r'__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
#res = lm.minimize(residual, args=(rad, v, v_error), method='emcee',
#                  nan_policy='omit', burn=int(.3*nsamples), steps=nsamples, 
##                  thin=20, 
#                  nwalkers = 300,
#                  params=out.params,is_weighted=False )
#print(lm.report_fit(res.params))
#print(out.params.pretty_print())
#
##mini = lm.Minimizer(residual, out.params, args=(rad, v, v_error))
##ci = lm.conf_interval(mini, res)
##print(lm.printfuncs.report_ci(ci))
##ci, trace = lm.conf_interval(mini, res, sigmas=[1, 2],
##                                trace=True, verbose=False)
##cx, cy, grid = lm.conf_interval2d(mini, res, 'Rc', 'Mc', 30, 30)
##plt.contourf(cx, cy, grid, np.linspace(0, 1, 11))
##plt.xlabel('Rc')
##plt.colorbar()
##plt.ylabel('Mc')
##plt.show()
## lets work out a 1 and 2-sigma error estimate for 'Rc'
#print('parameter   1 sigma spread  2 sigma spread')
#
#fil= open("%s/cantidades_emcee.txt"%dirfitsG,"w+")
#units = {'Rc':'pc', 'Mc':'10^7 M_\odot', 're':'pc','rs':'kpc',
#          'Md':'10^10 M_\odot', 'ad':'kpc', 'Mb':'10^7 M_\odot', 'bb':'kpc', 
#          'MBH':'10^6 M_\odot'}
#fil.write('name \t units \t mean \t 1sigma \t 2sigma  \r\n')
#unsigma=[]
#dossigma=[]
#for i in res.var_names[:-1]:
#    quantiles = np.percentile(res.flatchain[i], [2.28, 15.9, 50, 84.2, 97.7])      
#    sigma1 = 0.5 * (quantiles[3] - quantiles[1])
#    sigma2 = 0.5 * (quantiles[4] - quantiles[0])
#    prom = res.params[i].value
#    if i == 'Rc':
#        sigma1 = sigma1*1000
#        sigma2 = sigma2*1000
#        prom = prom*1000
#    elif i == 'Mc':
#        sigma1 = sigma1*1000
#        sigma2 = sigma2*1000
#        prom = prom*1000
#    elif i == 're':
#        sigma1 = sigma1*1000
#        sigma2 = sigma2*1000
#        prom = prom*1000
#    elif i == 'MBH':
#        sigma1 = sigma1*10000
#        sigma2 = sigma2*10000
#        prom = prom*10000
#    elif i == 'Mb':
#        sigma1 = sigma1*1000
#        sigma2 = sigma2*1000
#        prom = prom*1000
#    unsigma.append(sigma1)
#    dossigma.append(sigma2)
#    fil.write('%s \t %s \t %.4f \t %.4f \t %.4f  \r\n'%(i, units[i], prom, sigma1, sigma2))
#fil.close()
#unsigma = np.array(unsigma); np.save('%sunsigma.npy'%(dirfitsG), unsigma)
#dossigma = np.array(dossigma); np.save('%sdossigma.npy'%(dirfitsG), dossigma)
#print(res.var_names)
#
#popt_dw = []; popt_up = []
#
#for i in res.var_names[:-1]:
#    per = np.percentile(res.flatchain[i], [2.5])
#    per2 = np.percentile(res.flatchain[i], [97.5])
#    popt_dw.append(per[0])
#    popt_up.append(per2[0])
#popt_dw = np.array(popt_dw); popt_up = np.array(popt_up)
#np.save('%straces.npy'%(dirfitsG), np.array([np.array(res.flatchain['Rc']),
#        np.array(res.flatchain['Mc']), np.array(res.flatchain['re']), 
#                 np.array(res.flatchain['rs']), np.array(res.flatchain['Md']),
#                          np.array(res.flatchain['ad']), np.array(res.flatchain['Mb']),
#                                   np.array(res.flatchain['bb']), np.array(res.flatchain['MBH'])]))
#traces= {r'$R_c$':np.array(res.flatchain['Rc']),
#         r'$M_c$':np.array(res.flatchain['Mc']),
#           r'$r_e$':np.array(res.flatchain['re']),
#           r'$r_{sN}$':np.array(res.flatchain['rs']),
#           r'$M_d$':np.array(res.flatchain['Md']),
#           r'$a_d$':np.array(res.flatchain['ad']),
#           r'$M_b$':np.array(res.flatchain['Mb']),
#           r'$a_b$':np.array(res.flatchain['bb']),           
#           r'$M_{BH}$':np.array(res.flatchain['MBH'])}
#traz = np.load('%straces.npy'%(dirfitsG))
#Rc, Mc, re, rs, Md, ad, Mb, bb, MBH, _ = list(res.params.valuesdict().values())
#
popt = np.load('%spopt.npy'%(dirfitsG))
#np.savetxt('%sparams.txt'%(dirfitsG), popt)
popt_dw = np.load('%spopt_dw.npy'%(dirfitsG))
popt_up = np.load('%spopt_up.npy'%(dirfitsG))
Rc, Mc, re, rs, Md, ad, Mb, bb, MBH = popt
#
rmin, rmax = 1e-3, np.amax(rad)
r = np.linspace(rmin, rmax, 100000)
vdisk = RC_exponential(r, G, Md, ad)
vbulge = exp_bulge(r, G, Mb, bb)
y_min= func(r, *popt_dw)
y_max= func(r, *popt_up)
pts.plotmultiple([r,r, r,r,r,r], 
                 [v_tot(r,*popt),
#                  modelo(res.params,r),
                  np.sqrt(vdisk**2 + vbulge**2),
                 vdisk, vbulge, np.sqrt(v2_DM(r, G, Rc, Mc, re, rs)),
                 vBH(r, G, MBH)],
                 [r'Disc+Bulge+SMBH+SFDM', r'Disc+Bulge',
                  r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3), 
                  r'SFDM  $R_c = %.1f$ pc, $M_c = %.1f \times 10^{7} M_\odot$, $r_e=%.1f$pc, $r_{sN}=%.1f$kpc'%(Rc*1e3, Mc*1e3, re*1e3, rs),
                  r'SMBH, $M_{BH}=%.2f\times 10^{6} M_\odot$'%(MBH*1.0e4)],              
                 r'$r$[kpc]', r'$v$[km/s]', '',
                 '%sDM_fit_emcee.png'%(dirfitsG),
                 xlim = (rmin,rmax), #ylim = (20,500),
                 data = True, xd = rad, yd = v, err = True, yerr = v_error,
                 fill_between = True, logx = True,logy = True,
                 fbx = r, fby1 = y_min, fby2 = y_max, xv=[Rc, re, rs])
#pts.plotmultiple([r,r, r,r,r,r], 
#                 [v_tot(r,*popt),
##                  modelo(res.params,r), 
#                  np.sqrt(vdisk**2 + vbulge**2),
#                 vdisk, vbulge, np.sqrt(v2_DM(r, G, Rc, Mc, re, rs)),
#                 vBH(r, G, MBH)],
#                 [r'Disc+Bulge+SMBH+SFDM', r'Disc+Bulge',
#                  r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
#                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3), 
#                  r'DM  $R_c = %.3f$ pc, $M_c = %.1f \times 10^{7} M_\odot$, $r_e=%.2f$pc, $r_{sN}=%.1f$kpc'%(Rc*1e3, Mc*1e3, re*1e3, rs),
#                  r'SMBH, $M_{BH}=%.4f\times 10^{6} M_\odot$'%(MBH*1.0e4),'Observed'],                
#                 r'$r$[kpc]', r'$v$[km/s]', '',
#                 '%sDM_fit_emcee2.png'%(dirfitsG),
#                 xlim = (rmin,rmax), ylim = (20,500),
#                 data = True, xd = rad, yd = v, err = True, yerr = v_error,
#                 fill_between = True,
#                 fbx = r, fby1 = y_min, fby2 = y_max, xv=[Rc, re, rs])
#
#name =   [r'$R_c$', r'$M_c$', r'$r_e$', r'$r_{sN}$', r'$M_d$', r'$a_d$',
#          r'$M_b$',r'$a_b$', r'$M_{BH}$']
#traces= {r'$R_c$':traz[0],r'$M_c$':traz[1],r'$r_e$':traz[2],r'$r_{sN}$':traz[3],
#         r'$M_d$':traz[4],r'$a_d$':traz[5], r'$M_b$':traz[6], r'$a_b$':traz[7],           
#         r'$M_{BH}$':traz[8]}
#todas_las_trazas(traces, name, '%sDM_fit_MCMC_emcee_pars.png'%(dirfitsG),
#                 point_estimate="mean")
##
#popt = [Rc, Mc, re, rs, Md, ad, Mb, bb, MBH]

#popt = np.load('%spopt.npy'%(dirfitsG))
#unsigma = np.load('%sunsigma.npy'%(dirfitsG))
#Rc, Mc, re, rs, Md, ad, Mb, bb, MBH = popt
#MctRc = M_t(Rc, *popt)
#R99 = 2.38167*Rc
#MctR99= M_t(R99, *popt)
##proba_eq(Rc, Mc, MctRc, MctR99, dirfitsG= dirfitsG, 
##         nomb = 'cantidades_emcee2')
#R200 = 264.469 #kpc
#Mc7 = Mc*1e3
#Mht12 = M_t(R200, Rc, Mc, re, rs, Md, ad, Mb, bb, MBH)*1e-2
#
#mu = np.sqrt(5.266/(Mc7*Mht12**(2./3.))) # ecuacion 60
#print(mu, 'ecuacion 60')
#Mct7 = M_t(R99, Rc, Mc, re, rs, Md, ad, Mb, bb, MBH)*1e3 
#mu = 1.4e2*Mht12**(1./3.)/(Mct7)# ecuacion 42
#print(mu, 'ecuacion 42')
#mu = np.sqrt(np.sqrt(1.63e5/(Mc7*Mct7**2)))#execuacion 51
#print(mu, 'ecuacion 51')


#def Mc7func(m22, Mht12):
#    return 5.266/(m22**2 * Mht12**(2./3.))
#
#Mht = np.linspace(1e-5, 1e3, 1000)
#m1 =0.55
#m2 = 1.0
#m3 = 23.31

#pts.plotmultiple([Mht,Mht,Mht,Mht], 
#                 [Mc7func(m1, Mht), Mc7func(m2, Mht), Mc7func(m3, Mht)],
#                 [r'$m_{22} = %.2f $'%m1, r'$m_{22} = %.2f $'%m2,
#                  r'$m_{22} = %.2f $'%m3], 
#                 r'$M_{h,12}^t$', r'$M_{c,7}$', '',
#                 '', ylim=(1e-4,1e6),
#                 xlim=(1e-4, 1e2), logy= True, save = True, loc_leg='best',
#                 angular=False, xangular=False, logx = True, show = True,
#                 data=True, xd=[0.1782/m1, 0.1782/m2, 0.1782/m3],
#                 yd=[Mc7func(m1,0.1782/m1), Mc7func(m2,0.1782/m2), 
#                     Mc7func(m3,0.1782/m3)],
#                 markersize = 20,
#                 xv=[])

#def ec50(m_22,M_ext): 
#    a = 1.636*10**5./(m_22**4.) 
#    b = M_ext 
#    coef = (5.1962*(27.*a**2.+4.*a*b**3.)**(1./2)+27.*a+2.*b**3.)**(1./3) 
#    return (1./3)*(0.7937*coef+1.2599*b**2./coef-2.*b)
#
#Mext = np.linspace(1e-4, 1e5, 100000)
#pts.plotmultiple([Mext,Mext,Mext], 
#                 [ec50(m1, Mext), ec50(m2, Mext), ec50(m3, Mext)],
#                 [r'$m_{22} = %.2f $'%m1, r'$m_{22} = %.2f $'%m2,
#                  r'$m_{22} = %.2f $'%m3], 
#                 r'$M_{ext,7}$', r'$M_{c,7}$', '',
#                 '', ylim=(1e-9,1e3),
#                 xlim=(1e-4, 1e5), logy= True, save = True, loc_leg='best',
#                 logx = True, show = True)
##
#def f4(M_core): 
#    return (M_core/(1.298e-4))**(1./2)
#Mext = np.linspace(1e-4, 1e4, 100000)
#pts.plotmultiple([Mext,Mext,Mext], 
#                 [f4(ec50(m1, Mext)), f4(ec50(m2, Mext)), f4(ec50(m3, Mext))],
#                 [r'$m_{22} = %.2f $'%m1, r'$m_{22} = %.2f $'%m2,
#                  r'$m_{22} = %.2f $'%m3], 
#                 r'$M_{ext,7}$', r'$R_{c}$[pc]', '',
#                 '', ylim=(1e-2,2e3),
#                 xlim=(1e-4, 1e4), logy= True, save = True, loc_leg='best',
#                 logx = True, show = True)

#pts.plotmultiple([Mext,Mext,Mext], 
#                 [ec50(m1, Mext)/f4(ec50(m1, Mext))**3,
#                  ec50(m2, Mext)/f4(ec50(m2, Mext))**3,
#                  ec50(m3, Mext)/f4(ec50(m3, Mext))**3],
#                 [r'$m_{22} = %.2f $'%m1, r'$m_{22} = %.2f $'%m2,
#                  r'$m_{22} = %.2f $'%m3], 
#                 r'$M_{ext,7}$', r'$\bar{\rho_{c}}\left[\rm{pc}^{-3}\right]$', '',
#                 '', ylim=(1e-7,1e-2),
#                 xlim=(1e-4, 1e5), logy= True, save = True, loc_leg='best',
#                 logx = True, show = True)

#unsigma = np.array([unsigma[0]/1e3, unsigma[1]/1e3, unsigma[2]/1e3, unsigma[3],
#           unsigma[4], unsigma[5], unsigma[6]/1e3, unsigma[7],unsigma[8]/1e4])
#unsigma = unsigma + popt
#Rc2 = unsigma[0]
#Mc2 = unsigma[1]
#MctRc2 = M_t(Rc2, *unsigma)
#R992 = 2.38167*Rc2
#MctR992= M_t(R992, *unsigma)
#
#
#Mc7 = Mc2*1e3
#Mht12 = M_t(R200, *unsigma)*1e-2
#
#mu = np.sqrt(5.266/(Mc7*Mht12**(2./3.))) # ecuacion 60
#print(mu, 'ecuacion 60')
#Mct7 = M_t(R99, *unsigma)*1e3 
#mu = 1.4e2*Mht12**(1./3.)/(Mct7)# ecuacion 42
#print(mu, 'ecuacion 42')
#mu = np.sqrt(np.sqrt(1.63e5/(Mc7*Mct7**2)))#execuacion 51
#print(mu, 'ecuacion 51')
#proba_eq(Rc2, Mc2, MctRc2, MctR992, dirfitsG= dirfitsG, 
#         nomb = 'cantidades_emcee2_1sig')


#print('wolfram script....')
#call(["wolframscript", "-file", "Plot_galaxy_densities_mathematica3.wls"])    
#call(["eog", "%s/bulgedens.png"%dirfitsG])    
#call(["eog", "%s/discdens.png"%dirfitsG])     
#call(["eog", "%s/galaxydens.png"%dirfitsG])
