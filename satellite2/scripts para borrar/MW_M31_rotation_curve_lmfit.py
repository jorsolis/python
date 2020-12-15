#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

Fit MCMC de la via lactea 

DM          mixSFDM             pars        mu, lambda
Disco   Razor exponential       pars      Md, ad
Bulge   Exponential             pars      Mb, bb
BH      Newtonian               pars      MBH

@author: jordi
"""
import plots_jordi as pts
from scipy.interpolate import interp1d
import numpy as np
from def_potenciales import(vHernquist, vnagai, Miyamoto_Nagai_3,
                            M_Miyamoto2,M_hernquist, exp_bulge,M_exp_bulge,
                            M_exp_disk, M_two_bulge, vBH, RC_exponential,
                            v2_DM, M_CNFW)
import lmfit as lm
from pymc_tools import todas_las_trazas
from constants_grav import G, Gentc2, mu22, hc, c

###############################################################################
np.random.seed(12345)
galaxy = 'Milky Way'
nsamples = 1e3
paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/mixSFDM/'%dirdata

#    datalogRCMW= np.loadtxt("%s/LogRC_data.dat.txt"%dirdata) 
#    rad,_, v, v_error = datalogRCMW.T
#    rad,_, v, v_error = rad[:57],_, v[:57], v_error[:57]

    dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 
    rad, v, v_error = dataGrandMW.T
#    rad, v, v_error =rad[:118], v[:118], v_error[:118]
    
ncor = 6  ## 4 y 5 m = 1
    
def vdm(x, rlam, mu):
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))
    xn = x2[:-150]/(mu*rlam)
    vdmn = rlam**2*vdm[:-150]*c
    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate" 
                  )
    return ve(x)

#MO = {'rlam' : [0.001, 0.1, 0.02], 'mu' : [1.556, 155.6,  15.5655], 
#      'Md' : [1e-2, 30., 8.15], 'ad' : [0.1, 5., 3.86],
#      'Mb': [.1, 1.5, 0.86], 'bb' : [0.05, 0.5, 0.15],
#      'Mbi': [0.001,0.050, 0.02], 'bbi' : [0.001, 0.100, 0.01],
#      'BH' : [1e-4, 1e-3, .00037]}

MO = {'rlam' : [0.01, 0.1, 0.02], 'mu' : [1.556, 155.6,  15.5655], 
      'Md' : [1e-2, 30., 8.15], 'ad' : [0.1, 10., 3.86],
      'Mb': [.1, 1.5, 0.86], 'bb' : [0.05, 0.5, 0.15],
      'Mbi': [0.001,0.050, 0.0], 'bbi' : [0.001, 0.100, 0.01],
      'BH' : [1e-4, 1e-3, .0]}

def func(r,rlam,mu, Md, ad, Mb, bb, Mbi, bbi, MBH):
    vh = vdm(r, rlam, mu)
    vdisk = RC_exponential(r, G, Md, ad)
    vbul = exp_bulge(r, G, Mb, bb)
    vbuli = exp_bulge(r, G, Mbi, bbi)
    model =  np.sqrt(vh**2 + vdisk**2 + vbul**2 + vbuli**2 + vBH(r, G, MBH)**2) 
    return model 

def residual(params, r, data, eps_data):
    rlam = params['rlam']
    mu = params['mu']
    Md = params['Md']
    ad = params['ad']
    Mb = params['Mb']
    bb = params['bb']
    Mbi = params['Mbi']
    bbi = params['bbi']    
    MBH = params['MBH']    
    model =  func(r,rlam,mu, Md, ad, Mb, bb, Mbi, bbi, MBH)
    return (data-model) / eps_data

def modelo(params, r):
    rlam = params['rlam']
    mu = params['mu']
    Md = params['Md']
    ad = params['ad']
    Mb = params['Mb']
    bb = params['bb']
    Mbi = params['Mbi']
    bbi = params['bbi']
    MBH = params['MBH']
    model =  func(r,rlam,mu, Md, ad, Mb, bb, Mbi, bbi, MBH) 
    return model   
    
params = lm.Parameters()
params.add('rlam', value = MO['rlam'][2], min = MO['rlam'][0], max = MO['rlam'][1])
params.add('mu', value = MO['mu'][2], min = MO['mu'][0], max = MO['mu'][1],
           vary = False)
params.add('Md', value = MO['Md'][2], min = MO['Md'][0], max = MO['Md'][1])
params.add('ad', value = MO['ad'][2], min = MO['ad'][0], max = MO['ad'][1])
params.add('Mb', value = MO['Mb'][2], min = MO['Mb'][0], max = MO['Mb'][1])
params.add('bb', value = MO['bb'][2], min = MO['bb'][0], max = MO['bb'][1])
params.add('Mbi',value = MO['Mbi'][2],min = MO['Mbi'][0],max = MO['Mbi'][1],
           vary = False)
params.add('bbi', value = MO['bbi'][2], min = MO['bbi'][0], max = MO['bbi'][1],
            vary = False)
params.add('MBH', value = MO['BH'][2], min = MO['BH'][0], max = MO['BH'][1],
            vary = False)

################################################################################
#############################        MCMC        ##############################3
###############################################################################
res = lm.minimize(residual, args=(rad, v, v_error), method='emcee',
                  nan_policy='omit', burn=int(.3*nsamples), steps=nsamples, 
                  nwalkers = 300, params = params, is_weighted=True )
print(lm.report_fit(res.params))
print(res.params.pretty_print())

def sigmas(res, dirfitsG):
    # lets work out a 1 and 2-sigma error estimate for 'Rc'
    print('parameter   1 sigma spread  2 sigma spread')   
    fil= open("%s/cantidades_emcee.txt"%dirfitsG,"w+")   
    fil.write('name \t mean \t 1sigma \t 2sigma  \r\n')
    unsigma=[]
    dossigma=[]
#   print(res.var_names)
    for i in res.var_names:
        quantiles = np.percentile(res.flatchain[i], [2.28, 15.9, 50, 84.2, 97.7])      
        sigma1 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2 = 0.5 * (quantiles[4] - quantiles[0])
        prom = res.params[i].value
        unsigma.append(sigma1)
        dossigma.append(sigma2)
        print(i, sigma1, sigma2)
        fil.write('%s \t %.4f \t %.4f \t %.4f  \r\n'%(i, prom, sigma1, sigma2))
    fil.close()
    unsigma = np.array(unsigma); np.save('%sunsigma.npy'%(dirfitsG), unsigma)
    dossigma = np.array(dossigma); np.save('%sdossigma.npy'%(dirfitsG), dossigma)
    return unsigma, dossigma

unsigma, dossigma = sigmas(res, dirfitsG)
print(unsigma)
print(dossigma)

popt_dw = []; popt_up = []

for i in res.var_names[:]:
    per = np.percentile(res.flatchain[i], [2.5])
    per2 = np.percentile(res.flatchain[i], [97.5])
    popt_dw.append(per[0])
    popt_up.append(per2[0])
popt_dw = np.array(popt_dw); popt_up = np.array(popt_up)

traces= {r'$\sqrt{\lambda}$':np.array(res.flatchain['rlam']),
#         r'$\mu$':np.array(res.flatchain['mu']),
           r'$M_d$':np.array(res.flatchain['Md']),
           r'$a_d$':np.array(res.flatchain['ad']),
           r'$M_b$':np.array(res.flatchain['Mb']),
           r'$b_b$':np.array(res.flatchain['bb']),           
#           r'$M_{bi}$':np.array(res.flatchain['Mbi']),
#           r'$b_{bi}$':np.array(res.flatchain['bbi']),  
#           r'$M_{BH}$':np.array(res.flatchain['MBH'])
           }

rlam, mu, Md, ad, Mb, bb, Mbi, bbi, MBH = list(res.params.valuesdict().values())

#rmin, rmax = 1e-3, np.amax(rad)
rmin, rmax = 1e-1, np.amax(rad)

r = np.linspace(rmin, rmax, 100000)
vdisk = RC_exponential(r, G, Md, ad)
vbulge = exp_bulge(r, G, Mb, bb)
vbulgei = exp_bulge(r, G, Mbi, bbi)

#rlamu, Mdu, adu, Mbu, bbu, Mbiu, bbiu, MBHu = popt_up
#rlamd, Mdd, add, Mbd, bbd, Mbid, bbid, MBHd = popt_dw
#y_min= func(r, rlamd, mu, Mdd, add, Mbd, bbd, Mbid, bbid, MBHd)
#y_max= func(r, rlamu, mu, Mdu, adu, Mbu, bbu, Mbiu, bbiu, MBHu)

#y_min= func(r, *popt_dw)
#y_max= func(r, *popt_up)
pts.plotmultiple([r, r, r, r, r, r, r], 
                 [modelo(res.params,r), 
                  np.sqrt(vdisk**2 + vbulge**2+ vbulgei**2+ vBH(r, G, MBH)**2),
                  vdm(r, rlam, mu), vdisk, vbulgei, vbulge, vBH(r, G, MBH)],
                 [r'Disc+Bulge+BH+SFDM', r'Disc+Bulge+ BH',
                  r'DM  $\sqrt{\lambda} = %.5f$, $\mu = %f \times 10^{-23}$'%(rlam,mu*hc*1e23),
                  r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
                  r'Inner Bulge $M_{bi} = %.1f\times 10^{7} M_\odot$, $a_{bi}=%.2f$pc'%(Mbi*1e3,bbi*1e3), 
                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3), 
                  r'BH, $M_{BH}=%.4f\times 10^{6} M_\odot$'%(MBH*1.0e4),
                  'Observed'], r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_emcee.png'%(dirfitsG),
                 xlim = (rmin,rmax), #ylim = (20,500),
                 data = True, xd = rad, yd = v, err = True, yerr = v_error,
#                 fill_between = True,fbx = r, fby1 = y_min, fby2 = y_max,
                 logx = True,logy = True)
pts.plotmultiple([r, r, r, r, r, r, r],
                 [modelo(res.params,r), 
                  np.sqrt(vdisk**2 + vbulge**2 + vbulgei**2 + vBH(r, G, MBH)**2),
                  vdm(r, rlam, mu), vdisk, vbulgei, vbulge,vBH(r, G, MBH)],
                 [r'Disc+Bulge+BH+SFDM', r'Disc+Bulge + BH',
                  r'DM  $\sqrt{\lambda} = %.5f$, $\mu = %f \times 10^{-23}$'%(rlam,mu*hc*1e23),
                  r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
                  r'Inner Bulge $M_{bi} = %.1f\times 10^{7} M_\odot$, $a_{bi}=%.2f$pc'%(Mbi*1e3,bbi*1e3),
                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3), 
                  r'BH, $M_{BH}=%.4f\times 10^{6} M_\odot$'%(MBH*1.0e4),
                  'Observed'], r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_emcee2.png'%(dirfitsG),
#                 xlim = (rmin,rmax),
                 xlim = (0, 30),
                 ylim = (0,300), data = True, xd = rad, yd = v, err = True, 
                 yerr = v_error, 
#                 fill_between = True, fbx = r, fby1 = y_min, fby2 = y_max
                 )
#name =   [r'$\sqrt{\lambda}$', r'$\mu$', r'$M_d$', r'$a_d$',
#          r'$M_b$',r'$b_b$', r'$M_{bi}$', r'$b_{bi}$', r'$M_{BH}$']
#name =   [r'$\sqrt{\lambda}$', r'$M_d$', r'$a_d$',
#          r'$M_b$',r'$b_b$', r'$M_{bi}$', r'$b_{bi}$', r'$M_{BH}$']

name =   [r'$\sqrt{\lambda}$', r'$M_d$', r'$a_d$',
          r'$M_b$',r'$b_b$']

todas_las_trazas(traces, name, '%sDM_fit_MCMC_emcee_pars.png'%(dirfitsG),
                 point_estimate="mean")