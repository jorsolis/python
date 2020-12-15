#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

Fit MCMC de la via lactea 

DM          mixSFDM             pars        mu, lambda
Disco       Miyamoto-Nagai      pars        Md, ad, bd
Bulge       Miyamoto-Nagai      pars        Mb, bb

@author: jordi
"""
import plots_jordi as pts
from scipy.interpolate import interp1d
import numpy as np
from def_potenciales import vnagai
import lmfit as lm
from pymc_tools import todas_las_trazas, sigmas, popt_up_dw
from constants_grav import G, c, hc
###############################################################################
np.random.seed(12345)
galaxy = 'Milky Way'
nsamples = 1e4

#nsamples= 1e3

paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'

ncor = 6 ## 4 y 5 m = 1

def ord_data(dataMW):
    r = dataMW[:,0]
    vel = dataMW[:,3]
    err = errMW[0,:]    
    sortindex = np.argsort(r, axis = 0)
    rord = np.sort(r, axis= 0)    
    velord = vel[sortindex]
    errord = err[sortindex]
    return rord, velord, errord

def noceros_error(error):
    for i in range(0, np.shape(error)[0],1):
        if error[i]==0:
            error[i]=1
    return error

if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/mixSFDM/'%dirdata
    dataMW = np.loadtxt("%s/tab_rcmwall.dat.txt"%dirdata)
    errMW = np.array([-dataMW[:,4] + dataMW[:,3], dataMW[:,5] - dataMW[:,3]])
    rad, v, verror = ord_data(dataMW)
    v_error = noceros_error(verror)

#    dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 
#    rad, v, v_error = dataGrandMW.T
#    rad, v, v_error = rad[:-8], v[:-8], v_error[:-8]

def vdm(x, rlam, mu):
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))
    xn = x2[:-150]/(mu*rlam)
    vdmn = rlam*vdm[:-150]*c
#    vdmn = rlam**2vdm[:-150]*c###viejo antes servia
    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
                  fill_value = "extrapolate" 
                  )
    return ve(x)

MO = {'rlam' : [0.001, 0.1, 0.02], 'mu' : [15.56, 155.6,  15.5655], 
      'Md' : [1e-2, 7., 3.15], 'ad' : [0.1, 10., 5.86],
      'bd' : [.8, 3., 1.15],
      'Mb': [.1, 1.5, 0.86], 'bb' : [0.05, 0.5, 0.15]}

#MO = {'rlam' : [0.03, 0.1, 0.04], 'mu' : [1.556, 155.6,  15.5655],
#      'Md' : [1e-2, 30., 8.15], 'ad' : [0.1, 10., 3.86],
#      'bd' : [.8, 3., 1.15],
#      'Mb': [.1, 1.5, 0.86], 'bb' : [0.05, 0.5, 0.15]}###viejo antes servia

def func(r,rlam,mu, Md, ad,bd, Mb, bb):
    vh = vdm(r, rlam, mu)
    vdisk = vnagai(r, G=G, M = Md, a = ad, b = bd)
    vbul = vnagai(r, G=G, M = Mb, a = bb, b = 0)
    model =  np.sqrt(vh**2 + vdisk**2 + vbul**2) 
    return model  

def residual(params, r, data, eps_data):
    rlam = params['rlam']
    mu = params['mu']
    Md = params['Md']
    ad = params['ad']
    bd = params['bd']
    Mb = params['Mb']
    bb = params['bb']
    model = func(r, rlam, mu, Md, ad,bd, Mb, bb)
    return (data-model) / eps_data

def modelo(params, r):
    rlam = params['rlam']
    mu = params['mu']
    Md = params['Md']
    ad = params['ad']
    bd = params['bd']
    Mb = params['Mb']
    bb = params['bb']
    return func(r, rlam, mu, Md, ad,bd, Mb, bb)

params = lm.Parameters()
params.add('rlam', value = MO['rlam'][2], min = MO['rlam'][0], max = MO['rlam'][1])
params.add('mu', value = MO['mu'][2], min = MO['mu'][0], max = MO['mu'][1])
params.add('Md', value = MO['Md'][2], min = MO['Md'][0], max = MO['Md'][1])
params.add('ad', value = MO['ad'][2], min = MO['ad'][0], max = MO['ad'][1])
params.add('bd', value = MO['bd'][2], min = MO['bd'][0], max = MO['bd'][1])
params.add('Mb', value = MO['Mb'][2], min = MO['Mb'][0], max = MO['Mb'][1])
params.add('bb', value = MO['bb'][2], min = MO['bb'][0], max = MO['bb'][1])
##############################################################################
###################           FIT CHI^2            ###########################
#############################################################################
#params['mu'].set(vary = False)
#
#
##params['rlam'].set(vary = False)
#
#
out = lm.minimize(residual, params, args=(rad, v, v_error))
print(lm.printfuncs.report_fit(out.params, min_correl=0.5))
print(out.params.pretty_print())
rlam, mu, Md, ad, bd, Mb, bb = list(out.params.valuesdict().values())

rmin, rmax = 1e-3, np.amax(rad)
#rmin, rmax = 1e-1, np.amax(rad)

r = np.linspace(rmin, rmax, 100000)

vdisk = vnagai(r, G=G, M = Md, a = ad, b = bd)
vbulge = vnagai(r, G=G, M = Mb, a = bb, b = 0)

DMlab =  r'DM  $\sqrt{\lambda} = %.5f$, $\mu = %f \times 10^{-25}$ eV/$c^2$'%(rlam,mu*hc*1e25)
dsclab = r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad)
blglab= r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3)

pts.plotmultiple([r, r, r, r, r, r, r],
                 [modelo(out.params,r), 
                  np.sqrt(vdisk**2 + vbulge**2),
                  vdm(r, rlam, mu), vdisk, vbulge],
                 [r'Disc+Bulge+SFDM', r'Disc+Bulge', DMlab, dsclab,blglab],
                  r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_emcee2_nsol%d_CHI2_v4.png'%(dirfitsG, ncor),
                 xlim = (rmin,rmax),
                 ylim = (0,300), data = True, xd = rad, yd = v, err = True, 
                 yerr = v_error)
#pts.plotmultiple([r], [vdm(r, rlam, mu)],
#                 [r'DM  $\sqrt{\lambda} = %.5f$, $\mu= %f \times 10^{-25}$'%(rlam, mu*hc*1e25)],
#                  r'$r$(kpc)', r'$v$(km/s)', 'Miky Way', '')
###############################################################################
############################        MCMC        ##############################3
##############################################################################
#par = out.params
#par['bd'].set(vary = False)
#res = lm.minimize(residual, args=(rad, v, v_error), method='emcee',
#                  nan_policy = 'omit', burn = int(.3*nsamples), 
#                  steps=nsamples, params = par, is_weighted = True)
#print(lm.report_fit(res.params))
#print(res.params.pretty_print())
##lm.model.save_modelresult(res, 'result_emcee_nsol%d.sav'%(ncor))
#            
#print('Chi2red', res.redchi)    
#unsigma, dossigma = sigmas(res, dirfitsG, ncor, ID = 'v4')
#popt_up, popt_dw = popt_up_dw(res, dirfitsG, ncor, ID='v4')
#
#rlam, mu, Md, ad, bd, Mb, bb = list(res.params.valuesdict().values())
#popt = np.array([rlam, mu, Md, ad, bd, Mb, bb])
#np.save('%spopt_nsol%d_v4.npy'%(dirfitsG,ncor), popt)
#
#unsigma = np.load('%sunsigma_nsol%d_v4.npy'%(dirfitsG,ncor))
#dossigma = np.load('%sdossigma_nsol%d_v4.npy'%(dirfitsG,ncor))
#
#rlam, mu, Md, ad, bd, Mb, bb = np.load('%spopt_nsol%d_v4.npy'%(dirfitsG,ncor))
#
#rlamu, muu, Mdu, adu, Mbu, bbu = np.load('%spoptup_nsol%d_v4.npy'%(dirfitsG,ncor))
#rlamd, mud, Mdd, add, Mbd, bbd = np.load('%spoptdw_nsol%d_v4.npy'%(dirfitsG,ncor))
#
##rlamu, Mdu, adu, Mbu, bbu = np.load('%spoptup_nsol%d_v4.npy'%(dirfitsG,ncor))
##rlamd, Mdd, add, Mbd, bbd = np.load('%spoptdw_nsol%d_v4.npy'%(dirfitsG,ncor))
##muu = mud = mu
#
#rmin, rmax = 1e-1, np.amax(rad)
#r = np.linspace(rmin, rmax, 100000)
#
#vdisk = vnagai(r, G=G, M = Md, a = ad, b = bd)
#vbulge = vnagai(r, G=G, M = Mb, a = bb, b = 0)
#y_min= func(r, rlamd, mud, Mdd, add, bd, Mbd, bbd)
#y_max= func(r, rlamu, muu, Mdu, adu, bd, Mbu, bbu)
#fited = modelo(res.params,r)
#
#pts.plotmultiple([r, r, r, r, r, r, r], 
#                 [fited, np.sqrt(vdisk**2 + vbulge**2),
#                  vdm(r, rlam, mu), vdisk, vbulge],
#                 [r'Disc+Bulge+SFDM', r'Disc+Bulge',
#                  r'DM  $\sqrt{\lambda} = %.5f$, $\mu= %f \times 10^{-25}$'%(rlam, mu*hc*1e25),
#                  r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
#                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3), 
#                  'Observed'], r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
#                 '%sDM_fit_emcee_nsol%d_v4.png'%(dirfitsG, ncor),
#                 xlim = (rmin,rmax), #ylim = (20,500),
#                 data = True, xd = rad, yd = v, err = True, yerr = v_error,
#                 fill_between = True,fbx = r, fby1 = y_min, fby2 = y_max,
#                 logx = True,logy = True)
#pts.plotmultiple([r, r, r, r, r, r, r],
#                 [fited, np.sqrt(vdisk**2 + vbulge**2),
#                  vdm(r, rlam, mu), vdisk, vbulge],
#                 [r'Disc+Bulge+SFDM', r'Disc+Bulge',
#                  r'DM  $\sqrt{\lambda} = %.5f$, $\mu= %f \times 10^{-25}$'%(rlam, mu*hc*1e25),
#                  r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
#                  r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3)],
#                  r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
#                 '%sDM_fit_emcee2_nsol%d_v4.png'%(dirfitsG, ncor),
#                 xlim = (0,rmax), loc_leg='lower right',
#                 ylim = (0,300), data = True, xd = rad, yd = v, err = True, 
#                 yerr = v_error, 
#                 fill_between = True, fbx = r, fby1 = y_min, fby2 = y_max)
#
#pts.residual(v, modelo(res.params,rad), datalabel=r'$v$(km/s)', lowess=True)
#
#name =   [r'$\sqrt{\lambda}$', r'$\mu$', r'$M_d$', r'$a_d$',
#          r'$M_b$',r'$b_b$']
#traces= {r'$\sqrt{\lambda}$':np.array(res.flatchain['rlam']),
#         r'$\mu$':np.array(res.flatchain['mu']),
#           r'$M_d$':np.array(res.flatchain['Md']),
#           r'$a_d$':np.array(res.flatchain['ad']),
##           r'$b_d$':np.array(res.flatchain['bd']),
#           r'$M_b$':np.array(res.flatchain['Mb']),
#           r'$b_b$':np.array(res.flatchain['bb']),
#           }
#todas_las_trazas(traces, name, '%sDM_fit_MCMC_emcee_pars_nsol%d_v4.png'%(dirfitsG, ncor),
#                 point_estimate="mode")
#
##import corner
##emcee_corner = corner.corner(res.flatchain,
##                             labels=res.var_names,
##                             truths=list(res.params.valuesdict().values()))
