#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

Fit MCMC de la via lactea 

DM              mixSFDM                 pars      mu, lambda
Disco           Razor exponential       pars      Md, ad
Inner bulge     Exponential             pars      Mbi, bbi
Bulge           Exponential             pars      Mb, bb
BH              Newtonian               pars      MBH

Data            datalogRCMW

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
from pymc_tools import todas_las_trazas, sigmas, popt_up_dw
from constants_grav import G, Gentc2, hc, c

###############################################################################
np.random.seed(12345)
galaxy = 'Milky Way'
nsamples = 1e3
paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/mixSFDM/'%dirdata
    datalogRCMW= np.loadtxt("%s/LogRC_data.dat.txt"%dirdata) 
    rad,_, v, v_error = datalogRCMW.T
#    rad,_, v, v_error = rad[:57],_, v[:57], v_error[:57]
    
ncor = 6  ## 4 y 5 m = 1
    
def vdm(x, rlam, mu):
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))
    xn = x2[:-150]/(mu*rlam)
    vdmn = rlam*vdm[:-150]*c
    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate" 
                  )
    return ve(x)
MO = {'rlam' : [0.001, 0.1, 0.02], 'mu' : [1.556, 155.6,  15.5655], 
      'Md' : [1e-2, 30., 8.15], 'ad' : [0.1, 5., 3.86],
      'Mb': [.1, 1.5, 0.86], 'bb' : [0.05, 0.5, 0.15],
      'Mbi': [0.001,0.050, 0.02], 'bbi' : [0.001, 0.100, 0.01],
      'BH' : [1e-4, 1e-3, .00037]}

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
    model = func(r,rlam,mu, Md, ad, Mb, bb, Mbi, bbi, MBH)
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
    return func(r,rlam,mu, Md, ad, Mb, bb, Mbi, bbi, MBH) 
    
params = lm.Parameters()
params.add('rlam', value = MO['rlam'][2], min = MO['rlam'][0], max = MO['rlam'][1])
params.add('mu', value = MO['mu'][2], min = MO['mu'][0], max = MO['mu'][1])
params.add('Md', value = MO['Md'][2], min = MO['Md'][0], max = MO['Md'][1])
params.add('ad', value = MO['ad'][2], min = MO['ad'][0], max = MO['ad'][1])
params.add('Mb', value = MO['Mb'][2], min = MO['Mb'][0], max = MO['Mb'][1])
params.add('bb', value = MO['bb'][2], min = MO['bb'][0], max = MO['bb'][1])
params.add('Mbi',value = MO['Mbi'][2],min = MO['Mbi'][0],max = MO['Mbi'][1])
params.add('bbi',value = MO['bbi'][2],min = MO['bbi'][0],max = MO['bbi'][1])
params.add('MBH',value = MO['BH'][2], min = MO['BH'][0], max = MO['BH'][1])

#params['mu'].set(vary = False)

params['Mbi'].set(vary = False)
params['bbi'].set(vary = False)
params['MBH'].set(vary = False)
################################################################################
#############################        MCMC        ##############################3
###############################################################################
res = lm.minimize(residual, args=(rad, v, v_error), method='emcee',
                  nan_policy='omit', burn = int(.3*nsamples), steps = nsamples, 
#                  nwalkers = 300,
                  params = params, is_weighted = True )
print(lm.report_fit(res.params))
print(res.params.pretty_print())

unsigma, dossigma = sigmas(res, dirfitsG, ncor, ID='v2')
popt_up, popt_dw = popt_up_dw(res, dirfitsG, ncor, ID='v2')

traces= {r'$\sqrt{\lambda}$':np.array(res.flatchain['rlam']),
         r'$\mu$':np.array(res.flatchain['mu']),
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

#rlamu, Mdu, adu, Mbu, bbu = popt_up
#rlamd, Mdd, add, Mbd, bbd= popt_dw
#muu = mud = mu

rlamu, muu, Mdu, adu, Mbu, bbu = popt_up
rlamd, mud, Mdd, add, Mbd, bbd= popt_dw

y_min= func(r, rlamd, mud, Mdd, add, Mbd, bbd, 0, 1, 0)
y_max= func(r, rlamu, muu, Mdu, adu, Mbu, bbu, 0, 1, 0)
DMlab =  r'DM  $\sqrt{\lambda} = %.5f$, $\mu = %f \times 10^{-25}$ eV/$c^2$'%(rlam,mu*hc*1e25)
dsclab = r'Disc $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad)
blginlab = r'Inner Bulge $M_{bi} = %.1f\times 10^{7} M_\odot$, $a_{bi}=%.2f$pc'%(Mbi*1e3,bbi*1e3)
blglab= r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $a_b=%.2f$pc'%(Mb,bb*1e3)
BHlab = r'BH, $M_{BH}=%.4f\times 10^{6} M_\odot$'%(MBH*1.0e4)
pts.plotmultiple([r, r, r, r, r, r, r], [modelo(res.params,r), 
                  np.sqrt(vdisk**2 + vbulge**2+ vbulgei**2+ vBH(r, G, MBH)**2),
                  vdm(r, rlam, mu), vdisk, vbulgei, vbulge, vBH(r, G, MBH)],
                 [r'Disc+Bulge+BH+SFDM', r'Disc+Bulge+ BH',
                  DMlab, dsclab, blginlab, blglab, BHlab], r'$r$(kpc)',
                  r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_emcee_v2.png'%(dirfitsG),
                 xlim = (rmin,rmax), #ylim = (20,500),
                 data = True, xd = rad, yd = v, err = True, yerr = v_error,
                 fill_between = True,fbx = r, fby1 = y_min, fby2 = y_max,
                 logx = True, logy = True, xv = [bbi, bb, ad])
pts.plotmultiple([r, r, r, r, r, r, r],
                 [modelo(res.params,r), 
                  np.sqrt(vdisk**2 + vbulge**2 + vbulgei**2 + vBH(r, G, MBH)**2),
                  vdm(r, rlam, mu), vdisk, vbulgei, vbulge,vBH(r, G, MBH)],
                 [r'Disc+Bulge+BH+SFDM', r'Disc + Bulge + BH', DMlab, dsclab, blginlab, blglab, BHlab],
                 r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                 '%sDM_fit_emcee2_v2.png'%(dirfitsG), xlim = (0, 30),
                 ylim = (0,300), data = True, xd = rad, yd = v, err = True, 
                 yerr = v_error, xv = [bbi, bb, ad],
                 fill_between = True, fbx = r, fby1 = y_min, fby2 = y_max)

#name =   [r'$\sqrt{\lambda}$', r'$M_d$', r'$a_d$', r'$M_b$',r'$b_b$']

name =   [r'$\sqrt{\lambda}$', r'$\mu$', r'$M_d$', r'$a_d$',
          r'$M_b$',r'$b_b$']

#name =   [r'$\sqrt{\lambda}$', r'$M_d$', r'$a_d$',
#          r'$M_b$',r'$b_b$', r'$M_{bi}$', r'$b_{bi}$', r'$M_{BH}$']

todas_las_trazas(traces, name, '%sDM_fit_MCMC_emcee_pars_v2.png'%(dirfitsG),
                 point_estimate="mean")