#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:43:11 2020
  
                                  NO BORRAR NUNCA
      
@author: jordi
"""
import plots_jordi as pts
import numpy as np
from MW_M31_rotation_curve import (fitting, plot_data, ord_data)
from constants_grav import c, G, Gentc2, hc
from def_potenciales import (dens_Miyamoto,dens_Miyamoto_Nagai,M_Miyamoto,
                             M_Miyamoto2,M_hernquist,dens_doble_exp,
                             M_doble_disk, f, g, RC_miyamoto, 
                             vnagai, RC_exponential, exp_bulge,vBH,
                             RC_double_exponential_aprox,
                             v2_DM, M_CNFW, M_exp_disk, M_two_bulge)
sat = '/home/jordi/satellite'
dirfitsG = '/home/jordi/satellite/MW_rotation_curve_data/Fits/Gaussian'
dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
dataMW = np.loadtxt("%s/tab_rcmwall.dat.txt"%dirdata)
errMW = np.array([-dataMW[:,4] + dataMW[:,3], dataMW[:,5] - dataMW[:,3]])
dataMW2 = np.loadtxt("%s/tab_rcmw.dat.txt"%dirdata) 
errMW2 = np.array([-dataMW2[:,4] + dataMW2[:,3], dataMW2[:,5] - dataMW2[:,3]])
modelMW = np.loadtxt("%s/tab_rcmw-model.dat.txt"%dirdata) 
dataM31 = np.loadtxt('%s/M31_rotation_curve_data.txt'%sat)
dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 

galaxy = 'Milky Way'

if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/Gaussian/'%dirdata
    datalogRCMW= np.loadtxt("%s/LogRC_data.dat.txt"%dirdata) 
    rad,_, v, v_error = datalogRCMW.T
#    rad,_, v, v_error = rad[:54],_, v[:54], v_error[:54]               

def func_CNFW(r, Rc, Mc, re, rs, Md, ad, Mb, bb, MBH):## M es entre 10^10 M_sol
    ve2 = v2_DM(r, G, Rc, Mc, re, rs)
#    vdisk = RC_double_exponential_aprox(r, G, Md, ad, bd)
#    vdisk = RC_miyamoto(r, G, Md, ad, bd)
    vdisk = RC_exponential(r, G, Md, ad)
    vbul = exp_bulge(r, G, Mb, bb)
#    vbul = RC_miyamoto(r, G, Mb, 0., bb)
    return np.sqrt(ve2 + vdisk**2 + vbul**2 + vBH(rad, G, MBH)**2) 

def M_t(r, Rc, Mc, re, rs, Md, ad, Mb, bb, MBH):
    Mhalo = M_CNFW(r, Rc, Mc, re, rs)

#    Mdisk =  M_Miyamoto(r, Md, ad, bd)
#    Mdisk = M_doble_disk(r, Md, ad, bd)
#    Mbulge = M_Miyamoto(r, Mb, 0., bb)
#    Mbulge = M_hernquist(r, Mb, bb)
   
    Mdisk =  M_exp_disk(r, Md, ad)
    Mbulge = M_two_bulge(r, Mb, bb, 0, 1)

    return Mhalo + Mdisk + Mbulge + MBH

if __name__ == '__main__':
#    plot_data(dataMW, dataM31, modelMW)
    bound = [[0.001, 0.1, 0.001, 0.001, 0.1,  0.1,    0.1, 0.05, 1e-4], 
             [0.28,  4.,  1.00,   30. , 30.,  10.,   1.5, 1.0,  1e-3]]

    name =   ['Rc', 'Mc', 're',  'rs',  'Md', 'ad',   'Mb', 'bb', 'BH']
    
    rmin, rmax = np.amin(rad), np.amax(rad)
    
    r = np.linspace(rmin, rmax, 1000)
    loss = ['linear', 'soft_l1','huber','cauchy' ,'arctan']  
    for i in loss:
        try:
            popt, popt_up, popt_dw, r2, _ = fitting(func_CNFW, rad, v, bound, name, 
                                                error = True,err= v_error,
                                                    loss = 'linear')
    #                                                        loss = i)
            Rc, Mc, re, rs, Md, ad, Mb, bb, MBH = popt     
            y_fit = func_CNFW(rad, *popt)
            vdisk = RC_exponential(r, G, Md, ad)
            vbul = exp_bulge(r, G, Mb, bb)            
            pts.plotmultiple([rad,r, r, r, r,r], [y_fit, np.sqrt(vdisk**2 + vbul**2),
                             vdisk, vbul, np.sqrt(v2_DM(r, G, Rc, Mc, re, rs)),
                             vBH(r, G, MBH)],
                             [r'Disk+Bulge+SFDM', r'Disk+Bulge',
                              r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
                              r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$kpc'%(Mb,bb), 
                              r'DM  $R_c = %.1f$ kpc, $M_c = %.1f \times 10^{10} M_\odot$, $r_e=%.2f$kpc, $r_s=%.1f$kpc'%(Rc, Mc, re, rs),
                              r'BH, $M_{BH}=%.4f\times 10^{6} M_\odot$'%(MBH*1.0e4),'Observed'],
                             r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                             '%sDM_fit_MCMC_v5.png'%(dirfitsG),
                             xlim = (rmin,rmax), data = True, xd = rad, yd = v, err = True,
                             yerr = v_error, fill_between = True, logx = True,logy = True,
                             fbx = rad, fby1 = func_CNFW(rad, *popt_dw), fby2 = func_CNFW(rad, *popt_up), xv=[Rc, re])
            
        except RuntimeError:
                print(i, 'Optimal parameters not found')
#                print('Optimal parameters not found')
    
