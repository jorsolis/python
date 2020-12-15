#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:20:13 2020

           SPARC
      
@author: jordi
"""
import numpy as np
import plots_jordi as pts
from SPARC_desc import tipos, data_dict
dirdata = '/home/jordi/SPARC'

if __name__ == '__main__':   
    withbulge= (31, 40, 46, 47, 50, 67, 73, 74, 77, 80, 81, 86, 88, 90, 92, 93, 95, 107, 108, 109, 110, 111, 112, 113, 120, 132, 135, 136, 141, 160, 163, 169)
    bulgeless = (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 70, 71, 72, 75, 76, 79, 82, 83, 84, 85, 87, 89, 94, 97, 98, 100, 102, 103, 105, 106, 114, 115, 116, 117, 118, 119, 122, 123, 124, 125, 126, 127, 128, 130, 131, 133, 134, 137, 138, 140, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175)
    gdag = 1.2e-10/1e3 #km/s**2
#    for i in bulgeless:
    for i in withbulge:   
        data = np.loadtxt("%s/%d.dat"%(dirdata,i))   
        rad, vobs, err, vgas, vdisk, vbul, _, _ = data.T
        effrad = data_dict['Effective Radius at [3.6]'][i - 1]
        name = data_dict['Name'][i - 1]
        tipo = tipos[data_dict['Type'][i - 1]]
        vbar = np.sqrt(vdisk**2 + vgas**2 + vbul**2)              
#        pts.plotmultiple([rad, rad, rad, rad],
#                         [vgas, vdisk, vbul, vbar],
#                         ['gas', 'disk', 'bulge','baryons', 'data'],
#                         r'$r$(kpc)',r'$v$(km/s)',
#                         tipo + name,'',
#                         data=True, xd=rad, yd=vobs, err=True, yerr=err)
        radm = rad*3.086e16
#        pts.plotmultiple([rad, rad, rad, rad],
#                         [vbar**2/radm, np.ones(np.shape(rad)[0])*2.54*gdag],
#                         [r'$g_{bar}$', r'$2.54 g^\dagger$'], r'$r$(kpc)',r'$g_{bar}$(km/$s^2$)',
#                         tipo + name,'')
        pts.plotmultiple([rad, rad, rad, rad],
                         [vobs**2/radm, np.ones(np.shape(rad)[0])*2.54*gdag],
                         [r'$g_{bar}$', r'$2.54 g^\dagger$'], r'$r$(kpc)',r'$g_{bar}$(km/$s^2$)',
                         tipo + name,'')     
#    pts.histo(MTL,r'$\gamma$', bins =bins,normalized = False,
#              nom_archivo ='%s/MtL_%s.png'%(dirfitsG, ID))      
    

            