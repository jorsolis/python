#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:33:48 2020

@author: jordi
"""

import numpy as np
import plots_jordi as pts

dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
data = np.loadtxt("%s/tab_rcmwall.dat.txt"%dirdata)
err = np.array([-data[:,4] + data[:,3], data[:,5] - data[:,3]])
rut = "/home/jordi/satellite/schrodinger_poisson/potpaco"
di = 'baja_dens/'
mue = 25

nsol = 6
r6 = np.load("%s/%spot_%d/model_pot_%d_r_m%d.npy"%(rut,di,nsol,nsol,mue))
y6 = np.load("%s/%spot_%d/model_pot_%d_y_m%d.npy"%(rut,di,nsol,nsol,mue))

nsol = 2
r2 = np.load("%s/%spot_%d/model_pot_%d_r_m%d.npy"%(rut,di,nsol,nsol,mue))
y2 = np.load("%s/%spot_%d/model_pot_%d_y_m%d.npy"%(rut,di,nsol,nsol,mue))

pts.plotmultiple([r6,r6,r6,r6,r2], [y6[0], y6[1],y6[4],y6[3], y2[3]],
                 ['Disk', 'Bulge', 'Disk+bulge',
                  r'Total $\frac{M_{100}}{M_{210}}=3$',
                  r'Total $\frac{M_{100}}{M_{210}}=0.36$'],
                 r'$r$(kpc)', r'$v$(km/s)', '', "%s/%sRC_m%d.png"%(rut,di,mue), ylim=(0,450),
                 xlim=(0,29.5),
                 save = True, loc_leg='upper right', angular=False,
                 markersize = 20,
                 data=True, xd=data[:,0],yd=data[:,3],err=True, yerr= err)


