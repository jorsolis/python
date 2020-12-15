#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 23:55:54 2020

@author: jordi
"""

import pandas as pd
from numpy import nan
import numpy as np
import plots_jordi as pts
dirdata = '/home/jordi/LSB_galaxies'

#X = pd.read_csv("%s/datafile1.txt"%dirdata, sep = "\t",
#                skipinitialspace = True, header = 0)
#print(X.head)
#print(X['HI'].values)

X2 = pd.read_csv("%s/datafile2.txt"%dirdata, sep = " ",
                skipinitialspace = True, header = None, skip_blank_lines=False)

a= ['Name','PosRad','Vel','e_Vel']

X2.columns = a

#is_NaN = X2.isnull()
#row_has_NaN = is_NaN.any(axis=1)
#rows_with_NaN = X2[row_has_NaN]
##print(rows_with_NaN)
#print(is_NaN)
#for i in range(0, 535):
##    print(X2['Name'][i])
#    if X2['Name'][i]==nan:
#        print('nan')

#fil= open("%s/Fits/Gaussian/cantidades.txt"%dirdata,"w+")
#
#fil.write('Nfile \t Name \t Type \t Rc \t errRc \t Mc \t errMc \t M/L \t errM/L \t r2 \r\n')
#fil.close()

for i in range(1,27,1):  
    data = pd.read_csv("%s/%d.txt"%(dirdata,i), sep = " ",
                skipinitialspace = True, header = None, skip_blank_lines=False)
    a= ['Name','PosRad','Vel','e_Vel']   
    data.columns = a
    rad = data['PosRad']
    vobs = data['Vel']
    err = data['e_Vel']

    name = data['Name'][0]
#    tipo = tipos[data_dict['Type'][i - 1]]
#    distance = data_dict['Distance'][i - 1]
#    effrad = data_dict['Effective Radius at [3.6]'][i - 1]
    
    pts.plotmultiple([rad],
                     [vobs],
                     ['obs'],
                     r'$r$(arcsec)', r'$v$(km/s)',
                     'Name: %s'%(name),
                     '%s/%d.png'%(dirdata,i), save = True, 
                     data=True, xd=rad, yd=vobs,err=True, yerr=err) 
