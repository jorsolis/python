#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 23:55:20 2020

@author: jordi
"""
import pandas as pd
dirdata = '/home/jordi/SPARC'


X = pd.read_csv("%s/R_descripcion.txt"%dirdata, sep = " ",
                skipinitialspace = True, header = None)

a= ['Name','Type','Distance','Mean error on D','Distance Method',
    'Inclination','Mean error on Inc','Total Luminosity at [3.6]',
    'Mean error on L[3.6]','Effective Radius at [3.6]',
    'Effective Surface Brightness at [3.6]','Disk Scale Length at [3.6]',
    'Disk Central Surface Brightness at [3.6]','Total HI mass',
    'HI radius at 1 Msun over pc2','Asymptotically Flat Rotation Velocity',
    'Mean error on Vflat','Quality Flag','References for HI and Ha data']

X.columns = a
data_dict = X.to_dict('series') 
  
tipos = {0 : 'S0 ',  1 : 'Sa ', 2 : 'Sab', 3 : 'Sb ', 4  : 'Sbc', 5 : 'Sc ',
         6 : 'Scd', 7  : 'Sd ', 8 : 'Sdm', 9 : 'Sm ', 10 : 'Im ', 11 : 'BCD'}
quality = {1:'High', 2:'Medium', 3:'Low'}
if __name__ == '__main__':
#    print(X.head())
#    print(X['Type'].values)
#    print(data_dict['Type'])
#    print(data_dict['Type'][0])
#    print(tipos[data_dict['Type'][0]])
#    print(data_dict['Name'])
#    for i in range(0, 175,1):
#        if data_dict['Type'][i]== 11:
##        if data_dict['Quality Flag'][i]== 1:
#            print('No file', i+1, ', ', tipos[data_dict['Type'][i]],
#    #                  ', name:',data_dict['Name'][i],
#    #                  ', Quality:', quality[data_dict['Quality Flag'][i]],
#                  ', Eff. radius:', data_dict['Effective Radius at [3.6]'][i],'kpc',
#                  ', Disk scale lenght:', data_dict['Disk Scale Length at [3.6]'][i]
#                  )
    cont = 0
#    for i in range(1, 176):
#        if data_dict['Quality Flag'][i - 1] == 3:
#            if 
#            cont +=1
#    print(cont)