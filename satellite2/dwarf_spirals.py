#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:35:23 2020

@author: jordi
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plots_jordi import jordi_style
jordi_style()
dirdata = '/home/jordi/The_universal_rotation_curve_of_dwarf_disk_galaxies'

def read_original_data():
    dat = pd.read_csv('%s/Galaxy_RCs_vopt.txt'%(dirdata), sep = "\t",
                    skipinitialspace = True, header = None, skip_blank_lines=False)
    dat.columns = ['Name','D','RD','Vopt', 'MK']
    for i in range(1, 37,1):
        a = np.loadtxt('%s/%d'%(dirdata,i)).T
        rn, vn, dvn = a
        r = 3.2*rn*dat['RD'][i-1]
        v = vn*dat['Vopt'][i-1]
        verr = dvn*dat['Vopt'][i-1]
        np.save('%s/%d.npy'%(dirdata,i), np.array([r, v, verr]))

def plot_data():
    for i in range(1, 37,1):
        r, v, verr = np.load('%s/%d.npy'%(dirdata,i))
        plt.errorbar(r, v, yerr=verr,fmt='o',alpha = 0.95,
                             ms = 2.5, capsize = 2.5, elinewidth = 0.5)# linestyle="None")    
    plt.xlabel(r'$r$(kpc)')
    plt.ylabel(r'$v$(km/s)')
    plt.title('Dwarf spirals RCs',fontsize=20)
    plt.show()

def plot_syntheticRC():
    dat = pd.read_csv('%s/URC_data.txt'%(dirdata), sep = "\t",
                    skipinitialspace = True, skip_blank_lines=False)  
    plt.errorbar(dat.Ri, dat.Vi, yerr=dat.dVi,fmt='o',alpha = 0.95,
                             ms = 2.5, capsize = 2.5, elinewidth = 0.5)    
    plt.xlabel(r'$r$(kpc)')
    plt.ylabel(r'$v$(km/s)')
    plt.title('Dwarf disc galaxies synthetic RC',fontsize=20)
    plt.ylim((0,50))
    plt.show()
    
if __name__ == '__main__':
    plot_data()  
    plot_syntheticRC()