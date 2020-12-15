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
dirdata = '/home/jordi/LSB_galaxies/URC'

bines = {1:r'$v_{opt}=$[24-60]km/s', 2:r'$v_{opt}=$[60-85]km/s', 
         3:r'$v_{opt}=$[85-120]km/s', 4:r'$v_{opt}=$[120-154]km/s',
         5:r'$v_{opt}=$[154-300]km/s'}
def plot_data():
    for i in range(1, 6,1):
        _, r,_, v, verr = np.loadtxt('%s/bin%d.txt'%(dirdata, i)).T
        plt.errorbar(r, v, yerr=verr,fmt='o',alpha = 0.95,
                     label = bines[i], 
                             ms = 2.5, capsize = 2.5, elinewidth = 0.5)  
    plt.xlabel(r'$r$(kpc)')
    plt.ylabel(r'$v$(km/s)')
    plt.legend()
    plt.title('LSB URCs',fontsize=20)
    plt.savefig('%s/URC.png'%dirdata, bbox_inches='tight')
    plt.show()

    for i in range(1, 6,1):
        r, _,_, v, verr = np.loadtxt('%s/bin%d.txt'%(dirdata, i)).T
        plt.errorbar(r, v, yerr=verr,fmt='o',alpha = 0.95,
                     label = bines[i], 
                             ms = 2.5, capsize = 2.5, elinewidth = 0.5)  
    plt.xlabel(r'$r/R_d$')
    plt.ylabel(r'$v$(km/s)')
    plt.legend()
    plt.title('LSB URCs',fontsize=20)
    plt.show()
if __name__ == '__main__':
    plot_data()  

