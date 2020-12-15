#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:20:20 2020

@author: jordi
"""

c = 2.99792458e5 ## km/s
Gentc2 = 4.799e-7 ##G/c^2 en kpc/(10^10 M_sol)
G = Gentc2*c*c

mukpc = {20 : 1565000.,
         21 : 156500.,
         22 : 15650., 
         23 : 1565.1, 
         24 : 156.51, 
         25 : 15.651,
         26 : 1.5651}#1/kpc

ckpc = 0.307*1e-3 #kpc/year
mupc = {21:156.51, 22: 15.651, 23:1.5651, 24:0.15651, 25:0.015651}#1/pc
hc = 0.1973269804e-6/3.086e19 # eV*kpc

letr = {'theha': '\u03B8', 'lambda':'\u03BB', 'mu':'\u03BC', 'rho': '\u03C1'}
if __name__ == '__main__':
    print(G)
    print(mukpc[22])
    greek_letterz=[chr(code) for code in range(945,970)]
    codes =[code for code in range(945,970)]
    print(greek_letterz)
    print(codes)