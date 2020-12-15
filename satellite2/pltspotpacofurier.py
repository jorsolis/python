#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:49:16 2019

@author: jordi
"""

import plots_jordi as pts
import numpy as np
import pandas as pd


X =[]
Y=[]
X2=[]
Y2=[]

for modo in range(1,3,1):
    den = np.loadtxt("rhoadiferentest/rho%d_t_%d.txt"%(modo,0))   

    x=[]
    y = []
    
    for i in range(0,203):
        x.append(den[i][0])
        y.append(den[i][1])
    X.append(x)
    Y.append(y)


for modo in range(1,3,1):
    for nt in range(20,180,40):
        den = np.loadtxt("rhoadiferentest/rho%d_t_%d.txt"%(modo,nt))
    
        x=[]
        y = []
        
        for i in range(0,203):
            x.append(den[i][0])
            y.append(den[i][1])
        X.append(x)
        Y.append(y)

    fur = np.loadtxt("mdft_psi%d_max.dat"%modo)
    x=[]
    y = []
    
    for i in range(0,203):
        x.append(fur[i][0])
        y.append(fur[i][1])
    X2.append(x)
    Y2.append(y)

#pts.plotmultiple(X,Y,[r'$|\Psi_{100}|^2$',r'$|\Psi_{210}|^2$'],'$z$','',
#                 '','100210.png',xlim=(-6,6))
#
#pts.plotmultiple(X2,Y2,[r'Re$(\Psi_{100})$',r'Re$(\Psi_{210})$'],'$\gamma$',
#                 '','','fourier.png', xlim=(0,2.5))

uno = np.loadtxt('rho_1.t')
dos = np.loadtxt('rho2_1.t')

print(uno.T[1][0])
time = uno.T[0]
eme = uno.T[1]/uno.T[1][0]
print(dos.T[1][0])
time2 = dos.T[0]
eme2 = dos.T[1]/dos.T[1][0]
pts.plotmultiple([time[:-50], time2[:-50]],[eme[:-50], eme2[:-50]],[r'$M_{100}$', r'$M_{210}$'],'$t$',
                 '','','Multistate100210_M.png', xlim=(0,180))
