#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:15:23 2020

@author: jordi
"""

import numpy as np
import plots_jordi as pts
#from SP_shooting_axysimm_mixtate_100_21m import r95
from def_potenciales import multi_SFDM, SFDM, multistate, integralesferica, integral
pts.jordi_style()

state = 200
DM = multistate(state)
dirshoot = DM.dirshoot

rlam = 1
mu = 1


labelsn = []
erres = []
vint = []
rnn = []

R95=[]

def main(i, dirshoot):        
    r = SFDM(state, i).x
    V00, P0, Psi100, F1, Psi210, F2, E10, E20 = SFDM(state, i).funcs
    E1 = E10[-1]
    E2 = E20[-1]
    dF2 = 2.*(V00 - E2)*r**2*Psi210
    dF1 = 2.*(V00 - E1)*r**2*Psi100  
    N1 = integralesferica(Psi100**2, r, r[-1])
    N2 = integralesferica(Psi210**2, r, r[-1])
    W210 = 4.*np.pi*integralesferica(Psi210**2*V00, r, r[-1])
    K210 = -2.*np.pi*integralesferica(dF2*Psi210, r, r[-1])
    W100 = 4.*np.pi*integralesferica(Psi100**2*V00, r, r[-1])
    K100 = -2.*np.pi*integral(Psi100*dF1, r, r[-1])
    Et = (E1*N1 + E2*N2)/(N2+N1)
    print('%.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t '%(P0[-1], E1, E2, Et, N2/N1, Psi100[0]/Psi210[0]))
#    Nmax, N95, r95v = r95(r, P0, tol = 0.05)
#    R95.append(r95v)

    labelsn.append(P0[-1])
    
    rn = np.linspace(0, 10, 300)
    rnn.append(rn)
    
    SDM = multi_SFDM(state, rn, rlam, mu, i)
    vint.append(SDM.circ_vel())

DM.calc_cir_vel()    #####   para guardar los archivos de vel de toda la fam
      
for i in DM.nsols: 
    main(i, dirshoot)    
    
#pts.multiplot_colorbar(rnn, vint, labelsn, r'$\hat r$', r'$v/c$',
#                       '', '%s/vcirc2.png'%dirshoot, r'$N_T$', 
#                       ticks =labelsn, ylim=(0,2.3))
DM.plot_family()