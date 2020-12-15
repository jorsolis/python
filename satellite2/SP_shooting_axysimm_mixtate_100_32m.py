#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:15:23 2020

@author: jordi
"""

import numpy as np
import plots_jordi as pts
from def_potenciales import multi_SFDM, SFDM, integralesferica, integral,  multistate

pts.jordi_style()

state = 320
DM = multistate(state)
#DM.calc_cir_vel()    #####   para guardar los archivos de vel de toda la fam
dirshoot = DM.dirshoot

C = 2./np.sqrt(5)
C2 = 9./8.
DC = {0: {'m' : 0, 'c1' :  2.*np.sqrt(5)/7., 'c2' :  6./7.},
      1: {'m' : 1, 'c1' :     np.sqrt(5)/7., 'c2' : -4./7.},
      2: {'m' : 2, 'c1' : -2.*np.sqrt(5)/7., 'c2' :  1./7.}}

m = 0
c1 = DC[m]['c1']  
c2 = DC[m]['c2'] 

rlam = 1
mu = 1
V4s = []
labelsn = []
erres = []
vint = []
rnn = []
dens1 = []; dens2 = []; V0i = []; V2i =[]

R95=[]

def potencial_sph_coords(r, th, V00, V20, V40):
    return V00 + r**2*V20*np.sqrt(5)*(-1. + 3.*np.cos(th)**2)/2. + 3.*r**4*V40*(3. - 30.*np.cos(th)**2 +35.*np.cos(th)**4)/8.

def density_sph_coords(r, th, dens100, dens320):
    return dens100 + 5.*r**4*dens320*3.*(-1. + 3.*np.cos(th)**2)**2/4.

def main(i, dirshoot):    
    r = SFDM(320, i).x
    V00, P0, V20, P2, V40, P4, Psi100, F1, Psi320, F3, E10, E30 = SFDM(320, i).funcs
    E1 = E10[-1]
    E3 = E30[-1]
    dF3 = -6.*F3/r + 2.*(V00 + c1*r**2*V20 +c2*r**4*V40 - E3)*Psi320
    dF1 = 2.*(V00 - E1)*r**2*Psi100  
    N1 = integralesferica(Psi100**2, r, r[-1])
    N3 = integralesferica(r**4*Psi320**2, r, r[-1])
    Et = (E1*N1 + E3*N3)/(N3+N1)    
    W320 = 4.*np.pi*(integralesferica(r**4*Psi320**2*V00, r, r[-1]) + c1*integralesferica(r**4*Psi320**2*r**2*V20, r, r[-1])+ c2*integralesferica(r**4*Psi320**2*r**4*V40, r, r[-1]))
    K320 = -4.*np.pi*(6.*integralesferica(r*Psi320*F3, r, r[-1]) + integralesferica(r**2*dF3*Psi320, r, r[-1]))
    W100 = 4.*np.pi*integralesferica(Psi100**2*V00, r, r[-1])
    K100 = -4.*np.pi*integral(Psi100*dF1, r, r[-1])
#    print(i, 'N3/N1=',N3/N1, 'NT=', P0[-1])
#    print(i, 'WT= ', W320+W100, 'KT= ', K320+K100, 'K/W= ', (K320+K100)/abs(W320+W100))
#    print(i, 'E100=',E1 ,'E320=', E3, 'ET= ', Et)
#    print(i, 'Psi2=', Psi320[0], 'Psi1/Psi2= ', Psi100[0]/Psi320[0])
    print('%.1f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t '%(P0[-1], E1, E3, Et, N3/N1, Psi100[0]/Psi320[0]))

#    Nmax, N95, r95v = r95(r, P0, tol = 0.005)
#    Nmax, N95, r95v = r95(r, P0, tol = 0.05)
#    R95.append(r95v)
    labelsn.append(P0[-1])
    erres.append(r)
    V4s.append(V40)

    rn = np.linspace(0.0, 10, 100)
    rnn.append(rn)
    
    SDM = multi_SFDM(320, rn, rlam, mu, i)
    vint.append(SDM.circ_vel())
#     
    uno, dos = SDM.density()
    dens1.append(uno)
    dens2.append(dos)
##
    Vcero, Vdos, Vcuat = SDM.potential()
    V0i.append(Vcero)
    V2i.append(Vdos)
##    
#    th = np.linspace(0, np.pi, 101)
#    R, TH = np.meshgrid(rn, th)   
#
#    f = potencial_sph_coords(R, TH, Vcero, Vdos, Vcuat)
#    pts.densityplot(R, TH, f, r"$r$",r"$\theta$",r"$V(r,\theta)$",
#                    '', name='%s/%d/pot_rth.png'%(dirshoot,i), aspect='1/1')
##    
#    f = density_sph_coords(R, TH, uno, dos)
#    pts.densityplot(R, TH, f, r"$r$",r"$\theta$",r"$\rho(r,\theta)$",
#                    '', name='%s/%d/dens_rth.png'%(dirshoot,i), aspect='1/1')

if __name__ == '__main__':  
    for i in DM.nsols:  
        main(i, dirshoot)    
#    pts.multiplot_colorbar(erres, V4s, labelsn, r'$\hat r$', r'$V_{40}$', 
#                     '', '%s/V40.png'%dirtodos, r'$N_T$', ticks =labelsn)
#
    pts.multiplot_colorbar(rnn, vint, labelsn, r'$\hat r$', r'$v/c$',
                           '', '%s/vcirc2.png'%dirshoot, r'$N_T$', 
                           ticks =labelsn, ylim=(0,2.3))
#    pts.multiplot_colorbar(rnn, dens1, labelsn, r'$\hat r$', r'$\rho_{100}$',
#                           '', '%s/dens1.png'%dirshoot, r'$N_T$', ticks =labelsn)    
#    pts.multiplot_colorbar(rnn, dens2, labelsn, r'$\hat r$', r'$\rho_{210}$',
#                           '', '%s/dens2.png'%dirshoot, r'$N_T$', ticks =labelsn)
#    pts.multiplot_colorbar(rnn, V0i, labelsn, r'$\hat r$', r'$V_{00}$',
#                           '', '%s/V00_1.png'%dirshoot, r'$N_T$', ticks =labelsn)    
#    pts.multiplot_colorbar(rnn, V2i, labelsn, r'$\hat r$', r'$V_{20}$',
#                           '', '%s/V20_1.png'%dirshoot, r'$N_T$', ticks =labelsn)
    
    DM.plot_family()