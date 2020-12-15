#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:15:23 2020

@author: jordi
"""
import numpy as np
import plots_jordi as pts
from def_potenciales import multi_SFDM, SFDM, multistate, integralesferica, integral
pts.jordi_style()

m = 1

if m == 0:
    Ct = 2./np.sqrt(5)
    state = 210
elif m == 1:
    Ct = -1./np.sqrt(5)
    state = 211
    
DM = multistate(state)
dirshoot = DM.dirshoot

def r95(r, N, tol = 0.001):
    Nxmax = np.amax(N)
    N95 = 0.95*Nxmax
    print('Nmax=', np.amax(N), 'N95=', N95)
    try:
        index = np.where((N < N95 + tol) & (N > N95 -tol))[0][0]
        r95 = r[index] 
        print('r95=', r95)
        print('fi(r95)=', N[index])
        print('psi(r95)=', r[index])
    
    except IndexError:
        print('subir tolerancia')
    try:
        return Nxmax, N95, r95
    except UnboundLocalError:
        print('subir tolerancia')
        

def potencial_sph_coords(r, th, V00, V20):
    return V00 + r**2*V20*np.sqrt(5)*(-1. + 3.*np.cos(th)**2)/2.

def density_sph_coords(r, th, dens100, dens210, m):
    if m == 0:
        return dens100 + r**2*dens210*3.*np.cos(th)**2
    elif m == 1:
        return dens100 + r**2*dens210*3.*np.sin(th)**2/2


rlam = 1
mu = 1

labelsn = []
vint = []
rnn = []
dens1 = []; dens2 = []; V0i = []; V2i =[]
R95=[]

rr = []
UNO = []
DOS = []

def main(i, dirshoot):    
    r = SFDM(state, i).x
    V00, P0, V20, P2, Psi100, F1, Psi210, F2, E10, E20 = SFDM(state, i).funcs
    E1 = E10[-1]
    E2 = E20[-1]
    dF2 = -4.*F2/r + 2.*(V00 + Ct*r**2*V20 - E2)*Psi210
    dF1 = 2.*(V00 - E1)*r**2*Psi100  
    N1 = integralesferica(Psi100**2, r, r[-1])
    N2 = integralesferica(r**2*Psi210**2, r, r[-1])
    Et = (E1*N1 + E2*N2)/(N2+N1)    
    W210 = 4.*np.pi*(integralesferica(r**2*Psi210**2*V00, r, r[-1]) + Ct*integralesferica(r**2*Psi210**2*r**2*V20, r, r[-1]))
    K210 = -2.*np.pi*(4.*integralesferica(r*Psi210*F2, r, r[-1]) + integralesferica(r**2*dF2*Psi210, r, r[-1]))
    W100 = 4.*np.pi*integralesferica(Psi100**2*V00, r, r[-1])
    K100 = -2.*np.pi*integral(Psi100*dF1, r, r[-1])
    print('%.1f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t '%(P0[-1], E1, E2, Et, V00[0], V20[0], N2/N1, Psi100[0]/Psi210[0]))
    
#    Nmax, N95, r95v = r95(r, P0, tol = 0.005)
#    Nmax, N95, r95v = r95(r, P0, tol = 0.05)
    
#    UNO.append(P0)
#    DOS.append(np.sqrt(5)*r**3*(r*P2 + 2.*V20)/2.)
#    rr.append(r)
#    R95.append(r95v)
    
    labelsn.append(P0[-1]) 
    rn = np.linspace(0.01, 10, 100)    
    rnn.append(rn)
    SDM = multi_SFDM(state, rn, rlam, mu, i)
    vint.append(SDM.circ_vel())

    th = np.linspace(0, np.pi, 101)
    R, TH = np.meshgrid(rn, th)   
    Vcero, Vdos = SDM.potential()
    V0i.append(Vcero)
    V2i.append(Vdos)
#    f = potencial_sph_coords(R, TH, Vcero, Vdos)
#    pts.densityplot(R, TH, f, r"$r$",r"$\theta$",r"$V(r,\theta)$",
#                    '', name='%s/%d/pot_1.png'%(dirshoot,i), aspect='1/1')
    uno, dos = SDM.density()
    dens1.append(uno)
    dens2.append(dos)    
#    f = density_sph_coords(R, TH, uno, dos, m)
#    pts.densityplot(R, TH, f, r"$r$",r"$\theta$",r"$\rho(r,\theta)$",
#                    '', name='%s/%d/densrth.png'%(dirshoot,i), aspect='1/1')

if __name__ == '__main__': 
    DM.calc_cir_vel()    #####   para guardar los archivos de vel de toda la fam
    print('$N_T$ \t $E_{100}$ \t $E_{21%d}$ \t $E_T$ \t $V_{00}(0)$ \t $V_{20}(0)$ \t $N_{21%d}/N_{100}$ \t $\psi_{100}/\psi_{21%d}$'%(m, m, m))
    for i in DM.nsols: 
        main(i, dirshoot)
#    pts.multiplot_colorbar(rr, UNO, labelsn, r'$\hat r$', r'$P_0$',
#                           '', '%s/P0.png'%dirshoot, r'$N_T$', 
#                           ticks =labelsn)#, ylim=(0,0.25))      
#    pts.multiplot_colorbar(rr, DOS, labelsn, r'$\hat r$', r'$Cr^3(rP_2 + 2V_{20})$',
#                           '', '%s/ot.png'%dirshoot, r'$N_T$', 
#                           ticks =labelsn) 
    
    pts.multiplot_colorbar(rnn, vint, labelsn, r'$\hat r$', r'$v_h/c$',
                           '', '%s/vcirc2.png'%dirshoot, r'$N_T$', 
                           ticks =labelsn, ylim=(0,2.3))
#    pts.multiplot_colorbar(rnn, dens1, labelsn, r'$\hat r$', r'$\rho_{100}$',
#                           '', '%s/dens1.png'%dirshoot, r'$N_T$', ticks =labelsn)    
#    pts.multiplot_colorbar(rnn, dens2, labelsn, r'$\hat r$', r'$\rho_{21%d}$'%m,
#                           '', '%s/dens2.png'%dirshoot, r'$N_T$', ticks =labelsn)
#    pts.multiplot_colorbar(rnn, V0i, labelsn, r'$\hat r$', r'$V_{00}$',
#                           '', '%s/V00_1.png'%dirshoot, r'$N_T$', ticks =labelsn)    
#    pts.multiplot_colorbar(rnn, V2i, labelsn, r'$\hat r$', r'$V_{20}$',
#                           '', '%s/V20_1.png'%dirshoot, r'$N_T$', ticks =labelsn)
#    DM.plot_family()