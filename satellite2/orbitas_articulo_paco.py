#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas
"""
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
from resolvedor import solver_artpaco
#####   Constantes    ############################################
mu = 15.6378 # en 1/pc
c = 0.3  # pc/year
Mmw = 0.005  #pc
Mhal= 0.05
mu = 1.
###########     Diccionarios     #######################################
vels = {1.*10**-10: '$10^{-10}$', 1.*10**-9: '$10^{-9}$', 1.*10**-8: '$10^{-8}$'}
pies = {1*np.pi/8 : r'$\pi/8$', 2*np.pi/8 : r'$\pi/4$',3*np.pi/8 : r'$3\pi/8$', 4*np.pi/8 : r'$\pi/2$', 5*np.pi/8 : r'$5\pi/8$',6*np.pi/8 : r'$3\pi/4$', 7*np.pi/8 : r'$7\pi/8$', 8*np.pi/8 : r'$\pi$'}
#####       condiciones iniciales    ########################################
tf = 10.#50.#00.*1e3#    #kyear
ri = 1.#250. #  # kpc
######       condiciones iniciales    ######################################
y0_0 = 0.       #r'(0)/c
y1_0 = ri*mu    #x(0)= \mu r(0)
ex = -9
y2_0 = 1.*10**ex     #theta'(0)/(c \hat{\mu})
y3_0 = 3*np.pi/8  # theta(0)
#################################################################### 
u0 = 0.   #  u = mu c t 
uf = c*mu*tf
du = uf/100# /1000
#########################################################################
#R = np.float64(1e2)
#n = np.float64(1e4)
R = 1.
n= 1.
#########################################################################
labco = r"$r(0)=$ %s kpc, $ \theta(0)=$ %s, $v_r(0)=$ %d $c$,  $v_\theta(0)=$ %s $c\mu$ \
$L\mu=$ %f $\hbar$, $l=1$, $t_f=$ %s M years"
ncor = 1
print(y1_0)
L = y1_0*np.sqrt(Mmw+Mhal)/np.sqrt(ri)
L=1.
print( L)
y0 = [y0_0, y1_0, y2_0, y3_0,0.]
co = (ri, pies[y0[3]], y0[0], vels[y0[2]],L, tf/1e3)
solver_artpaco(u0,uf,du,y0,ncor,'LSODA',L,mu,R,n,labco % co,
               'green','/home/jordi/satellite/art_paco',plot3d=True)