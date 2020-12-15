#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas

"""
import numpy as np
import resolvedor as res
from resolvedor import my_range
#####   Constantes    ############################################
mu = 15.6378 # en pc
c = 0.3  # pc/año
Mmw = 0.005  #pc
Mhal= 0.05
####################################################################
#####       condiciones iniciales    ########################################
tf = 500.*1e6    #años
ri = 60.*1e3        # parcecs
###### fit PDE  #####################################
M = 3558.
b = 6555.
d = -254.6
f = 3.989
b_bar= 0.#mu*Mmw
########### Parametros fit integral green
#b_bar= 0.#mu*Mmw
#M = 3534.
#b = 17.38
#d = -138.5
#f = 256.
######       condiciones iniciales    ######################################
y0_0 = 0.       #r'(0)/c
y1_0 = ri*mu    #x(0)= \mu r(0)
ex = -9
y2_0 = 1.*10**ex     #theta'(0)/(c \hat{\mu})
y3_0 = 4*np.pi/8  # theta(0)
#################################################################### 
ncor = 1
L = y1_0*np.sqrt(Mmw+Mhal)/np.sqrt(ri)
#L = 10*L
#print "L=",L
y0 = [y0_0, y1_0, y2_0, y3_0,0.]
#  u = mu c t 
u0 = 0.
uf = c*mu*tf
du = uf/1000
u = np.arange(u0, uf, du)
max_steps = 1050
vels = {1.*10**-9: '$10^{-9}$'}
pies = {1*np.pi/8 : r'$\pi/8$', 2*np.pi/8 : r'$\pi/4$',3*np.pi/8 : r'$3\pi/8$',
        4*np.pi/8 : r'$\pi/2$', 5*np.pi/8 : r'$5\pi/8$',6*np.pi/8 : r'$3\pi/4$',
        7*np.pi/8 : r'$7\pi/8$', 8*np.pi/8 : r'$\pi$'}

###########################################################################
labcond = r"$L=$ %f, $v_r(0)=$ %d $c$, $r(0)=$ %s Kpc, $v_\theta(0)=$ %s $c\mu$, $ \theta(0)=$ %s \
$M=$%s,$b=$%s,$d=$%s,$f=$%s,$b_{bar}=$%s, $t_f=$ %s M years"
cond = (L, y0[0], ri/1000, y0[2], y0[3], M, b, d, f, b_bar, tf/1e6)
ncor = 8  
for ri in my_range(300.*1e3,500.*1e3,100.*1e3):
    print ncor
    y1_0 = ri*mu    #x(0)= \mu r(0)
    L = ri*mu*np.sqrt(Mmw)/np.sqrt(ri)
    L = 100.*L 
    y0 = [y0_0, y1_0, y2_0, y3_0,0.]
    cond = (L, y0[0], ri/1000, vels[y0[2]], pies[y0[3]], M, b, d, f,
            b_bar, tf/1e6)
    ########################################################################
    res.solver(u0,uf,y0,ncor,metodo,u,L,M,b,d,f,b_bar,labcond % cond,
               'mim','esf/mim/paramt')
   res.solver2(u0, uf, y0,ncor, metodo ,max_steps, L, M, b, d, f,b_bar,
               labcond % cond,'mim_sol2','esf/mim/paramt/green')
   res.solver3(u0,uf,du,y0,ncor, L, M, a, b, d, f, b_bar,labcond % cond,
           'mim_ode','esf/mim/paramt/green')
    ncor += 1
#############################################################################
#########################       NEWTON    ###################################
#############################################################################
#########       condiciones iniciales
#tf = 15000.*1e6    #años
#ri = 25500.        # parcecs
#M = 0.
#a= 0.
#b = mu*Mmw
#d = 0.
#f = 0.
#b_bar = 0.
##
#y0_0 = 0.       #r'(0)/c
#y1_0 = ri*mu    #x(0)= \mu r(0)
#ex = -9
#y2_0 = 10**ex     #theta'(0)/(c \hat{\mu})
#y3_0 = np.pi/2  # theta(0)   
#ncor = 9
#L = y1_0*np.sqrt(Mmw)/np.sqrt(ri)
#y0 = [y0_0, y1_0, y2_0, y3_0,0.]
#labcond = r"$L=$ %f, $v_r(0)=$ %d, $r(0)=$ %d pc, $v_\theta(0)=1e($%f) $c\mu$, $ \theta(0)=$ %f"
#cond = (L, y0_0, ri, ex, y3_0)
###  u = mu c t 
#u0 = 0.
#uf = c*mu*tf
#du = uf/1000
#u = np.arange(u0, uf, du)
#max_steps = 65
##
##
#print "tf=", tf/1e6, "M años"
#
#res.solver(u0,uf,y0,ncor,'LSODA',u,L,M,a,b,d,f,b_bar,labcond % cond,
#           'sol','esf/new/paramt') #method= 'RK45', 'Radau' or 'LSODA'
##solver2(u0, uf, y0,ncor, 'LSODA' ,max_steps, L, M, a, b, d, f,b_bar,
##            labcond % cond,'sol','esf/new/paramt')
#res.solver3(u0,uf,du,y0,ncor, L, M, a, b, d, f, b_bar,labcond % cond,'sol',
#        'esf/new/paramt')
