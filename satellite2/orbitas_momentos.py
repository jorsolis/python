#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas

"""
import scipy.integrate as spi
import numpy as np
import resolvedormomentos as rm
#####   Constantes    ############################################
mu = 15.6378 # en pc
c = 0.3  # pc/año
Mmw = 0.005  #pc
Mhal= 0.05
#####################################################################
####################################################################
#####       condiciones iniciales    ########################################
tf = 50.*1e6    #años
ri = 30.*1e3        # parcecs
###### SFDM Potential Parameters + Baryons  #####################################
#M = 11590.
#a = 0.2957
#b = 1378.
#d = 4943.
#f = 1096.
#b_bar= mu*Mmw
########### Parametros fit integral green
#M = 3567.
#a = 0.9806
#b = 15.75
#d = -0.09623
#f = 116.9
#b_bar= mu*Mmw
M = 0.
a = 0.
b = mu*Mmw
d = -0.09623
f = 116.9
b_bar= mu*Mmw
######       condiciones iniciales    ######################################
y0_0 = 0.       #r'(0)/c
y1_0 = ri*mu    #x(0)= \mu r(0)
ex = -9
y2_0 = 1.*10**ex     #theta'(0)/(c \hat{\mu})
y3_0 = 4*np.pi/8  # theta(0)
#################################################################### 
ncor = 20
L = y1_0*np.sqrt(Mmw+Mhal)/np.sqrt(ri)
L = 100*L
#print "L=",L
y0 = [y0_0, y1_0, y2_0, y3_0,L*mu/c,0.]
#  u = mu c t 
u0 = 0.
uf = c*mu*tf
print uf
du = uf/1000
u = np.arange(u0, uf, du)
max_steps = 65

#########################################################################
labcond = r"$L=$ %f, $v_r(0)=$ %f $c$, $r(0)=$ %s Kpc, $v_\theta(0)=$ %f $c\mu$, $ \theta(0)=$ %f \
$M=$%s, $a=$%f,$b=$%s,$d=$%s,$f=$%s,$b_{bar}=$%s, $t_f=$ %s M years"
cond = (L, y0[0], ri/1000, y0[2], y0[3], M, a, b, d, f, b_bar, tf/1e6)
##########################################################################
metodo= 'LSODA'

rm.solver(u0,uf,y0,ncor,metodo,u,L,M,a,b,d,f,b_bar,labcond % cond,
       'mim','esf/mim/paramt/green')
rm.solver2(u0, uf, y0,ncor, metodo,max_steps, L, M, a, b, d, f, b_bar,
        labcond % cond,'mim','esf/mim/paramt/green')  

