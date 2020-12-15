#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas

"""
import numpy as np
import resolvedor as resc 
#####   Constantes    ############################################
mu = 15.6378 # en pc
c = 0.3  # pc/año
Mmw = 0.005  #pc
Mhal= 0.05
############ Newtonian potential  #############################################
#M = 0.
#d = 4943.
#f = 1096.
#b_bar= mu*Mmw
#b = b_bar
####################################################################
#####       condiciones iniciales    ########################################
tf = 6.*1e6    #años
ri = 25.5*1e3        # parcecs
########### Parametros fit integral green
M = 3534.
b = 17.38
d = -138.5
f = 256.
########       condiciones iniciales    ######################################
ncor = 1
y0 = [1e-3, ri*mu,1e-3,ri*mu, 0.,0.]
#  u = mu c t 
u0 = 1000.
uf = c*mu*tf
du = uf/1000
u = np.arange(u0, uf, du)
max_steps = 65

#########################################################################
labcond = r"$v_x(0)=$ %f $c$, $x(0)=$ %s Kpc, $v_y(0)=$ %f $c\mu$, $y(0)=$ %s Kpc \
$M=$%s,$b=$%s,$d=$%s,$f=$%s, $t_f=$ %s M years"
cond = (y0[0], ri/1000, y0[2], ri/1000, M, b, d, f, tf/1e6)
##########################################################################
metodo= 'LSODA'
#
resc.solver_cart(u0,uf,y0,ncor,metodo,u,M,b,d,f,'',#labcond % cond,
            'mim','cart/mim/paramt') #method= 'RK45', 'Radau' or 'LSODA'         