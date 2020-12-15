#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ecuaciones diferenciales

"""
import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import rcParams

rcParams['figure.figsize'] = (10, 6)
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
#rcParams['text.usetex'] = True  # or plt.rc('text', usetex=True)
#rcParams['font.family'] = 'serif' #or plt.rc('font', family='serif')
#constantes
G = 1.
kappa = 8*np.pi*G
gama = 1.
q = 1.
m = 1e-21
omega0 = 0.1*m
omega = (kappa/6)**(1/2)*omega0
M = (kappa/6)**(1/2)*m
lambda0 = 0.01
Lambda = kappa*lambda0/12
mu = 1.
print m
#condiciones iniciales [-1,.5,1,-1]
y0_0 = 1.  #H(0)
y1_0 = 1. #n(0)
y2_0 = 1. #n'(0)
y3_0 = 1 # S'(0)
y4_0 = 1.  #A(0)
y5_0 = 1. #B(0)
y6_0 = 1. #b(0)
y0 = [y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0]
print "condiciones iniciales", y0

t0 = -13.81
tf = 0.
dt = 0.01

def PIX(y):
    "Pi"
    return (gama*y[6]**2+y[2]**2/(2*y[1])+2*y[1]*(y[3]-omega/y[0]+y[4])**2), 

def func(y, t):
    return [-3*(gama*y[6]**2+y[2]**2/(2*y[1])+2*y[1]*(y[3]-omega/y[0]+y[4])**2)*y[0]/2,
            (6/kappa)**(1/2)*y[2],
            3*y[2]*((gama*y[6]**2+y[2]**2/(2*y[1])+2*y[1]*(y[3]-omega/y[0]+y[4])**2)/2 -1) + (6/kappa)**(1/2)*y[2]**2 +
            2*(6/kappa)**(1/2)*(M**2*(1 + lambda0*y[1]/m**2)/y[0]**2 + (y[3]-omega/y[0] + y[4])**2),
            3*y[3]*((gama*y[6]**2+y[2]**2/(2*y[1])+2*y[1]*(y[3]-omega/y[0]+y[4])**2)/2 -1) + 3*omega/y[0] - (6/kappa)**(1/2)*y[2]*(y[3]-omega/y[0] + y[4])/y[1],
            3*(gama*y[6]**2+y[2]**2/(2*y[1])+2*y[1]*(y[3]-omega/y[0]+y[4])**2)*y[4]/2 + 12*q*y[5]/(kappa*y[0]),
            -3*(gama*y[6]**2+y[2]**2/(2*y[1])+2*y[1]*(y[3]-omega/y[0]+y[4])**2)*y[5]/2 - mu*M*y[1]*(y[3]-omega/y[0] + y[4])/(m*y[0]),
            3*((gama*y[6]**2+y[2]**2/(2*y[1])+2*y[1]*(y[3]-omega/y[0]+y[4])**2)-gama)*y[6]/2]

t = np.arange(t0, tf, dt)
y = spi.odeint(func, y0, t)
#
def coordsplot(x1,x2,t,x1label,x2label,title,nom_archivo):
    "x1, x2 y t son arrays. x1label,x2label,title son strings"
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')    
    plt.plot(t, x1, 'b-', label=x1label)
    plt.plot(t, x2, 'r--', label=x2label)
    if np.amax(x1) > np.amax(x2):
        ylim_max = np.amax(x1)
    else:
        ylim_max = np.amax(x2)
    #    
    if np.amin(x1) > np.amin(x2):
        ylim_min = np.amin(x2)
    else:
        ylim_min = np.amin(x1)
    plt.ylim(ylim_min-0.3, ylim_max+0.3)
    plt.grid()
    plt.xlabel(r'$t$')#,fontsize=16)
    plt.xticks(np.arange(0, tf+tf/10 , step=tf/5),fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    plt.title(title, fontsize=20, color='black')
    plt.legend(loc='upper right')
    plt.savefig(nom_archivo)
    plt.show()

def coordplot(x1,t,x1label,title,nom_archivo):
    "x1 y t son arrays. x1label,title son strings"
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')    
    plt.plot(t, x1, 'b-', label=x1label)
    ylim_max = np.amax(x1)
    ylim_min = np.amin(x1)
    plt.ylim(ylim_min-0.3, ylim_max+0.3)
    plt.grid()
    plt.xlabel(r'$N$')#,fontsize=16)
    plt.xticks(np.arange(0, tf+tf/10 , step=tf/5),fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    plt.title(title, fontsize=20, color='black')
    plt.legend(loc='upper right')
    plt.savefig(nom_archivo)
    plt.show()

coordplot(y[:, 1],t,r'$n(N)$',r"$4$" ,"prueba")
#
#coordsplot(y[:, 1],y[:, 3],t,r'$r(t)$',r'$\theta(t)$',
#           r"$v_r(0)=0$, $r(0)=1$, $v_\theta(0)=1$, $\theta(0)=\pi/4$",
#           "orbita_axisimetrica")
#
def parametricplot(x,y,xlabel,ylabel,title,nom_archivo):
    "x e y son arrays. xlabel,ylabel,title son strings"
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(x, y, 'b-')
    plt.ylim(np.amin(y)-0.3, np.amax(y)+0.3)
    plt.grid()
    plt.xlabel(xlabel)#,fontsize=16)
    plt.ylabel(ylabel)#,fontsize=16)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.title(title,fontsize=20)
    plt.savefig(nom_archivo)
    plt.show()

#parametricplot(y[:, 1], y[:, 3], r'$r(t)$', r'$\theta(t)$', r'$\theta$ vs $r$', "orbita_axisimetrica_rvstheta")

#parametricplot(y[:, 1], y[:, 0], r'$r(t)$', r'$\dot{r}(t)$', r'Phase space',"Phase_space_r")
#parametricplot(y[:, 3], y[:, 2], r'$\theta(t)$', r'$\dot{\theta}(t)$', r'Phase space',"Phase_space_theta")
