#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:25:44 2018

@author: jordis
"""
import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import csv
from matplotlib import rcParams
import scipy.special as spe
rcParams['text.usetex'] = True  # or plt.rc('text', usetex=True)
rcParams['font.family'] = 'serif' #or plt.rc('font', family='serif')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(13.3,10))
#
#
#        PLOT SPHERICAL HARM
#def f(x, y):
#    return np.sin(np.sqrt(x ** 2 + y ** 2))
#def f(x, y):
#    return np.cos(x)*np.cos(y)
nu = 1
kmu = 0.458263#k/mu
def f(x, y):
    return spe.jv(nu,kmu*x)**2 * spe.legendre(1)(np.cos(y))**2
#
x = np.linspace(0, 6, 30)
y = np.linspace(0, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis',
                    edgecolor='none')
ax.set_xlabel(r'$x$')
plt.xticks(fontsize=16, rotation=0)
ax.set_ylabel(r'$y$')
plt.yticks(fontsize=16, rotation=0)
ax.set_zlabel(r'$z$')
ax.view_init(60, 45)# elevation of 60 degrees (that is, 60 degrees above the x-y plane) and an azimuth of 35 degrees (that is, rotated 35 degrees counter-clockwise about the z-axis)
plt.show()
#
#        
#r = np.linspace(0, 6, 100)
#print 'r shape', r.shape
#theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
#r, theta = np.meshgrid(r, theta)
#print 'r shape', r.shape
#X = r * np.sin(theta)
#Y = r * np.cos(theta)
#Z = f(X, Y)
#print 'Y shape', Y.shape
#print 'Z shape', Z.shape
#axe = plt.axes(projection='3d')
#axe.set_xlabel(r'$x$')
#plt.xticks(fontsize=16, rotation=0)
#axe.set_ylabel(r'$y$')
#plt.yticks(fontsize=16, rotation=0)
#axe.set_zlabel(r'$z$')
#axe.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                cmap='viridis', edgecolor='none');
#axe.view_init(45, 35)# elevation of 45 degrees (that is, 45 degrees above the x-y plane) and an azimuth of 35 degrees (that is, rotated 35 degrees counter-clockwise about the z-axis)

#
#
#            SCATTER Plot
#        
#r = np.array([])
#theta = np.array([])
#z = np.array([])
#
#with open("coords.csv") as tsvfile:
#    lines = csv.reader(tsvfile, delimiter='\t')
#    for row in lines:
#        r= np.append(r,float(row[0]))
#        theta =np.append(theta,float(row[1]))
#with open("val.csv") as tsvfile:
#    lines = csv.reader(tsvfile, delimiter='\t')
#    for row in lines:
#        z=np.append(z,float(row[0]))
#print r.size, theta.size, z.size
#  
#fig = plt.figure(figsize=(13.3,10))
#ax = plt.axes(projection='3d')
#ax.scatter(r, theta, z, marker='_')
#plt.xticks(fontsize=16, rotation=0)
#ax.set_xlabel(r'$r$',fontsize=16)
#plt.yticks(np.arange(0, 5*np.pi/4, step=np.pi/4),(r'$0$', r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$'),fontsize=16, rotation=0)
#ax.set_ylabel(r'$\theta$',fontsize=16)
#ax.set_zlabel(r'$\psi$',fontsize=16)
#plt.show
