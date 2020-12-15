#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:49:27 2018

@author: jordis
"""
import numpy as np
from mayavi.mlab import *

n_mer, n_long = 6, 11
dphi = np.pi / 1000.0
phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
mu = phi * n_mer
x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
z = np.sin(n_long * mu / n_mer) * 0.5

print( mu[0], x[0], y[0], z[0])

#plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='Spectral')
#axes(nb_labels=6)
#show()

x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]

scalars = x * x * 0.5 + y * y + z * z * 2.0

obj = volume_slice(scalars, plane_orientation='x_axes')
axes(nb_labels=6)
show()
#
#def f(x, y):
#    return np.sin(x + y) + np.sin(2 * x - y) + np.cos(3 * x + 4 * y)
#
#x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
#print( 'x shape=', x.shape,'f shape=', f(x,y).shape)
#s = surf(x, y, f)
##cs = contour_surf(x, y, f)
#axes(nb_labels=6)
#show()