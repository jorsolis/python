#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 20:14:19 2018

@author: jordis
"""
import csv
import matplotlib.pyplot as plt
from mayavi import mlab
import scipy.special as spe

x = []
y = []
z = []
p = []
with open("psi_poisson.tsv") as tsvfile:
    lines = csv.reader(tsvfile, delimiter='\t')
    for row in lines:
        x.append(float(row[0]))
        y.append(float(row[1]))
        z.append(float(row[2]))
        p.append(float(row[3])) 
figure = mlab.figure('Psi',size=(700,830))
colormap = "copper", "Spectral"
pts = mlab.points3d(x, y, z, p, colormap="jet",
                    scale_mode='none', scale_factor=0.07,
                    resolution=10)
mlab.axes(nb_labels=8)
mlab.title("Psi",size=0.5)
mlab.savefig("psi_poisson.jpg")
mlab.show()

#ppl.scalar_field(x, y, z, p)


# Create the data.
#from numpy import pi, sin, cos, mgrid
#
#dphi, dtheta = pi/250.0, pi/250.0
#[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
#r = spe.legendre(1)(cos(theta))
#x = r*sin(phi)*cos(theta)
#y = r*cos(phi)
#z = r*sin(phi)*sin(theta)
#
## View it.
#s = mlab.mesh(x, y, z)
#mlab.show()
