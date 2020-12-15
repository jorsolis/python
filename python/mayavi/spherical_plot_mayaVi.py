#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:15:32 2018

@author: jordis
"""

from numpy import pi,sin,cos,meshgrid,linspace,array,mgrid
import scipy.special as spe 
from mayavi import mlab


th, phi = meshgrid(linspace(0,   pi,  91),
                   linspace(0, 2*pi, 181))
xyz = array([sin(th)*sin(phi),sin(th)*cos(phi),cos(th)]) 
l=2
m=0
#
Y_lm = spe.sph_harm(m,l, phi, th)
r = abs(Y_lm.real)*xyz
    
mlab.figure(size=(700,830))
mlab.mesh(r[0], r[1], r[2], scalars=Y_lm.real, 
          colormap="cool")
mlab.view(azimuth=45, elevation=100, distance=2.4, 
          roll=-50)
mlab.axes(nb_labels=4)
mlab.title("armonicos esfericos",size=0.5)
mlab.savefig("Y_%i_%i.jpg" % (l,m))
mlab.show()

dphi, dtheta = pi/250.0, pi/250.0
print( dphi*1.5, dphi)
[phi,theta] = mgrid[0:pi:dtheta,0:2*pi:dphi]
r= abs(spe.sph_harm(0,3, phi, theta))
x = r*sin(theta)*cos(phi)
y = r*sin(theta)*sin(phi)
z = r*cos(theta)

# View it.
mlab.figure
s = mlab.mesh(x, y, z)
mlab.axes(nb_labels=4)
mlab.savefig("Y_%i_%i_2.jpg" % (l,m))
mlab.show()

