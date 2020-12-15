#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from fipy import *
import pylab
import numpy as np
import csv
import scipy.special as ss
import matplotlib.pyplot as plt
from mayavi.mlab import *
#constants
kappa = 8 * np.pi
mu = 10**(-21)
#
mesh = Gmsh3D('''
// Gmsh project created on Sun Sep 16 20:49:24 2018
rad = DefineNumber[ 10, Name "Parameters/rad" ];
ndens = DefineNumber[ 1, Name "Parameters/ndens" ];
Point(1) = {0, 0, 0, ndens};
Point(2) = {rad, 0, 0, ndens};
Point(3) = {-rad, 0, 0, ndens};
Point(4) = {0, rad, 0, ndens};
Point(5) = {0, -rad, 0, ndens};
Point(6) = {0, 0, rad, ndens};
Point(7) = {0, 0, -rad, ndens};
Circle(1) = {2, 1, 4};
Circle(2) = {4, 1, 3};
Circle(3) = {3, 1, 5};
Circle(4) = {5, 1, 2};
Circle(5) = {2, 1, 6};
Circle(6) = {6, 1, 3};
Circle(7) = {3, 1, 7};
Circle(8) = {7, 1, 2};
Circle(9) = {4, 1, 6};
Circle(10) = {6, 1, 5};
Circle(11) = {5, 1, 7};
Circle(12) = {7, 1, 4};
Line Loop(1) = {2, 7, 12};
Ruled Surface(1) = {1};
Line Loop(2) = {2, -6, -9};
Ruled Surface(2) = {2};
Line Loop(3) = {3, -10, 6};
Ruled Surface(3) = {3};
Line Loop(4) = {3, 11, -7};
Ruled Surface(4) = {4};
Line Loop(5) = {4, -8, -11};
Ruled Surface(5) = {5};
Line Loop(6) = {4, 5, 10};
Ruled Surface(6) = {6};
Line Loop(7) = {1, 9, -5};
Ruled Surface(7) = {7};
Line Loop(8) = {1, -12, 8};
Ruled Surface(8) = {8};
Surface Loop(1) = {8, 7, 2, 1, 4, 3, 6, 5};
Volume(1) = {1};
''')
h = mesh.cellDistanceVectors
X = mesh.cellCenters[0]
Y = mesh.cellCenters[1]
Z = mesh.cellCenters[2]

print h.shape, X.shape, Y.shape, Z.shape
u= (X*X + Y*Y + Z*Z)**(1/2)
#x=[0,0.1,0.2,0.3]
#print ss.spherical_jn(1,x)
#print ss.lpmv(1,3,x)

#variable
psi = CellVariable(name=r'$\psi$', mesh=mesh, hasOld=True)
#The hasOld flag causes FiPy to store the variable value from the previous time step, which is necessary for solving equations with nonlinear coefficients or for coupling between PDEs
phi = CellVariable(name="source", mesh=mesh)
phi.setValue(numerix.exp(1j*(X)))
print phi
#poisson equation
poiEq = (DiffusionTerm() == kappa * mu**2 * phi * phi)
#constraints
psi.constrain(-1000., where=mesh.facesLeft)
psi.constrain(0., where=mesh.facesRight)
psi.faceGrad.constrain([0.], where=mesh.facesTop)# estan bien???
psi.faceGrad.constrain([0.], where=mesh.facesBottom) # estan bien???


for i in range(10):
    psi.updateOld()
    poiEq.solve(psi)


viewer = Viewer(vars=(psi,), datamin=min(psi), datamax=max(psi))
#
#viewer.plot("psi.png")
TSVViewer(vars=(psi, )).plot(filename="psi_poisson.tsv")
TSVViewer(vars=(psi.grad)).plot(filename="psi_sch_poi_grad.tsv")


#    Fuentes

styles=['b-','b--','r-','r--','g-','g--']
r = np.arange(0, 10.01, 0.01)
fig = plt.figure(figsize=(8,6))
for i in range (0,6):
    plt.plot(r, ss.spherical_jn(i,r), dashes=[i, 1], label=r'$j_%i(r)$' % i)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$r$',fontsize=16)
plt.ylabel(r'$j_n(r)$',fontsize=16)
plt.title(r"SFDM radial function", fontsize=16, color='black')
plt.legend(loc='upper right')
plt.savefig("SFDM_radial_function")
plt.show()

theta = np.arange(0, np.pi, 0.01)
fig = plt.figure(figsize=(8,6))
for i in range (0,6):
    plt.plot(theta, ss.lpmv(1,i,np.cos(theta)),styles[i], label=r'$P_%i(\cos \theta)$' % (i))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$\theta$',fontsize=16)
plt.ylabel(r'$P_n(\cos \theta)$',fontsize=16)
plt.title(r"SFDM angular function", fontsize=16, color='black')
plt.legend(loc='best')
plt.savefig("SFDM_angular_function")
plt.show()

#r = np.arange(1., np.pi, 0.1) 
#theta = np.arange(1., np.pi, 0.1)
#def f(n, r, theta):
#    return  ss.spherical_jn(n,r)*ss.lpmv(1,n,np.cos(theta))
#
##print f(1,r,theta)
#s = surf(r, theta, f(2,r,theta))
##cs = contour_surf(x, y, f)
#axes(nb_labels=6)
#show()