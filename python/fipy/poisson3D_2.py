#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from fipy import *
import pylab
import numpy as np
import csv
import scipy.special as ss
import matplotlib.pyplot as plt
#constants
kappa = 8 * np.pi
mu = 10**(-21)
#
mesh = Gmsh3D('mesh/sphere_tet.msh')

X = mesh.cellCenters[0]
Y = mesh.cellCenters[1]
Z = mesh.cellCenters[2]

u= (X*X + Y*Y + Z*Z)**(1/2)

#print ss.spherical_jn(1,x)
#print ss.lpmv(1,3,x)

#variable
psi = CellVariable(name=r'$\psi$', mesh=mesh, hasOld=True)
#The hasOld flag causes FiPy to store the variable value from the previous time step, which is necessary for solving equations with nonlinear coefficients or for coupling between PDEs
phi = CellVariable(name="source", mesh=mesh)
phi.setValue(np.sin(u))
#print phi
#poisson equation
poiEq = (DiffusionTerm() == kappa * mu**2 * phi * phi)
#constraints
psi.constrain(-100., where= u<0.1)
psi.constrain(0., where= u>1)
psi.faceGrad.constrain([0.], where=mesh.facesTop)# estan bien???
psi.faceGrad.constrain([0.], where=mesh.facesBottom) # estan bien???

#
for i in range(10):
    psi.updateOld()
    poiEq.solve(psi)
#
#
viewer = Viewer(vars=(psi,), datamin=min(psi), datamax=max(psi))
#
viewer.plot("psi.png")
#TSVViewer(vars=(psi, )).plot(filename="psi_poisson.tsv")
#TSVViewer(vars=(psi.grad)).plot(filename="psi_sch_poi_grad.tsv")
#
#    Fuentes

styles=['b-','b--','r-','r--','g-','g--']
r = np.arange(0, 10.01, 0.01)
for i in range (0,6):
    plt.plot(r, ss.spherical_jn(i,r), dashes=[i, 1], label=r'$j_%i(r)$' % i)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'$r$',fontsize=16)
    plt.ylabel(r'$j_n(r)$',fontsize=16)
plt.title(r"SFDM radial function",
          fontsize=16, color='black')
plt.legend(loc='upper right')
plt.savefig("SFDM_radial_function")
plt.show()

theta = np.arange(0, np.pi, 0.01)
for i in range (0,6):
    plt.plot(theta, ss.lpmv(1,i,np.cos(theta)),styles[i], label=r'$P_%i(\cos \theta)$' % (i))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'$\theta$',fontsize=16)
    plt.ylabel(r'$P_n(\cos \theta)$',fontsize=16)
plt.title(r"SFDM angular function",
          fontsize=16, color='black')
plt.legend(loc='best')
plt.savefig("SFDM_angular_function")
plt.show()