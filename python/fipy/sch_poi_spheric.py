#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:36:30 2018

@author: jordi
"""
from fipy import *
import pylab
import numpy as np
#grid
dx = 0.025 #dr
nx = 1000 #nr
mesh= Grid1D(dx=dx, nx=nx)
#constants
kappa = 8 * np.pi
mu = 1
#
x = mesh.cellCenters()
#variables
u = CellVariable(name=r'$r\Phi$', mesh=mesh, hasOld=True)
psi = CellVariable(name=r'$\psi$', mesh=mesh, hasOld=True)
#The hasOld flag causes FiPy to store the variable value from the previous time step, which is necessary for solving equations with nonlinear coefficients or for coupling between PDEs
#
#poisson equation
poiEq = (DiffusionTerm() == kappa * mu**2 * u * u)
#KG equation|
kgeq = (DiffusionTerm()  == ImplicitSourceTerm(2 * mu**2 * psi))
u.constrain(1., where=mesh.facesLeft)
u.constrain(0., where=mesh.facesRight) 
psi.constrain(-1000., where=mesh.facesLeft)
psi.constrain(0., where=mesh.facesRight) 

for i in range(10):
    psi.updateOld()
    u.updateOld()
#    print(i)
    poiEq.solve(psi)
    kgeq.solve(u)

viewer = Viewer(vars=(psi,), datamin=-1000., datamax=1.)
viewer2 = Viewer(vars=(u,), datamin=0., datamax=1.)
#        
#viewer.plot("psisph.png")
#viewer2.plot("rphisph.png")