#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from fipy import *
import pylab
import numpy as np
#grid
dx = 0.25 #dr
dy = np.pi/200 #dtheta
nx = 200 #nr
ny = 200 #ntheta
L = dx * nx
mesh= Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
#constants
kappa = 8 * np.pi
mu = 10**(-21)
#
x, y = mesh.cellCenters()
#variables
phi = CellVariable(name=r'$\Phi$', mesh=mesh, hasOld=True)
psi = CellVariable(name=r'$\psi$', mesh=mesh, hasOld=True)
#The hasOld flag causes FiPy to store the variable value from the previous time step, which is necessary for solving equations with nonlinear coefficients or for coupling between PDEs
#
#poisson equation & KG equation|
poiEq = (DiffusionTerm() + numerix.dot([[0], [1]], psi.grad) / (np.tan(y) *x**2) + (1-x**2) * numerix.dot([[0], [1]], numerix.dot([[0], [1]], psi.grad).grad) / x**2 + 2 * numerix.dot([[1], [0]], psi.grad) / x == kappa * mu**2 * phi * phi)
kgeq = (DiffusionTerm() + numerix.dot([[0], [1]], phi.grad) / (np.tan(y) *x**2) + (1-x**2) * numerix.dot([[0], [1]], numerix.dot([[0], [1]], phi.grad).grad) / x**2 + 2 * numerix.dot([[1], [0]], phi.grad) / x  == ImplicitSourceTerm(2 * mu**2 * psi))
#constraints
phi.constrain(1., where=mesh.facesLeft)
phi.constrain(0., where=mesh.facesBottom)
phi.faceGrad.constrain([0.], where=mesh.facesTop)
phi.faceGrad.constrain([0.], where=mesh.facesBottom) 
psi.constrain(-1000., where=mesh.facesLeft)
psi.constrain(0., where=mesh.facesRight)
psi.faceGrad.constrain([0.], where=mesh.facesTop)# estan bien???
psi.faceGrad.constrain([0.], where=mesh.facesBottom) # estan bien???
#
#
for i in range(10):
    psi.updateOld()
    phi.updateOld()
    poiEq.solve(psi)
    kgeq.solve(phi)
#
#print "Phi=" ,phi
print( "max(Phi)=", max(phi), "min(Phi)=", min(phi))
#print "psi=",psi
print( "max(psi)=", max(psi), "min(psi)=", min(psi))
#
viewer = Viewer(vars=(psi,), datamin=-10000., datamax=10000.)
viewer2 = Viewer(vars=(phi,), datamin=-10., datamax=10.)
#
viewer.plot("psi.png")
viewer2.plot("phi.png")
TSVViewer(vars=(phi,)).plot(filename="phi_sch_poi_cuad.tsv")
TSVViewer(vars=(psi, )).plot(filename="psi_sch_poi_cuad.tsv")
TSVViewer(vars=(phi.grad)).plot(filename="phi_sch_poi_cuad_grad.tsv")
TSVViewer(vars=(psi.grad)).plot(filename="psi_sch_poi_cuad_grad.tsv")