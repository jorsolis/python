#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from fipy import *
import pylab
import numpy as np
import csv
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
''')#.extrude(extrudeFunc= lambda r: 1.1 * r)

#x, y = mesh.cellCenters()
#variables
phi = CellVariable(name=r'$\Phi$', mesh=mesh, hasOld=True)
psi = CellVariable(name=r'$\psi$', mesh=mesh, hasOld=True)
#The hasOld flag causes FiPy to store the variable value from the previous time step, which is necessary for solving equations with nonlinear coefficients or for coupling between PDEs

#poisson equation & KG equation
poiEq = (DiffusionTerm() == kappa * mu**2 * phi * phi)
kgeq = (DiffusionTerm() == ImplicitSourceTerm(2 * mu**2 * psi))
#constraints
phi.constrain(1., where=mesh.facesLeft)
phi.constrain(0., where=mesh.facesRight)
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
#
#viewer = Viewer(vars=(psi,), datamin=min(psi), datamax=max(psi))
#viewer2 = Viewer(vars=(phi,), datamin=-10., datamax=10.)
#
#viewer.plot("psi.png")
#viewer2.plot("phi.png")
TSVViewer(vars=(phi,)).plot(filename="phi_sch_poi.tsv")
TSVViewer(vars=(psi, )).plot(filename="psi_sch_poi.tsv")
#TSVViewer(vars=(phi.grad)).plot(filename="phi_sch_poi_grad.tsv")
#TSVViewer(vars=(psi.grad)).plot(filename="psi_sch_poi_grad.tsv")
