# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import numpy
import cmath as math
from fipy import *
from fipy import numerix
#
nx = 50
dx = 1. / float(nx)
#
mesh = Grid1D(nx=nx,dx=dx)
#
X = mesh.cellCenters[0]
#
phi = CellVariable(mesh=mesh, name="Solution")
#
vi = Viewer(vars=phi,datamin=0.0, datamax=1.0)
vi.plot()
#raw_input("Initialization ...Press <return>")
#
#
phi.constrain(1., mesh.facesLeft)
phi.constrain(0., mesh.facesRight)
phi_sq = CellVariable(mesh=mesh)
phi_sq.setValue( phi*phi )
# u = phi, du/dt + d^2u/dx^2 + |u|^2*u = 0
# du/dx : is a convection term with a unit scalar coefficient, i.e.
# du/dt : is a transient term that one can treat as before, i.e. 
#TransientTerm(var=u). one can add a second order derivative as 
#ExplicitDiffusionTerm(coeff=D)
eq = TransientTerm(coeff=1., var=phi) + ExponentialConvectionTerm(coeff=(1.,),
                   var=phi) + abs((phi_sq))*(phi) == 0.0
dt = 0.01
steps = 100
for step in range(steps):
    eq.sweep(dt=dt)
#    print(step)
    phi_sq.setValue( phi * phi )
    vi.plot()
# We then add a trial analytical solution to test for validity in the given PDE above, namely e^ix
phiAnalytical = CellVariable(name="Analytical value", mesh=mesh)
phiAnalytical.setValue(numerix.exp(1j*(X)))
vi = Viewer(vars=(phi, phiAnalytical))
vi.plot()
#raw_input("Press <return> ...")