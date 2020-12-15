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
dx = dy = 0.0025
nx = ny = 500
mesh= Grid2D(dx=dx, dy=dy,nx=nx, ny=ny)
#
#fixed timesteps
dt = 5e-4
#
#
#funciones incognitas
phase = CellVariable(name=r'$\phi$', mesh=mesh, hasOld=True)
dT = CellVariable(name=r'$\Delta T$', mesh=mesh, hasOld=True)
#The hasOld flag causes FiPy to store the variable value from the previous time step, which is necessary for solving equations with nonlinear coefficients or for coupling between PDEs
#
#
#heat equation
DT = 2.25 
heatEq = (TransientTerm() == DiffusionTerm(DT) + (phase - phase.getOld()) / dt)
alpha = 0.015
c = 0.02
N = 6.
theta = np.pi / 8.
psi = (theta + np.arctan2(phase.getFaceGrad()[1], phase.getFaceGrad()[0]))
Phi = np.tan(N * psi/2)
beta = (1. - Phi**2) / (1. + Phi**2)
DbetaDpsi = -N * 2 * Phi / (1 + Phi**2)
D = (alpha**2 * (1. + c * beta) * ((1. + c * beta) * (( 1, 0),( 0, 1)) + (c * DbetaDpsi) * (( 0,-1),( 1, 0))))
#fase equation
tau = 3e-4
kappa1 = 0.9
kappa2 = 20.
phaseEq = (TransientTerm(tau)== DiffusionTerm(D)+ ImplicitSourceTerm((phase - 0.5- kappa1 / np.pi * np.arctan(kappa2 * dT))* (1 - phase)))
#
#
#seed
R = dx * 5.
C = (nx * dx/2, ny * dy / 2)
x, y = mesh.cellCenters()
phase.setValue(1., where=((x-C[0])**2 + (y-C[1])**2) < R**2)
dT.setValue(-0.5)

#class DendriteViewer(
#        Matplotlib2DGridViewer):
#        def __init__(self, phase, dT, title=None, limits={}, **kwlimits):
#            self.phase = phase
#            self.contour = None
#            Matplotlib2DGridViewer.__init__(
#                self, vars=(dT,), title=title,
#                cmap=pylab.cm.hot,
#                limits=limits, **kwlimits)
#
#def _plot(self):
#    Matplotlib2DGridViewer._plot(self)
#    if self.contour is not None:
#        cc = self.contour.collections
#        for c in cc:
#            c.remove()
#    mesh = self.phase.getMesh()
#    shape = mesh.getShape()
#    x, y = mesh.getCellCenters()
#    z = self.phase.getValue()
#    x, y, z = [a.reshape(shape, order="FORTRAN") for a in (x, y, z)]
#    self.contour = pylab.contour(x, y, z, (0.5,))
#
#viewer = DendriteViewer(phase=phase, dT=dT, 
#                        title=r"%s & %s" % (phase.name, dT.name), 
#                        datamin=-0.1, datamax=0.05)

viewer2 = MatplotlibViewer(vars=phase, 
                           limits={'ymin': 0.1, 'ymax': 0.9}, 
                           datamin=-0.1, datamax=0.05,
                           title="Phase",
#                           colorbar=True,
                           cmap=pylab.cm.hot)
viewer3 = MatplotlibViewer(vars=dT, 
                           limits={'ymin': 0.1, 'ymax': 0.9}, 
                           datamin=-0.1, datamax=0.05,
                           title="dT",
                           colorbar=True)

for i in range(10000):
    phase.updateOld()
    dT.updateOld()
    phaseEq.solve(phase, dt=dt)
    heatEq.solve(dT, dt=dt)
    if i % 10 == 0:
#        viewer.plot()
         viewer2.plot() 
         viewer3.plot() 
        