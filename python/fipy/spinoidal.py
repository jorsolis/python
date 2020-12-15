#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 22:04:18 2018

@author: jordis
"""
from fipy import *
import pylab
import numpy as np
#grid
mesh = Grid2D(nx=1000, ny=1000, dx=0.25, dy=0.25)
phi = CellVariable(name=r'$\phi$', mesh=mesh)
phi.setValue(GaussianNoiseVariable(mesh=mesh, mean=0.5, variance=0.01))
#viewer = Viewer(vars=(phi,), datamin=0., datamax=1.)
PHI = phi.getArithmeticFaceValue()
D = a = eps = 1.
eq = (TransientTerm() == DiffusionTerm(coeff= D * a**2 * (1 - 6 * PHI * (1-PHI))) - DiffusionTerm(coeff=(D, eps**2)))
dexp = -5
elapsed = 0.
while elapsed < 100.:#1000
#    dt = 10
    dt = min(100, np.exp(dexp))
    elapsed += dt
    dexp += 0.1 #0.01
    print(elapsed)
    print(dexp)
    eq.solve(phi, dt=dt)
    viewer.plot()
