#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 12:15:18 2018

@author: jordis
"""
from fipy import *
m = Grid1D(nx=100, Lx=1.)
#
v0 = CellVariable(mesh=m, hasOld=True, value=0.5, name='v_0')
v1 = CellVariable(mesh=m, hasOld=True, value=0.5, name='v_1')
v0.constrain(0, m.facesLeft)
v0.constrain(1, m.facesRight)
v1.constrain(1, m.facesLeft)
v1.constrain(0, m.facesRight)
vi = Viewer((v0, v1))
#
#
#metodo 1
#eq0 = TransientTerm() == DiffusionTerm(coeff=0.01) - v1.faceGrad.divergence
#eq1 = TransientTerm() == v0.faceGrad.divergence + DiffusionTerm(coeff=0.01)
#for t in range(100): 
#     v0.updateOld()
#     v1.updateOld()
#     res0 = res1 = 1e100
#     while max(res0, res1) > 0.1:
#         res0 = eq0.sweep(var=v0, dt=1e-5)
#         res1 = eq1.sweep(var=v1, dt=1e-5)
#     if t % 10 == 0:
#         vi.plot()
#
#
#metodo 2
v0.value = 0.5
v1.value = 0.5
eqn0 = TransientTerm(var=v0) == DiffusionTerm(0.01, var=v0) - DiffusionTerm(1, var=v1)
eqn1 = TransientTerm(var=v1) == DiffusionTerm(1, var=v0) + DiffusionTerm(0.01, var=v1)
eqn = eqn0 & eqn1
for t in range(1): 
     v0.updateOld()
     v1.updateOld()
     eqn.solve(dt=1.e-3)
     vi.plot()
#It is also possible to pose the same equations in vector form:

#v = CellVariable(mesh=m, hasOld=True, value=[[0.5], [0.5]], elementshape=(2,))
#v.constrain([[0], [1]], m.facesLeft)
#v.constrain([[1], [0]], m.facesRight)
#eqn = TransientTerm([[1, 0], 
#                      [0, 1]]) == DiffusionTerm([[[0.01, -1], 
#                                                  [1, 0.01]]])
#vi = Viewer((v[0], v[1]))
#for t in range(1): 
#     v.updateOld()
#     eqn.solve(var=v, dt=1.e-3)
#     vi.plot()
#v = CellVariable(mesh=m, hasOld=True, value=[[0.5], [0.5]], elementshape=(2,))
#v.constrain([[0], [1]], m.facesLeft)
#v.constrain([[1], [0]], m.facesRight)
#eqn = TransientTerm([[1, 0], 
#                      [0, 1]]) == DiffusionTerm([[[0.01, -1], 
#                                                  [1, 0.01]]])
#vi = Viewer((v[0], v[1]))
#for t in range(1): 
#     v.updateOld()
#     eqn.solve(var=v, dt=1.e-3)
#     vi.plot()