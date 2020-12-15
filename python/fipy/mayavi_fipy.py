#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:23:54 2018

@author: jordis
"""
#   1D
#
#from fipy import *
##
#mesh = Grid1D(nx=100)
#x, = mesh.cellCenters
#xVar = CellVariable(mesh=mesh, name="x", value=x)
#k = Variable(name="k", value=0.)
#viewer = MayaviClient(vars=(numerix.sin(k * xVar), numerix.cos(k * xVar / numerix.pi)), 
#                limits={'xmin': 10, 'xmax': 90}, 
#                datamin=-0.9, datamax=2.0,
#                title="MayaviClient test")
#for kval in numerix.arange(0,0.3,0.03):
#    k.setValue(kval)
#    viewer.plot()
#viewer._promptForOpinion()
#
#  2D
#
#from fipy import *
#mesh = (Grid2D(nx=5, ny=10, dx=0.1, dy=0.1)
#        + (Tri2D(nx=5, ny=5, dx=0.1, dy=0.1) 
#         + ((0.5,), (0.2,))))
#x, y = mesh.cellCenters
#xyVar = CellVariable(mesh=mesh, name="x y", value=x * y)
#k = Variable(name="k", value=0.)
#viewer = MayaviClient(vars=numerix.sin(k * xyVar), 
#                limits={'ymin': 0.1, 'ymax': 0.9}, 
#                datamin=-0.9, datamax=2.0,
#                title="MayaviClient test")
#for kval in range(10):
#    k.setValue(kval)
#    viewer.plot()
#viewer._promptForOpinion()
#
#
#  3D
from fipy import *
mesh = Grid3D(nx=50, ny=100, nz=10, dx=0.1, dy=0.01, dz=0.1)
x, y, z = mesh.cellCenters
xyzVar = CellVariable(mesh=mesh, name=r"x y z", value=x * y * z)
k = Variable(name="k", value=0.)
viewer = MayaviClient(vars=numerix.sin(k * xyzVar), 
                    limits={'ymin': 0.1, 'ymax': 0.9}, 
                    datamin=-0.9, datamax=2.0,
                    title="MayaviClient test")
for kval in range(10):
    k.setValue(kval)
    viewer.plot()
viewer._promptForOpinion()