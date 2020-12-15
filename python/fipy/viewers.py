#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:58:57 2018

@author: jordi
"""
from matplotlib import pylab
from fipy import *
#
#
pylab.ion()
#fig = pylab.figure()
#ax1 = pylab.subplot((221))
#ax2 = pylab.subplot((223))
#ax3 = pylab.subplot((224))
#
#
k = Variable(name="k", value=0.)
#
#MESH 1D
mesh1 = Grid1D(nx=10000, dx=0.01)
#coord
x, = mesh1.cellCenters
#funciones incognitas
xVar = CellVariable(mesh=mesh1, name="x", value=x)
#
#
viewer1 = MatplotlibViewer(vars=(numerix.sin(0.1 * k * xVar), 
                                 numerix.cos(0.1 * k * xVar / numerix.pi)), 
                           limits={'xmin': 10, 'xmax': 100}, 
                           datamin=-1.1, datamax=1.1,
                           title="Grid1D test",
#                           axes=ax1,
                           legend=None)
#
#MESH 2D
mesh2 = Grid2D(nx=500, ny=100, dx=0.01, dy=0.001)
#coords
x, y = mesh2.cellCenters
#función incognita
xyVar = CellVariable(mesh=mesh2, name="x y", value=2 * x * y)
#
viewer2 = MatplotlibViewer(vars=numerix.sin(k * xyVar), 
                           limits={'ymin': 0.1, 'ymax': 0.9}, 
                           datamin=-0.9, datamax=2.0,
                           title="Grid2D test",
#                           axes=ax2,
                           colorbar=True)
#
#IRREGULAR MESH 2D
mesh3 = (Grid2D(nx=5, ny=10, dx=0.1, dy=0.1)
         + (Tri2D(nx=5, ny=5, dx=0.1, dy=0.1) 
            + ((0.5,), (0.2,))))
x, y = mesh3.cellCenters
xyVar = CellVariable(mesh=mesh3, name="x y", value=x * y)
viewer3 = MatplotlibViewer(vars=numerix.sin(k * xyVar),                            limits={'ymin': 0.1, 'ymax': 0.9}, 
                           datamin=-0.9, datamax=2.0,
                           title="Irregular 2D test",
#                           axes=ax3,
                           cmap = pylab.cm.OrRd)
#viewer = MultiViewer(viewers=(viewer1, viewer2, viewer3))
for kval in range(10):
    k.setValue(kval)
    viewer1.plot()
    viewer2.plot()
    viewer3.plot()
#    viewer.plot()