#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from fipy import *
cellSize = 0.05
radius = 1.
mesh = Gmsh2D('''
              cellSize = %(cellSize)g;
              radius = %(radius)g;
              Point(1) = {0, 0, 0, cellSize};
              Point(2) = {-radius, 0, 0, cellSize};
              Point(3) = {0, radius, 0, cellSize};
              Point(4) = {radius, 0, 0, cellSize};
              Point(5) = {0, -radius, 0, cellSize};
              Circle(6) = {2, 1, 3};
              Circle(7) = {3, 1, 4};
              Circle(8) = {4, 1, 5};
              Circle(9) = {5, 1, 2};
              Line Loop(10) = {6, 7, 8, 9};
              Plane Surface(11) = {10};
              ''' % locals()) 
phi = CellVariable(name = "solution variable",
                   mesh = mesh,
                   value = 0.) 
viewer = None
if __name__ == '__main__':
    try:
        viewer = MatplotlibViewer(vars=phi,
#                           limits={'ymin': 0.1, 'ymax': 0.9}, 
                           datamin=-1.0, datamax=1.0,
#                           title="Phi",
                           cmap = pylab.cm.OrRd)
#        viewer = Viewer(vars=phi, datamin=-1, datamax=1.)
        viewer.plotMesh()
        raw_input("Irregular circular mesh. Press <return> to proceed...") 
    except:
        print "Unable to create a viewer for an irregular mesh (try Gist2DViewer, Matplotlib2DViewer, or MayaviViewer)"
D = 1.
eq = TransientTerm() == DiffusionTerm(coeff=D)
X, Y = mesh.faceCenters 
phi.constrain(X, mesh.exteriorFaces)
timeStepDuration = 10 * 0.9 * cellSize**2 / (2 * D)
steps = 10
for step in range(steps):
    eq.solve(var=phi,
             dt=timeStepDuration) 
    if viewer is not None:
        viewer.plot()
TSVViewer(vars=(phi, phi.grad)).plot(filename="myTSV.tsv")
