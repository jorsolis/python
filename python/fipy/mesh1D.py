#!/usr/bin/env python
from fipy import *
#
nx = 50
dx = 1.
mesh = Grid1D(nx=nx, dx=dx)
#
phi = CellVariable(name="solution variable", mesh=mesh)
phiAnalytical = CellVariable(name="analytical value", mesh=mesh)
#
D = FaceVariable(mesh=mesh, value=1.0)
L= nx * dx
#
X = mesh.faceCenters[0]
D.setValue(0.1, where=(L / 4. <= X) & (X < 3. * L / 4.))
valueLeft = 0.
fluxRight = 1.
phi.faceGrad.constrain([fluxRight], mesh.facesRight)
phi.constrain(valueLeft, mesh.facesLeft)
phi.setValue(0)
ec= DiffusionTerm(D) == 0
ec.solve(var=phi)
x = mesh.cellCenters[0]
phiAnalytical.setValue(x)
phiAnalytical.setValue(10 * x - 9. * L / 4. , 
                       where=(L / 4. <= x) & (x < 3. * L / 4.))
phiAnalytical.setValue(x + 18. * L / 4. , 
                       where=3. * L / 4. <= x)
#print phi.allclose(phiAnalytical, atol = 1e-8, rtol = 1e-8)
viewer = Viewer(vars=(phi, ), datamin=0., datamax=300.)
viewer2 = Viewer(vars=(phiAnalytical, ), datamin=0., datamax=300)
viewer.plot()
viewer2.plot()

