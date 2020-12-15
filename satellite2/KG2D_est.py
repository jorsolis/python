#!/usr/bin/env python
# coding: utf-8
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace,DOLFIN_EPS)
#from dolfin import *
from mshr import Rectangle,generate_mesh, Circle
import plots_jordi as pts
#from time import time
from schrodinger_poisson_fenics_plotting import (plots_pyplot_3, graf, plotses)
import numpy as np

def mainprog( a, ncor, Rf = 120, E= 0.034465, fif = .001,
             direct = '/home/jordi/KG2D', omega = 1./240., lam= 6.4, fimax = 2.*np.pi):
    ###########################     MESH         ##################################
    rec = Rectangle(Point(0,0),Point(Rf, fimax))
    mesh = generate_mesh(rec, 164)#
#    plot(mesh)
#############               Save mesh to a file           #####################
    File('%s/mesh_KG2D_%d.xml.gz'%(direct,ncor)) << mesh
#############           Define boundaries for  Newman conditions       ########
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)    

    class leftwall(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0, tol)

    class rightwall(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], Rf, tol)
        
    tol = 1E-14    

    class bottomwall(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0, tol)
    class uperwall(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], fimax, tol)
    
    left = leftwall()
    right = rightwall()
    bottom = bottomwall()
    up = uperwall()
    
    left.mark(boundary_markers, 1)
    bottom.mark(boundary_markers, 2)
    right.mark(boundary_markers, 3)
    up.mark(boundary_markers, 4)

    class PeriodicBoundary(SubDomain):
    
        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary)
    
        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[1] = x[1] - fimax
            y[0] = x[0]
        
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
####################     Define function space       ##########################
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1]), constrained_domain=PeriodicBoundary())
###################      Define test functions      ############################
    v_1, v_2= TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
#############         Split system functions to access components      ########
    PhiR, PhiI = split(u)
###############          Define expressions used in variational forms       ###
    f = Expression('a0*x[0]', degree=1, a0 = a)#
    f2 = Expression('a0*x[0]*x[0]', degree=2, a0 = a)#
####################         Constants           #############################
    k = Constant(E)
    g2 = Constant(0.)
    g3 = Constant(0.)
    Om = Constant(omega)
    L = Constant(lam)
##############          Define variational problem     #########################
    F = - f2*k*k*PhiR*v_1*dx + f2*dot(grad(PhiR), grad(v_1))*dx + f*PhiR.dx(0)*v_1*dx \
    + (1. - f2*(1. - Om*Om))*PhiR.dx(1)*v_1.dx(1)*dx \
    - 2.*k*Om*Om*f2*PhiI.dx(1)*v_1*dx + (1. - (PhiR*PhiR + PhiI*PhiI))*L*PhiR*f2*v_1*dx \
    - f2*k*k*PhiI*v_2*dx + f2*dot(grad(PhiI), grad(v_2))*dx + f*PhiI.dx(0)*v_2*dx \
    + (1. - f2*(1. - Om*Om))*PhiI.dx(1)*v_2.dx(1)*dx \
    + 2.*k*Om*Om*f2*PhiR.dx(1)*v_2*dx + (1. - (PhiR*PhiR + PhiI*PhiI))*L*PhiI*f2*v_2*dx \
    - g2*v_1*ds(1) - g3*v_2*ds(1)
#############           Define boundaries             #########################
#    def buttonwall(x):
#        return near(x[1], 0, tol)
    
#    def circlebord(x, on_boundary):
#        if on_boundary:
#            if near(x[0], rhomin, tol) or near(x[1],0, tol):
#                return False
#            else:
#                return True
#        else:
#            return False
###############          boundary conditions         ###########################
    bc = DirichletBC(V.sub(0), Constant(fif) ,right)  # on PhiR
    bc3 = DirichletBC(V.sub(1), Constant(fif) , right)  # on PhiI
##############           Solve variational problem         ###################
    solve(F == 0, u, [bc, bc3]) 
################          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "w")
    output_file.write(u, "solution")
    output_file.close()
#    
def save_sol_to_vtk(u, ncor, direct = '/home/jordi/KG2D'): 
    psi_1, phi_1 = u.split()
    vtkfile_psi = File('%s/psi_%d.pvd'%(direct,ncor))
    vtkfile_phi = File('%s/phi_%d.pvd'%(direct,ncor))
    vtkfile_psi << (psi_1)
    vtkfile_phi << (phi_1)
    
def read_saved_file(ncor, direct = '/home/jordi/KG2D'):
    ################         Read mesh         #################################
    mesh = Mesh('%s/mesh_KG2D_%d.xml.gz'%(direct,ncor))
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1]))
    ###############    Load solution        ###################################
    u = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "r")
    input_file.read(u, "solution")
    input_file.close()
    return u  
################################################################################
di2R100 = {1 :{"Rf" : 100., "En" : 1./2., "expon": -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)}, 
      2 :{"Rf" : 100., "En": 2./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},##con fif = 1e-7 tambien funciona
      3 :{"Rf" : 100., "En": 8./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},##con fif = 1e-7 tambien funciona
      4 :{"Rf" : 100., "En": 9./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},##con fif = 1e-7 tambien funciona
      5 :{"Rf" : 100., "En" : 12./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      6 :{"Rf" : 100., "En": 16./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      7 :{"Rf" : 100., "En": 17./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      8 :{"Rf" : 100., "En" : 18./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      9 :{"Rf" : 100., "En" : 40./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      10 :{"Rf" : 100., "En": 49./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      11 :{"Rf" : 100., "En": 50./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      12 :{"Rf" : 100., "En" : 55./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      13 :{"Rf" : 100., "En" : 83./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      14 :{"Rf" : 100., "En" : 60./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)}}

#################################################################################
dec = '/home/jordi/KG2D'
for ncor in range(1,15,1):
    expon = di2R100[ncor]["expon"]
    En = di2R100[ncor]["En"]
    Ener = En*10**expon
    Rf = 100.
    print(ncor)
#    mainprog(1., ncor, Rf= Rf, E = Ener) 
#
    u = read_saved_file(ncor, direct = '/home/jordi/KG2D')
    u.set_allow_extrapolation(True)
    plots_pyplot_3(u,Rf, ncor, En, expon)
##    plotses(u, En,expon, ncor)