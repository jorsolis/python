#!/usr/bin/env python
# coding: utf-8
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace)
#from dolfin import *
from mshr import Rectangle,generate_mesh, Circle
from schrodinger_poisson_fenics_plotting import (plots_pyplot_2,plotses,graf)
import numpy as np

def mainprog(rhomin, a, b, ncor, Rf = 10, Vf = -0.1, fif = .001,
             direct = '/home/jordi/gross_evolution', bet = 0.):
    T = 16.0# final time
    num_steps = 80 # number of time steps
    dt = T / num_steps #time step size
    ###########################     MESH         ##################################
    if rhomin==0.:
        circulo = Circle(Point(0.0,0.0), Rf) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    else:
        circulo = Circle(Point(0.0,0.0), Rf) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.)) - Rectangle(Point(rhomin,0.),Point(0.,Rf))
    mesh = generate_mesh(circulo, 164)#
#    schrodinger_poisson_fenics_plotting.graf(mesh,'mesh')
#############               Save mesh to a file           #####################
    File('%s/sol0_%d/mesh_gross_%d.xml.gz'%(direct,ncor,ncor)) << mesh
#############           Define boundaries for  Newman conditions       ########
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)    

    class leftwall(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], rhomin, tol)
    tol = 1E-14    

    class bottomwall(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0, tol)
    
    left = leftwall()
    bottom = bottomwall()
    tol = 1E-14    
    left.mark(boundary_markers, 1)
    bottom.mark(boundary_markers, 2)
    
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
####################     Define function space       ##########################
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1, P1]))
###################      Define test functions      ############################
    v_1, v_2, v_3= TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
    u_n = Function(V)
    t = 0
#############         Split system functions to access components      ########
    psi, PhiR, PhiI = split(u)
    psi_n, PhiR_n, PhiI_n = split(u_n)
###############          Define expressions used in variational forms       ###
####################      &   Constants           #############################
    fx = Expression('a0*x[0] + b0', degree=1, a0 = a, b0= b)#
    k = Constant(dt)
    g1 = Constant(0.)
    g2 = Constant(0.)
    g3 = Constant(0.)
    beta = Constant(bet)
##############          Define variational problem     #########################
    F2 = fx*dot(grad(psi), grad(v_1))*dx  + fx*PhiI*PhiI*v_1*dx + fx*PhiR*PhiR*v_1*dx \
    + k*fx*dot(grad(PhiR), grad(v_2))*dx + 2.*fx*PhiI*v_2*dx  + 2.*k*fx*psi*PhiR*v_2*dx \
    -2.*fx*PhiI_n*v_2*dx + 2.*k*fx*beta*(PhiR*PhiR + PhiI*PhiI)*PhiR*v_2*dx \
    + k*fx*dot(grad(PhiI), grad(v_3))*dx - 2.*fx*PhiR*v_3*dx  + 2.*k*fx*psi*PhiI*v_3*dx \
    +2.*fx*PhiR_n*v_3*dx + 2.*k*fx*beta*(PhiR*PhiR + PhiI*PhiI)*PhiI*v_3*dx \
    - g1*v_1*ds(2) - g2*v_1*ds(1) - g3*v_2*ds(1) - g3*v_3*ds(1)
#############           Define boundaries             #########################
    def buttonwall(x):
        return near(x[1], 0, tol)
    
    def circlebord(x, on_boundary):
        if on_boundary:
            if near(x[0], rhomin, tol) or near(x[1],0, tol):
                return False
            else:
                return True
        else:
            return False
##############          boundary conditions         ###########################
    bc = DirichletBC(V.sub(0), Constant(Vf) , circlebord) # on psi
    bc2 = DirichletBC(V.sub(1), Constant(fif) , circlebord)  # on Phi
    bc3 = DirichletBC(V.sub(1), Constant(0.) , bottom)  # on Phi
    bc4 = DirichletBC(V.sub(2), Constant(fif) , circlebord)  # on Phi
    bc5 = DirichletBC(V.sub(2), Constant(0.) , bottom)  # on Phi

################    Load solution  cero       ###################################
    mesh2 = Mesh('%s/sol0_%d/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor,ncor))
    u0 = Function(V)
    input_file = HDF5File(mesh2.mpi_comm(), "%s/sol0_%d/u_%d.h5"%(direct,ncor,ncor), "r")
    input_file.read(u0, "solution")
    input_file.close()
#    mesh2 = Mesh('%s/sol0_%d/mesh_gross_%d.xml.gz'%(direct,ncor,ncor))
#    u0 = Function(V)
#    input_file = HDF5File(mesh2.mpi_comm(), "%s/sol0_%d/u_%d_t162.h5"%(direct,ncor,ncor), "r")
#    input_file.read(u0, "solution")
#    input_file.close()
################     First time step Solve     #####################################################     
    u_n.assign(u0) #initial solution
#    solve(F2 == 0, u, [bc, bc2, bc3, bc4, bc5])
    solve(F2 == 0, u, [bc, bc2, bc4])
    u_n.assign(u) # update solution
    t = 0.2
    output_file = HDF5File(mesh.mpi_comm(),
                           "%s/sol0_%d/u_%d_t%d.h5"%(direct,ncor,ncor, 1), "w")
    output_file.write(u, "solution")
    output_file.close()
###############           Solve variational problem         ###################   
    for n in range(2,2+num_steps):
        print(t)
        t += dt
#        solve(F2 == 0, u, [bc, bc2, bc3, bc4, bc5]) 
        solve(F2 == 0, u, [bc, bc2, bc4]) 
        u_n.assign(u)  # update solution
################          Save solution               #########################
        output_file = HDF5File(mesh.mpi_comm(),
                               "%s/sol0_%d/u_%d_t%d.h5"%(direct,ncor,ncor, n), "w")
        output_file.write(u, "solution")
        output_file.close()
        
def read_saved_file0(ncor, direct = '/home/jordi/gross_evolution'):
    ################         Read mesh         #################################
    mesh = Mesh('%s/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor))
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1, P1]))
    ###############    Load solution        ###################################
    u = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "r")
    input_file.read(u, "solution")
    input_file.close()
    return u  

def read_saved_file(ncor,n, direct = '/home/jordi/gross_evolution'):
    ################         Read mesh         #################################
    mesh = Mesh('%s/mesh_gross_%d.xml.gz'%(direct,ncor))
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1, P1]))
    ###############    Load solution        ###################################
    u = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), "%s/u_%d_t%d.h5"%(direct,ncor,n), "r")
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
      13 :{"Rf" : 100., "En" : 83./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)}}

#################################################################################
dec = '/home/jordi/gross_evolution'

ncor = 5

Rf = di2R100[ncor]["Rf"]
Vf = di2R100[ncor]["Vf"]
fif = di2R100[ncor]["fif"]

lamb = 100.
fif = lamb*fif
Vf = lamb*Vf
Rf = Rf/np.sqrt(lamb)
beta = 0.
m=0
mainprog(0., 1., 0., ncor, Rf= Rf, Vf = Vf, fif =fif, 
         direct = '/home/jordi/gross_evolution/%d/m%d'%(int(beta),m), bet = beta) 

u = read_saved_file0(ncor, direct = '/home/jordi/gross_evolution/%d/sol0_%d'%(int(beta), ncor))
u.set_allow_extrapolation(True)
psi, PhiR, PhiI = split(u)
graf(PhiI, '', '/home/jordi/gross_evolution/%d/m%d/sol0_%d'%(int(beta),m, ncor), 
     name='phiR', zlabel= r'Im($\Phi$)')    
graf(PhiR, '', '/home/jordi/gross_evolution/%d/m%d/sol0_%d'%(int(beta),m, ncor),
     name='phiR', zlabel= r'Re($\Phi$)')
for i in range(1,81,4):
    u = read_saved_file(ncor,i, direct = '/home/jordi/gross_evolution/%d/m%d/sol0_%d'%(int(beta),m, ncor))
    u.set_allow_extrapolation(True)
    plots_pyplot_2(u,Rf, ncor,i, ruta='/home/jordi/gross_evolution/%d/m%d/sol0_%d'%(int(beta),m,ncor))