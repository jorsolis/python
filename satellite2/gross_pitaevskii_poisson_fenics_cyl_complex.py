#!/usr/bin/env python
# coding: utf-8
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace)
#from dolfin import *
from mshr import Rectangle,generate_mesh, Circle
import plots_jordi as pts
#from time import time
from schrodinger_poisson_fenics_plotting import (plotses,
                                                 plot_sf_sph2, plot_sf_sph,
                                                 plotbessel)
import numpy as np

def mainprog(rhomin, a, b, ncor, Rf = 10, E= 0.034465, Vf = -0.1, fif = .001,
             direct = '/home/jordi/gross', bet = 0.):
    ###########################     MESH         ##################################
    if rhomin==0.:
        circulo = Circle(Point(0.0,0.0), Rf) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    else:
        circulo = Circle(Point(0.0,0.0), Rf) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.)) - Rectangle(Point(rhomin,0.),Point(0.,Rf))
    mesh = generate_mesh(circulo, 164)#
#    schrodinger_poisson_fenics_plotting.graf(mesh,'mesh')
#############               Save mesh to a file           #####################
    File('%s/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor)) << mesh
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
    V = FunctionSpace(mesh, MixedElement([P1, P1,P1]))
###################      Define test functions      ############################
    v_1, v_2, v_3= TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
#############         Split system functions to access components      ########
    psi, Phi, PhiI = split(u)
###############          Define expressions used in variational forms       ###
    fx = Expression('a0*x[0] + b0', degree=1, a0 = a, b0= b)#
#    function = project(fx, FunctionSpace(mesh, 'P', 3))
#    plot(function)
#    plt.show()
####################         Constants           #############################
    k = Constant(E)
    g1 = Constant(0.)
    g2 = Constant(0.)
    g3 = Constant(0.)
    beta = Constant(bet)
##############          Define variational problem     #########################
    F = fx*dot(grad(psi), grad(v_1))*dx  + fx*Phi*Phi*v_1*dx \
    + fx*dot(grad(Phi), grad(v_2))*dx + 2.*fx*psi*Phi*v_2*dx + 2.*fx*k*Phi*v_2*dx \
    + 2.*fx*beta*Phi*Phi*Phi*v_2*dx \
    + fx*dot(grad(PhiI), grad(v_3))*dx  + 2.*fx*k*PhiI*v_3*dx \
    + 2.*fx*beta*Phi*Phi*Phi*v_2*dx- g1*v_1*ds(2) - g2*v_1*ds(1) - g3*v_2*ds(1)
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
    bc4 = DirichletBC(V.sub(2), Constant(0.) , circlebord)  # on PhiI
    bc5 = DirichletBC(V.sub(2), Constant(0.) , bottom)  # on PhiI
###############           Solve variational problem         ###################
    solve(F == 0, u, [bc, bc2, bc3, bc4,bc5]) 
################          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "w")
    output_file.write(u, "solution")
    output_file.close()
    
def save_sol_to_vtk(u, ncor, direct = '/home/jordi/satellite/schrodinger_poisson'): 
    psi_1, phi_1 = u.split()
    vtkfile_psi = File('%s/psi_%d.pvd'%(direct,ncor))
    vtkfile_phi = File('%s/phi_%d.pvd'%(direct,ncor))
    vtkfile_psi << (psi_1)
    vtkfile_phi << (phi_1)
    
def read_saved_file(ncor, direct = '/home/jordi/satellite/schrodinger_poisson'):
    ################         Read mesh         #################################
    mesh = Mesh('%s/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor))
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
      13 :{"Rf" : 100., "En" : 83./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)}}

#################################################################################
#dec = '/home/jordi/gross'
for ncor in range(1,14,1):
    Rf = di2R100[ncor]["Rf"]
    expon = di2R100[ncor]["expon"]
    En = di2R100[ncor]["En"]
    Vf = di2R100[ncor]["Vf"]
    fif = di2R100[ncor]["fif"]

    lamb = 100.
    fif = lamb*fif
    Vf = lamb*Vf
    En = lamb*En
    Rf = Rf/np.sqrt(lamb)

    Ener = En*10**expon
    beta = 0.
    print(beta)
    print(ncor)
    mainprog(0., 1., 0., ncor, Rf= Rf, E= Ener, Vf = Vf, fif =fif, 
             direct = '/home/jordi/gross_evolution/%d/sol0_%d'%(int(beta),ncor), bet = beta) 
