#!/usr/bin/env python
# coding: utf-8
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace)
from mshr import Rectangle,generate_mesh, Circle, Extrude2D
import plots_jordi as pts
from schrodinger_poisson_fenics_plotting import (graf, plots_pyplot_3D)
import numpy as np


##########      BOUNDARY markers, NEWMAN CONDITIONS IN F
##########       

def mainprog(ncor, Rf = 10, E= 0.034465, Vf = -0.1, fif = .001,
             direct = '/home/jordi/gross3D', bet = 0., rhomin=0.):
    ###########################     MESH         ##################################
    high = 2.*np.pi
    g2d = Circle(Point(0,0), Rf) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.)) 
    g3d = Extrude2D(g2d, high) # The z "thickness"  
    mesh = generate_mesh(g3d, 50)#164)  
#    plot(mesh)
#############               Save mesh to a file           #####################
    File('%s/beta%d/mesh_gross3D_%d.xml.gz'%(direct,bet,ncor)) << mesh
#############           Define boundaries for  Newman conditions       ########
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)    

    class rhocero(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], rhomin, tol)
    tol = 1E-14    
    class zetcero(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0, tol)
    
    left = rhocero()
    bottom = zetcero()
    tol = 1E-14    
    left.mark(boundary_markers, 1)
    bottom.mark(boundary_markers, 2)
#    
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
##########      Periodic Boundary conditions           #########################3
    class PeriodicBoundary(SubDomain): # Left boundary is "target domain" G
        def inside(self, x, on_boundary): 
#            return bool(x[2] < tol and x[2] > -tol and on_boundary)
            if on_boundary:
                if near(x[2], 0., tol):
                    return True
                else:
                    return False
            else:
                return False         
        def map(self, x, y): # Map right boundary (H) to left boundary (G)
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - high
    pbc = PeriodicBoundary() # Create periodic boundary condition
#####################     Define function space       ##########################
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1*P1, constrained_domain=pbc)
#####################      Define test functions      ############################
    v_1, v_2= TestFunctions(V)
##############          Define functions for SF and potential      ##############
    u = Function(V)
##############         Split system functions to access components      ########
    psi, Phi = split(u)
#################          Define expressions used in variational forms       ###
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1., b0= 0.)#
    fx2 = Expression('a0*x[0]*x[0]', degree=2, a0 = 1.)#
####################         Constants           #############################
    k = Constant(E)
    beta = Constant(bet)
    g1 = Constant(0.)
    g2 = Constant(0.)
    g3 = Constant(0.)
###############          Define variational problem     #########################
    F = fx2*dot(grad(psi), grad(v_1))*dx + fx*psi.dx(0)*v_1*dx + psi.dx(2)*v_1.dx(2)*dx \
    - psi.dx(2)*fx2*v_1.dx(2)*dx + fx2*Phi*Phi*v_1*dx \
    + fx2*dot(grad(Phi), grad(v_2))*dx + fx*Phi.dx(0)*v_2*dx + Phi.dx(2)*v_2.dx(2)*dx \
    - Phi.dx(2)*fx2*v_2.dx(2)*dx  + 2.*fx2*psi*Phi*v_2*dx + 2.*fx2*k*Phi*v_2*dx \
    + 2.*fx2*beta*Phi*Phi*Phi*v_2*dx #- g1*v_1*ds(2) - g2*v_1*ds(1) - g3*v_2*ds(1)
###############           Define boundaries             #########################
    def zetaigualacero(x, on_boundary):
        if on_boundary:
            if near(x[1], 0, tol):
                return True
            else:
                return False
        else:
            return False    
    def circlebord(x, on_boundary):
        if on_boundary:
            if near(x[0], rhomin, tol) or near(x[1],0, tol) or near(x[2], high, tol) or near(x[2], 0, tol):
                return False
            else:
                return True
        else:
            return False
#############          boundary conditions         ###########################
    bc = DirichletBC(V.sub(0), Constant(Vf) , circlebord) # on u(0)
    bc2 = DirichletBC(V.sub(1), Constant(fif) , circlebord)  # on Phi
    bc3 = DirichletBC(V.sub(1), Constant(0.) , zetaigualacero)  # on Phi
################           Solve variational problem         ###################
    solve(F == 0, u, [bc, bc2, bc3]) 
#################          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/beta%d/u_%d.h5"%(direct,bet,ncor), "w")
    output_file.write(u, "solution")
    output_file.close()
    
def read_saved_file(ncor, bet, direct = '/home/jordi/gross3D'):
    ################         Read mesh         #################################
    mesh = Mesh('%s/beta%d/mesh_gross3D_%d.xml.gz'%(direct,bet,ncor))
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1*P1)
    ###############    Load solution        ###################################
    u = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), "%s/beta%d/u_%d.h5"%(direct,bet,ncor), "r")
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

for ncor in range(2,15,1):
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
    beta = 100.
    print(ncor)
#    mainprog(ncor, Rf= Rf, E= Ener, Vf = Vf, fif =fif, bet = beta) 
    
    u = read_saved_file(ncor, beta)
    u.set_allow_extrapolation(True)
#    print(u[1]((Rf,Rf,0)))
#    print(u[1]((Rf,Rf,2.*np.pi))) 
#    print(u[0]((Rf,Rf,0)))
#    print(u[0]((Rf,Rf,2.*np.pi)))
    
    plots_pyplot_3D(u, Rf, ncor, ruta='/home/jordi/gross3D/beta%d'%beta)

#    p = 80   
#    x = np.linspace(-Rf , Rf , p)
#    y = np.linspace(0 , Rf , p)
#    z = np.linspace(-Rf , Rf , p)
#    
#    def ficuad3D(x,y,z):
#        Phi = []
#        for i in range(0,np.shape(x)[0]):
#            Phii=[]
#            for j in range(0,np.shape(y)[0]):
#                Phiii=[]
#                for k in range(0,np.shape(z)[0]):
##                    poin = (x[i], y[j], z[k])
#                    if z[k]**2 + x[i]**2 + y[j]**2> Rf**2:
#                        Phiii.append(np.nan)
#                    else:
#                        point = (np.sqrt(x[i]**2 + y[j]**2), z[k], np.arctan2(y[j], x[i]))
#                        if z[k]<0:
#                            point = (np.sqrt(x[i]**2 + y[j]**2), -z[k], np.arctan2(y[j], x[i]))
#                            Phiii.append((-u[1](point))**2)
#                        else:
#                            Phiii.append((u[1](point))**2)
#                Phii.append(Phiii)
#            Phi.append(Phii)
#        Phi = np.array(Phi)
#        return np.transpose(Phi)
#
#    a= ficuad3D(x,y,z)
#    pp= p**3
#    b = a.reshape(1,pp)

#    np.savetxt("/home/jordi/gross3D/beta%d/matrix_%d.CSV"%(beta,ncor),b,delimiter=',')
#
#    u.set_allow_extrapolation(True)
#    psi_1, phi_1 = u.split()
#    plot(phi_1**2)