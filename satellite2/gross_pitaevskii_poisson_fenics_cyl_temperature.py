#!/usr/bin/env python
# coding: utf-8
##############################################################################
###############         ES MAS GENERAL gross_pitaevskii_poisson_fenics_cyl_con_m
##############################################################################
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace,
                    assemble,div,SpatialCoordinate, cos, UnitIntervalMesh, 
                    NonlinearVariationalProblem, NonlinearVariationalSolver, 
                    TrialFunction, derivative)
from mshr import Rectangle,generate_mesh, Circle
from schrodinger_poisson_fenics_plotting import (plots_pyplot, graf,
                                                 density3d_files,plotses)
#from gross_pitaevskii_poisson_fenics_cyl_con_m.py import (read_saved_file, fi)
import numpy as np
import plots_jordi as pts

ncells = 164
linearsolver='umfpack'  
nonlinearsolver = 'newton' 
#nonlinearsolver = 'snes'

def mainprog(ncor, Rf = 10, E= 0.034465, Vf = -0.1, fif = .001,Nf=1.,
             direct = '/home/jordi/grossN', bet = 0.,lamb= 0., lambgam= 0., R0 = 0.01):
###########################     MESH         ##################################
#    circulo = Circle(Point(0.0,0.0), Rf) - Circle(Point(0.0,0.0), R0) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    
    circulo = Circle(Point(0.0,0.0), Rf) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    mesh = generate_mesh(circulo, ncells)
#############               Save mesh to a file           #####################
    File('%s/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor)) << mesh
#############           Define boundaries for  Newman conditions       ########
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)    
    tol = 1E-14
    class leftwall(SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary:
                if near(x[0], 0, tol):
                    return True
                
#                elif near(x[1], 0, R0) and near(x[0], 0, R0):
#                    return False 
                
                else:
                    return False
            else:
                return False       

    class bottomwall(SubDomain):
        def inside(self, x, on_boundary):
#            return on_boundary and near(x[1], 0, tol)
            if on_boundary:
                if near(x[1], 0, tol):
                    return True
                
#                elif near(x[1], 0, R0) and near(x[0], 0, R0):
#                    return False 
                
                else:
                    return False
            else:
                return False

    class no_orig(SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary:
                if near(x[1], 0, R0) and near(x[0], 0, R0):
                    return True           
                else:
                    return False
            else:
                return False  
        
    left = leftwall()
    bottom = bottomwall()
    no_origen = no_orig()
        
    left.mark(boundary_markers, 2)
    bottom.mark(boundary_markers, 1)
    no_origen.mark(boundary_markers, 3)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
####################     Define function space       ##########################
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1]))
###################      Define test functions      ############################
    v_1, v_2= TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
#    u_0 = Function(V)
#    t = 0
#############         Split system functions to access components      ########
    psi, Phi = split(u)
###############          Define expressions used in variational forms       ###
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0)#
####################         Constants           #############################
    k = Constant(E)
    g1 = Constant(0.)
    g2 = Constant(0.) #ésta está bien
    beta = Constant(bet)
    lambd = Constant(lamb)
    lambdgamma = Constant(lambgam)
##############          Define variational problem     #########################
    F = fx*dot(grad(psi), grad(v_1))*dx  + fx*Phi*Phi*v_1*dx \
    + fx*dot(grad(Phi), grad(v_2))*dx + 2.*fx*(psi - k)*Phi*v_2*dx \
    + fx*lambd*beta*Phi*Phi*Phi*v_2*dx + 2.*fx*(psi - k - 0.5)*lambdgamma*Phi*v_2*dx \
    - g1*v_1*ds(1) - g2*v_1*ds(2) - g1*v_2*ds(2)
####################          Jacobian                #########################
    du = TrialFunction(V)
    J = derivative(F, u, du)
#############           Define boundaries             #########################
    def circlebord(x, on_boundary):
        if on_boundary:
            if near(x[0], 0, tol) or near(x[1],0, tol):
                return False

#            elif near(x[1], 0, R0) and near(x[0], 0, R0):
#                return False

            else:
                return True
        else:
            return False  
##############          boundary conditions         ###########################
    bc = DirichletBC(V.sub(0), Constant(Vf) , circlebord) # on psi
    bc2 = DirichletBC(V.sub(1), Constant(fif) , circlebord)  # on Phi
    bc3 = DirichletBC(V.sub(1), Constant(0.) , bottom)  # on Phi
    boun = [bc, bc2, bc3]
###############           Solve variational problem         ###################
    problem= NonlinearVariationalProblem(F, u, bcs=boun, J = J)
    solver = NonlinearVariationalSolver(problem)

    solver.parameters['nonlinear_solver'] =  nonlinearsolver   
    solver.parameters['%s_solver'%nonlinearsolver]['absolute_tolerance'] = 1E-9
    solver.parameters['%s_solver'%nonlinearsolver]['relative_tolerance'] = 1E-8
    solver.parameters['%s_solver'%nonlinearsolver]['maximum_iterations'] = 10
#    solver.parameters['%s_solver'%nonlinearsolver]['preconditioner'] = 'ilu'
#    solver.parameters['%s_solver'%nonlinearsolver]['relaxation_parameter'] = 1.0
    solver.parameters['%s_solver'%nonlinearsolver]['linear_solver'] = linearsolver   
#    solver.parameters['%s_solver'%nonlinearsolver]['krylov_solver']['nonzero_initial_guess'] = False
#    solver.parameters['%s_solver'%nonlinearsolver]['report'] = True
    
    solver.parameters['%s_solver'%nonlinearsolver]['lu_solver']['report'] = True
    solver.parameters.update(solver.parameters)
    solver.solve()
################          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "w")
    output_file.write(u, "solution")
    output_file.close() 
     
def read_saved_file(ncor, direct = '/home/jordi/grossT'):
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

###############################################################################
################################################################################
di2R10 = {12:{"Rf" : 7.5, "En" : -1.73, "expon" : -2, "fif" : 1e-1,
              "N" : 1.369, 'lambda':0., "beta" : 0.0,'gamma':0., 'rlam': 1.} }
#################################################################################
#################################################################################
#################################################################################           
#
#de = float(input("run"))
#print(de, type(de))

#nc = float(de)
#nc = 2750

ncor = 12

dec= '/home/jordi/grossT'


expon = di2R10[ncor]["expon"]
En = di2R10[ncor]["En"]  
fif = di2R10[ncor]['fif']
Rf = di2R10[ncor]['Rf']

con = 1. - 1e-2####  con = lamb*T**2/4
beta = -3./(1.0 - con) 
lambgam = con/(1.0 - con)

T = 1e5
lamb = 4.*con/T**2
#Nf = nc/1000.

Nf = di2R10[ncor]['N']
nc = Nf*1000

Vf = -Nf/Rf  
Ener = En*10**expon

print('ncor=',ncor)
print('N=',nc/1000)

print('Nr=', Nf)
print('Rf=',Rf)
print('E=',Ener)
print('fif=',fif)
print('')
print('lambda*beta=',lamb*beta)
print('lambda*gamma=',lambgam)
print('autointeraccion=',lamb)
print('')
print('k_B T=', T,'*10^{-22} eV')
print('T=', T/8.617e-5*1e-22,'K')

DE = {21:{'mu': 156550.,'rlam': 1.0e-3, 'limb' : 4001, 'ref': 1},
      22:{'mu': 15655.0,'rlam': 1.0e-2, 'limb' : 4001, 'ref': 1},
      23:{'mu': 1565.5, 'rlam': 4.0e-3, 'limb' : 1001, 'ref': 1}}
print('')
mm = 22
mu = DE[mm]['mu']
print('mu=',mu, '/kpc')      
me2 = mu**2*(1.0 - con)
print('mu_ef=',np.sqrt(me2), '/kpc')


#mainprog(ncor, Rf = Rf, E= Ener, Vf = Vf, fif = fif , Nf=Nf, R0 = 0.1,
#         direct = "%s/%d"%(dec,nc), bet = beta, lamb= lamb, lambgam= lambgam)
#
#u = read_saved_file(ncor, direct = "%s/%d"%(dec,nc))
#u.set_allow_extrapolation(True)  
#
#
#plots_pyplot(u,Rf, En, expon, ncor,ruta="%s/%d"%(dec,nc), show = False,
#             dens=True, pot=False, otras = False)
