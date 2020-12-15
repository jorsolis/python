#!/usr/bin/env python
# coding: utf-8

###############################################################################
########      NO SE OBTIENE MISMO EIGENVALOR QUE POR SHOOTING    ##############
###############################################################################
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace,
                    assemble,div, TrialFunction, derivative,NonlinearVariationalSolver,
                    AdaptiveNonlinearVariationalSolver, NonlinearVariationalProblem)
from dolfin import *
from mshr import Rectangle,generate_mesh, Circle
import plots_jordi as pts
from schrodinger_poisson_fenics_plotting import plots_pyplot,plotses, graf
from schrodinger_poisson_fenics_cyl import plotrotationcurve
import numpy as np
from boson_star_shooting_solver_N import main
from schrodinger_poisson_shooting_solver import Area
from gross_pitaevskii_poisson_fenics_cyl import (virializacion, print_num_part,
                                                 r95, plot_Nrhoz,read_saved_file)
import matplotlib.pyplot as plt

"""
LINEAR SOLVERS
bicgstab       '  Biconjugate gradient stabilized method                      '
cg             '  Conjugate gradient method                                   '
default        '  default linear solver                                       '
gmres          '  Generalized minimal residual method                         '
minres         '  Minimal residual method                                     '
mumps          '  MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)'
petsc          '  PETSc built in LU solver                                    '
richardson     '  Richardson method                                           '
tfqmr          '  Transpose-free quasi-minimal residual method                '
umfpack        '  UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)' 
""" 
#print('##############   INFO PARAMETERS     ################################')
#info(parameters, True)
#print('##############   LINEAR SOLVER METHODS     ################################')
#list_linear_solver_methods()
#print('##############   KRYLOV SOLVER PRECONDITIONERS     ################################')
#list_krylov_solver_preconditioners()
#print('##############   NON LINEAR VARIATIONAL SOLVER PARMETERS     ################################')
#info(NonlinearVariationalSolver.default_parameters(), True)

linearsolver='umfpack'  
nonlinearsolver = 'newton' 
#nonlinearsolver = 'snes'

deg = 2
ncells = 64 #164

c = 1.5 #1.5
c1 = 1.5
   
def mainprog(ncor, R0 = 0.01, Rf = 10, E= 0.034465, V0=-1.5 ,Vf = -0.1,
             fi0= 1., fif = .001,
             direct = '/home/jordi/satellite/schrodinger_poisson/Positive'):
#    tol = 1E-14 
    tol = R0
    ###########################     MESH         ##################################
    circulo = Circle(Point(0.0,0.0), Rf) - Circle(Point(0.0,0.0), R0) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    mesh = generate_mesh(circulo, ncells)
#    plt.figure()
#    plot(mesh)
#    plt.show()    
##############               Save mesh to a file           #####################
    File('%s/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor)) << mesh
#############           Define boundaries for  Newman conditions       ########
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)    

    class leftwall(SubDomain):
#        tol = 1E-14
        def inside(self, x, on_boundary):
#            return on_boundary and near(x[0], 0, tol)
            if on_boundary:
                if near(x[0], 0, tol):
                    return True
                
                elif (near(x[1], 0, R0) and near(x[0], 0, R0)):
                    return False 
                
                else:
                    return False
            else:
                return False       

    class bottomwall(SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary:
                if near(x[1], 0, tol):
                    return True
                
                elif (near(x[1], 0, R0) and near(x[0], 0, R0)):
                    return False 
                
                else:
                    return False
            else:
                return False    

    class noorigen(SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary:
                if (near(x[1], 0, R0) and near(x[0], 0, R0)):
                    return True           
                else:
                    return False
            else:
                return False
            
#############           Define boundaries             #########################        
    def circlebord(x, on_boundary):
        if on_boundary:
            if (near(x[0], 0, tol) or near(x[1],0, tol)):
                return False
            elif (near(x[1], 0, R0) and near(x[0], 0, R0)):
                return False
            else:
                return True
        else:
            return False
        
    left = leftwall()
    bottom = bottomwall()
    no_origen = noorigen()
    
    left.mark(boundary_markers, 1)
    bottom.mark(boundary_markers, 2)
    no_origen.mark(boundary_markers, 3)
    
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
####################     Define function space       ##########################
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1]))
###################      Define test functions      ############################
    du = TrialFunction(V)
    v_1, v_2 = TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
#############         Split system functions to access components      ########
    psi, Phi = split(u)
###############          Define expressions used in variational forms       ###
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1., b0= 0.)#
####################         Constants           #############################
    k = Constant(E)
    g1 = Constant(0.)
    g2 = Constant(0.)
    g3 = Constant(0.)
##############          Define variational problem     #########################
    F = fx*dot(grad(psi), grad(v_1))*dx + fx*Phi*Phi*v_1*dx \
    + fx*dot(grad(Phi), grad(v_2))*dx  + 2.*fx*psi*Phi*v_2*dx + 2.*fx*k*Phi*v_2*dx \
    - g1*v_1*ds(2) - g2*v_1*ds(1) - g3*v_2*ds(1) - g3*v_2*ds(2) - g3*v_2*ds(3)
####################          Jacobian                #########################
    J = derivative(F, u, du)
##############          boundary conditions         ###########################
    u_D = Expression('a*exp(-pow((x[0] + b ), 2)/(2*pow(c,2)))', degree=2,
                     a=fi0, b=R0, c=c)
    u_D2 = Expression('a*exp(-pow((x[1] + b ), 2)/(2*pow(c,2)))', degree=2,
                      a=fi0, b=R0, c=c1)
    bc = DirichletBC(V.sub(0), Constant(Vf) , circlebord) # on psi
    bc2 = DirichletBC(V.sub(0), Constant(V0) , no_origen)  # on psi
    bc3 = DirichletBC(V.sub(1), Constant(fif) , circlebord)  # on Phi
    bc4 = DirichletBC(V.sub(1), Constant(fi0) , no_origen)  # on Phi
    bc5 = DirichletBC(V.sub(1), u_D, bottom)
    bc6 = DirichletBC(V.sub(1), u_D2, left)
    boun = [bc, bc2, bc3, bc4, bc5, bc6]
###############           Solve variational problem         ###################  
    problem= NonlinearVariationalProblem(F, u, bcs=boun, J = J)
    solver = NonlinearVariationalSolver(problem)

    solver.parameters['nonlinear_solver'] =  nonlinearsolver   
    solver.parameters['%s_solver'%nonlinearsolver]['absolute_tolerance'] = 1E-10
    solver.parameters['%s_solver'%nonlinearsolver]['relative_tolerance'] = 1E-9
    solver.parameters['%s_solver'%nonlinearsolver]['maximum_iterations'] = 40
#    solver.parameters['%s_solver'%nonlinearsolver]['preconditioner'] = 'ilu'
#    solver.parameters['%s_solver'%nonlinearsolver]['relaxation_parameter'] = 1.0
    solver.parameters['%s_solver'%nonlinearsolver]['linear_solver'] = linearsolver   
#    solver.parameters['%s_solver'%nonlinearsolver]['krylov_solver']['nonzero_initial_guess'] = False
#    solver.parameters['%s_solver'%nonlinearsolver]['report'] = True
    
    solver.parameters['%s_solver'%nonlinearsolver]['lu_solver']['report'] = True

    solver.parameters.update(solver.parameters)
    solver.solve()
###############          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "w")
    output_file.write(u, "solution")
    output_file.close() 
################################################################################ 
################################################################################ 
################################################################################ 
def write_catidades(u,Rf, Ener, beta, dec = '', deg = 1):
    N2, Nn = print_num_part(u,Rf)
    Ke, We = virializacion(u, deg= deg)
    psi, phi = u.split()
    psi.set_allow_extrapolation(True)
    phi.set_allow_extrapolation(True)
    print('Psi(Rf,0)=',psi(Rf,0))
    rf = Rf
    thf = np.pi/4
    rhof = rf*np.sin(thf)
    zf = rf*np.cos(thf)
    print('Psi(rf,pi/4)=',psi(rhof,zf))
    f= open("%s/cantidades_%d.txt"%(dec,ncor),"w+")
    f.write(" Ni= %f \r\n " %Nf)
    f.write(" N= %f, -N/rf= %f \r\n " % (Nn, -Nn/Rf))
    f.write(" N= %f, -N/rf= %f, Rf= %f \r\n " % (N2, -N2/Rf, Rf))
    f.write(" E_t= %f, Psi(Rf,0)= %f, Psi(rf,pi/4) = %f \r\n " % (Ener, psi(Rf,0), psi(rhof,zf)))
    f.write(" We= %f, Ke= %f, Ke/We= %f \r\n " % (We, Ke, Ke/abs(We)))
    f.write(" beta= %f, phif= %f \r\n " %(beta, phi(rhof,zf)))
    rf = 0.01
    thf = np.pi/4
    rhof = rf*np.sin(thf)
    zf = rf*np.cos(thf)
    f.write(' fi0 =%f, V0= %f'%(phi(rhof,zf), psi(rhof,zf)))
    f.close()
###############################################################################
def escalamiento(ncor, lam):
    R0 = 0.01/np.sqrt(lam)
    Rf = dic_boson[ncor]["Rf"]/np.sqrt(lam)
    En = dic_boson[ncor]["En"]*lam
    fi0 = dic_boson[ncor]["fi0"]*lam
    fif = dic_boson[ncor]["fif"]*lam
    V0 = dic_boson[ncor]["V0"]*lam
    Vf = dic_boson[ncor]["Vf"]*lam       
    Nf = 0.
    return [R0, Rf, En, fi0, fif,V0, Vf,Nf]  

dic_boson = {1 :{"Rf" : 10., "En" : 4.79128, "expon": -1,
                 'V0':-1.12, "Vf" : -1.18e-8, 'r95':3.81,
                 "fi0": .98, "fif" : 4.5e-10, 'N': 2.04},#BOSON STAR 
             2 :{"Rf" : 10., "En": 4.791, "expon" : -3,
                 'V0':-0.1, "Vf" : -1e-1, "fi0": -4.64e-4, "fif" : 1e-4},
             3 :{"Rf" : 10., "En": 4.791, "expon" : -3,
                 'V0':-0.1, "Vf" : -1e-1, "fi0": -4.64e-2, "fif" : 1e-2},
             4 :{"Rf" : 10., "En" : 4.80520618, "expon": -3,
                 'V0':-0.022, "Vf" : -2.27556040*1e-2,#SIRVE MIXSTATE 
                 "fi0": 0.927224*1e-4, "fif" : 4.21902766*1e-4},
             5 :{"Rf" : 10., "En": 1.25, "expon" : -2,
                 'V0': -0.1, "Vf" : -1e-1, "fi0":-4.85e-4 , "fif" : 1e-4},
             6 :{"Rf" : 5.0, "En": 4.80520618, "expon" : -3,
                 'V0':-0.022, "Vf" : -2.27556040*1e-2,#SIRVE MIXSTATE
                 "fi0": 0.927224*1e-2, "fif" : 4.21902766*1e-2},
             7 :{"Rf" : 10., "En" : 4.79128, "expon": -1,
                 'V0':-1.12, "Vf" : 0.13, 
                 "fi0" : 1. , 'fif': 0.04, 'N': 1.3}#k/w=0.5
             }

#dic_boson = {204:{"Rf" : 10., "En" : 4.80520618, "expon": -3,
#                 'V0':-0.022, "Vf" : -2.27556040*1e-2,#SIRVE MIXSTATE 
#                 "fi0": 08.347*1e-4, "fif" : 4.21902766*1e-4},
#             206 :{"Rf" : 5.0, "En": 4.80520618, "expon" : -3,
#                 'V0':-0.022, "Vf" : -2.27556040*1e-2,#SIRVE MIXSTATE
#                 "fi0": 0.05184, "fif" : 4.21902766*1e-2}}
archivosen = '/home/jordi/satellite/schrodinger_poisson/Positive'

#ya probe con En= 5*1e-5 hasta 45*1e-4 con fif=1e-4 y Vf=-1e-2



#####################           BOSON STAR           ##########################
#ncor = 4
##############################################################################

for ncor in range(1, 2, 1):
    Rf = dic_boson[ncor]["Rf"]
    expon = dic_boson[ncor]["expon"]
    En = dic_boson[ncor]["En"]
    fi0 = dic_boson[ncor]["fi0"]
    fif = dic_boson[ncor]["fif"]

    V0 = dic_boson[ncor]["V0"]
    R0 = 1e-10
    
    Nf = 0.
    Vf = dic_boson[ncor]["Vf"]
    
#    lam = 1./fi0
#    R0, Rf, En, fi0, fif,V0, Vf,Nf = escalamiento(ncor, lam) 
#    ncor = ncor + 200

#    Nf = 2.04  ##ncor+100#    
#    Nf = 1.3 ## ncor 107
#    Vf = -Nf/Rf ##ncor+100
#    ncor = ncor + 100


    Ener = En*10**expon
    mainprog(ncor, R0 = R0, Rf= Rf, E= Ener, V0 = V0, Vf = Vf,
             fi0 = fi0, fif =fif, direct = archivosen)
#    
    u = read_saved_file(ncor, direct = archivosen)
    u.set_allow_extrapolation(True)

    plots_pyplot(u,Rf, En, expon, ncor, ruta = archivosen, show=False, 
                 dens =True, pot=False, cc=c) 
#    plotses(u, En, expon, ncor, archivosen)
#    write_catidades(u,Rf, Ener, 0., dec = archivosen, deg = deg)

#    
#    plot_Nrhoz(u,Rf,ncor,direct= archivosen)
#    r95(ncor, dec=archivosen)
#    plotrotationcurve(ncor,Rf, archivosen)



#r0 = 0.01
#rf = 10.
#stp = 0.01
#
#V_at_length = 0.
#Vprime_at_0 = 0.
#Phi_at_length = 0.
#Phi_prime_at_0 = 0.
##
#ye_boundvalue = [V_at_length, Vprime_at_0, Phi_at_length, Phi_prime_at_0]
#V0_guess = -1.3418  
#Phi0_guess = 1. 
###############################################################################
#E_guess= -0.70
#ncor = 1
#al0 = main(ye_boundvalue, [V0_guess, Phi0_guess, E_guess], 0., r0, rf, stp)
##al0[0] = al0[0]/np.sqrt(al0[1][0,2])
##al0[1][:,2]= al0[1][:,2]/al0[1][0,2]
#
##print('(psi, x^2 dpsi, Phi, x^2 dPhi, E) final',al0[1][-1,:])
#print('(psi, x^2 dpsi, Phi, x^2 dPhi, E) inicial',al0[1][0,:])
#print('(psi, x^2 dpsi, Phi, x^2 dPhi, E) 750',al0[1][750,:])
##print('E=',al0[2])
##print('alpha =',al0[3])
##print(al0[1][:,2].shape)
#pts.plotmultiple([al0[0][0:750]], [al0[1][0:750,2]],[],r'$\hat\mu r$', r"$\Phi$",'',
#                 "", save = False)
#pts.plotmultiple([al0[0][0:750]], [al0[1][0:750,2]**2],[],r'$\hat\mu r$', r"$\rho$",'',
#                 "", save = False)
#pts.plotmultiple([al0[0][:]/15655.,al0[0][:]/156550.],
#                 [4.06e14*al0[1][:,2]**2, 4.06e16*al0[1][:,2]**2],
#                 ['$\mu = 10^{22}$','$\mu = 10^{21}$'],
#                 r'$r (kpc)$', r"$\rho (\frac{M_\odot}{pc^3})$",'',
#                 "", save = False, tipo='logx',logy=True)
##
#pts.plotmultiple([al0[0][0:750],al0[0][0:750]], [al0[1][0:750,0], 0.*al0[0][0:750]] ,[],r'$\hat\mu r$', r"$\psi$",'',
#                 "", save = False)
#Mh = 1e12
#eme = 1.25e9*(Mh/1e12)**(1./3.)
#print(eme)

#print('M=',(Area(al0[1][:,2],al0[0][:],rf,stp)/(4.799e-14 * 15.655))/1e12,'10^12 Msol')
