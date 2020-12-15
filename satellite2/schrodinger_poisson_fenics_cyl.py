#!/usr/bin/env python
# coding: utf-8

#################################################################################
################            EL MAS GENERAL ES gross_pitaevskii_poisson_fenics_cyl_con_m
################################################################################

from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace,
                    assemble,div,FacetNormal)
#from dolfin import *
from mshr import Rectangle,generate_mesh, Circle
import plots_jordi as pts
#from time import time
from schrodinger_poisson_fenics_plotting import (plots_pyplot,plotses,
                                                 plot_sf_sph2, plot_sf_sph,
                                                 plotbessel,graf)
import numpy as np

def mainprog(rhomin, a, b, ncor, Rf = 10, E= 0.034465, Vf = -0.1, fif = .001,
             direct = '/home/jordi/satellite/schrodinger_poisson'):
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
    V = FunctionSpace(mesh, MixedElement([P1, P1]))
###################      Define test functions      ############################
    v_1, v_2= TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
#############         Split system functions to access components      ########
    psi, Phi = split(u)
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
##############          Define variational problem     #########################
#    F = fx*dot(grad(psi), grad(v_1))*dx + fx*dot(grad(Phi), grad(v_2))*dx \
#    + fx*Phi*Phi*v_1*dx + fx*psi*Phi*v_2*dx + fx*k*Phi*v_2*dx \
#    - g1*v_1*ds(2) - g2*v_1*ds(1) - g3*v_2*ds(1)
    F = fx*dot(grad(psi), grad(v_1))*dx + fx*dot(grad(Phi), grad(v_2))*dx \
    + fx*Phi*Phi*v_1*dx + 2.*fx*psi*Phi*v_2*dx + 2.*fx*k*Phi*v_2*dx \
    - g1*v_1*ds(2) - g2*v_1*ds(1) - g3*v_2*ds(1)
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
###############           Solve variational problem         ###################
    solve(F == 0, u, [bc, bc2, bc3])  
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

di = {1 :{"Rf" : 10., "En" : 3.4465, "expon": -2, "Vf" : -1e-1, "fif" : 1e-3}, 
      2 :{"Rf" : 1000., "En": 1.1, "expon" : -4, "Vf" : -1e-3, "fif" : 1e-6},##con fif = 1e-7 tambien funciona
      3 :{"Rf" : 1000., "En": 5.1, "expon" : -4, "Vf" : -1e-3, "fif" : 1e-6},##con fif = 1e-7 tambien funciona
      4 :{"Rf" : 1000., "En": 4.1, "expon" : -4, "Vf" : -1e-3, "fif" : 1e-6},##con fif = 1e-7 tambien funciona
      5 :{"Rf" : 100., "En" : 3.4465, "expon" : -4, "Vf" : -1e-2, "fif" : 1e-5},
      6 :{"Rf" : 1000., "En": 3.4465, "expon" : -6, "Vf" : -1e-3, "fif" : 1e-7},
      7 :{"Rf" : 1000., "En": 3.4465, "expon" : -6, "Vf" : -1e-4, "fif" : 1e-7},
      8 :{"Rf" : 100., "En" : 3.4465, "expon" : -4, "Vf" : -1e-1, "fif" : 1e-5},
      9 :{"Rf" : 100., "En" : 3.4465, "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},
      10:{"Rf" : 1000., "En": 3.4465, "expon" : -6, "Vf" : -1e-3, "fif" : 1e-6},
      11:{"Rf" : 10., "En": 3.4465, "expon" : -2, "Vf" : -1e-1, "fif" : 1e-2},
      12 :{"Rf" : 10., "En" : 3.4465/2., "expon": -2, "Vf" : -1e-1, "fif" : 1e-3}}

diR100 = {1 :{"Rf" : 100., "En" : 1., "expon": -4, "Vf" : -1e-2, "fif" : 1e-4}, 
      2 :{"Rf" : 100., "En": 2., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},##con fif = 1e-7 tambien funciona
      3 :{"Rf" : 100., "En": 8., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},##con fif = 1e-7 tambien funciona
      4 :{"Rf" : 100., "En": 9., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},##con fif = 1e-7 tambien funciona
      5 :{"Rf" : 100., "En" : 12., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},
      6 :{"Rf" : 100., "En": 16., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},
      7 :{"Rf" : 100., "En": 17., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},
      8 :{"Rf" : 100., "En" : 18., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},
      9 :{"Rf" : 100., "En" : 40., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},
      10 :{"Rf" : 100., "En": 49., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},
      11 :{"Rf" : 100., "En": 50., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},
      12 :{"Rf" : 100., "En" : 55., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4},
      13 :{"Rf" : 100., "En" : 83., "expon" : -4, "Vf" : -1e-2, "fif" : 1e-4}}

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
#dec = '/home/jordi/satellite/schrodinger_poisson/solsR10'
#dec = '/home/jordi/satellite/schrodinger_poisson/sols2R100'
#dec = '/home/jordi/satellite/schrodinger_poisson/sols3R100'

#dec = '/home/jordi/satellite/schrodinger_poisson'

#for ncor in range(12,13,1):
#    Rf = di[ncor]["Rf"]
#    expon = di[ncor]["expon"]
#    En = di[ncor]["En"]
#    Vf = di[ncor]["Vf"]
#    fif = di[ncor]["fif"]
    
#    Rf = di2R100[ncor]["Rf"]
#    expon = di2R100[ncor]["expon"]
#    En = di2R100[ncor]["En"]
#    Vf = di2R100[ncor]["Vf"]
#    fif = di2R100[ncor]["fif"]
#
#    lamb = 100.
#    fif = lamb*fif
#    Vf = lamb*Vf
#    En = lamb*En
#    Rf = Rf/np.sqrt(lamb)
##
#    Ener = En*10**expon
#    mainprog(0., 1., 0., ncor, Rf= Rf, E= Ener, Vf = Vf, fif =fif, direct = dec) 
#    u = read_saved_file(ncor, direct = dec)
#    u.set_allow_extrapolation(True)
#    plots_pyplot(u,Rf, En, expon, ncor, ruta=dec)

#En = float(input("Energy"))
#print(En, type(En))
#ncor = float(En)
#Rf = 100
#expon = -4
#Ener = En*10**expon
#Vf = -1e-2
#fif = 1e-4
 
#for i in range(10, 11, 1):
#for i in (1,2,8,9,12,16,17,18,40,49,50,55,83):
#for i in (4,13,16,19,22,25,28,31,34,37,40,52,55,58,61,64,67,70,76,79,82,85,88,94,97):
#for i in (11,14,20,23,26,29,32,35,38,50,53,56,59,65,68,80,83,86,89,95,98):
#    ncor = i
#    u = read_saved_file(ncor, direct = dec)
#    u.set_allow_extrapolation(True)
#    Rf = 10
#    En = 24.5
#    expon = -1
#    plotses(u, En, expon, ncor, direct = dec)
#    plots_pyplot(u,Rf, En, expon, ncor, ruta = dec)



###############################################################################
def numero_particulas(f,r,R,dr):
    "integral definida de r*f(r,z)**2 drdz  r de 0 a R y z de -R a R"
    A=0.
    elem = int(np.rint(R/dr))
    for i in range(0,elem,1):
        for j in range(0,elem,1):
            A+= dr*f[i,j]*r[i]*dr
#    print(A*4.*np.pi)        
    return A*4.*np.pi

def fi(u, rho,z,Rf, n=1):
    Phi = []
    for i in range(0,np.shape(rho)[0]):
        Phii=[]
        for j in range(0,np.shape(z)[0]):
            point = (rho[i], z[j]) 
            if z[j]**2 + rho[i]**2 > Rf**2:
                Phii.append(0.)
            else:
                if z[j]<0:
                    if rho[i]<0:
                        point = (-rho[i], -z[j])
                        Phii.append((-u[1](point))**n)
                    else:
                        point = (rho[i], -z[j])
                        Phii.append((-u[1](point))**n)
                else:
                    if rho[i]<0:
                        point = (-rho[i], z[j])
                        Phii.append((u[1](point))**n)
                    else:
                        Phii.append((u[1](point))**n)    
        Phi.append(Phii)
    Phi = np.array(Phi)
    return np.transpose(Phi)  

####################        number of particles       ###############################3
#for ncor in range(12,13,1):
#    Rf = di2R100[ncor]["Rf"]
##    lamb = 100.
##    Rf = Rf/np.sqrt(lamb)
###
#    u = read_saved_file(ncor, direct = dec)
#    u.set_allow_extrapolation(True)   
#    tol = 0.001
#    rho = np.linspace(tol, Rf - tol, 201) 
#    dr = rho[1]-rho[0]
###    
#    N = numero_particulas(fi(u,rho,rho,Rf, n=2), rho, Rf, dr)
#    print(ncor)
#    print('N=',N, 'N/rf= ',N/Rf)
#    psi, phi = u.split()
#    print('rf=',Rf, 'Psi(rf)=',psi(Rf,0))
#    graf(phi,'','')
#    h = np.sqrt(N/4.)
#    graf(phi/h,'','')
    #print('M=',numero_particulas(fi(rho,rho,Rf, n=2), rho, Rf, dr)/(4.799e-14 * 15.655),'Msol')
#    psi, phi = u.split() 
#    energy = 0.5*dot(grad(psi), grad(psi))*dx
#    ene = phi*phi*dx
#    w = 0.5*phi*phi*psi*dx
#    ke = -0.5*phi*div(grad(phi))*dx
#    N = assemble(ene)
#    W = assemble(w)
#    K = assemble(ke)
#    print(N, W, K)
#    print(K/W)

#    mesh = Mesh('%s/mesh_schrodinger_poisson_%d.xml.gz'%(dec,ncor))
#    circulo = Circle(Point(0.0,0.0), Rf) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
#    mesh = generate_mesh(circulo, 164)
#    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)    
#    class leftwall(SubDomain):
#        tol = 1E-14
#        def inside(self, x, on_boundary):
#            return on_boundary and near(x[0], 0, tol)
#    tol = 1E-14    
#
#    class bottomwall(SubDomain):
#        tol = 1E-14
#        def inside(self, x, on_boundary):
#            return on_boundary and near(x[1], 0, tol)
#    left = leftwall()
#    bottom = bottomwall()
#    tol = 1E-14    
#    left.mark(boundary_markers, 1)
#    bottom.mark(boundary_markers, 2)
#    n = FacetNormal(mesh)
#    circ = circlebord()
#    tol = 1E-14    
#    circ.mark(boundary_markers, 1)  
#    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
#    flue = -dot(grad(phi), n)*ds(1)  
#    flujo = assemble(flue)
#    print(flujo)
    
#    flux = -dot(grad(phi), n)*ds
#    total_flux = assemble(flux)
#    print(total_flux)

####################        rotation curves       ###############################3
#dec = '/home/jordi/satellite/schrodinger_poisson/solsR10'
#ncor = 13
#Rf = 10
def plotrotationcurve(ncor,Rf, dec):
    u = read_saved_file(ncor, direct = dec)
    u.set_allow_extrapolation(True)
    psi_1, _ = u.split()
    V2 = psi_1.function_space()
    mesh2 = V2.mesh()
    degree = V2.ufl_element().degree()
    W = VectorFunctionSpace(mesh2, 'P', degree)    
    dV = project(grad(psi_1), W)
    dV.set_allow_extrapolation(True)
    dV_x, _ = dV.split(deepcopy=True) # extract components
    dV_x.set_allow_extrapolation(True)
    
    tol = 0.001
    rho = np.linspace(tol, Rf - tol, 101) 
    z = np.linspace(-Rf + tol, Rf - tol, 201) 
    Rho, Z = np.meshgrid(rho,z)
    def derrho(rho,z):    
        Phi = []
        for j in range(0,np.shape(rho)[0]):
            Phii=[]
            for i in range(0,np.shape(rho)[1]):
                point = (rho[j][i], abs(z[j][i])) 
                if rho[j][i]**2 + z[j][i]**2 > Rf**2:
                    Phii.append(np.nan)
                else:
                     Phii.append(dV_x(point))
            Phi.append(Phii)
        Phi = np.array(Phi)
        return Phi  
    
    v_cuad = rho*derrho(Rho,Z)
    pts.densityplot(Rho,Z,derrho(Rho,Z),r"$\rho$",r"$z$",
                   r"$\frac{\partial V(\rho,z)}{\partial \rho}$",'',
                   name='%s/dVdrho_%d'%(dec,ncor))
    pts.parametricplot(rho,v_cuad[99, :],r"$\hat{\mu}\rho$",r'$v^2/c^2$',
                       '','%s/rot_curve_%d'%(dec,ncor), save=True)

#for ncor in range(13,14,1):
#    plotrotationcurve(ncor,Rf, dec)
###############################################################################
#plotbessel(8,rf = 1000., kmu = 0.030)
#plotbessel(1,rf = 100., kmu = 0.10)
#pts.sphericalplot3D_harmonics(1,0)

#import numpy as np
#import scipy.special as spe
#x, z = np.meshgrid(np.linspace(-100,100, 100), np.linspace(-100,100, 100))
#nu = 1
#kmu = 0.04
#def f(x,z):
#    return spe.jv(nu+0.5,kmu*np.sqrt(x**2 + z**2))**2 * spe.legendre(nu)(np.cos(np.arctan2(x,z)))**2
#pts.densityplot(x,z,f(x,z),r'$x$',r'$z$',r'$\Phi^2 (x,0,z)$','density',name=None)

 
#Rf=100
#tol = 0.001
#z = np.linspace(-Rf + tol, Rf - tol, 201)
#rho = np.linspace(-Rf + tol, Rf - tol, 201)    
#Rho, Z = np.meshgrid(rho, z)
#
#arr = np.array([Rho,Z,fi(rho,z,Rf, n=2)])
#
#np.save("/home/jordi/satellite/schrodinger_poisson/orbitas/adelante/rhozfi.npy", arr)
