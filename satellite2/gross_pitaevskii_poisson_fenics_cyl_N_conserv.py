#!/usr/bin/env python
# coding: utf-8
##############################################################################
###############    No funciona por la ecuacion para N          ###############
##############################################################################
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace,
                    assemble,div)
#from dolfin import *
from mshr import Rectangle,generate_mesh, Circle
import plots_jordi as pts
#from time import time
from schrodinger_poisson_fenics_plotting import (plots_pyplot,plotses,
                                                 plot_sf_sph2, plot_sf_sph,
                                                 plotbessel, graf)
import numpy as np

def mainprog(ncor, Rf = 10, E= 0.034465, Vf = -0.1, fif = .001,Nf=1.,
             direct = '/home/jordi/grossN', bet = 0., R0 = 0.01):
###########################     MESH         ##################################
    circulo = Circle(Point(0.0,0.0), Rf) - Circle(Point(0.0,0.0), R0) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    mesh = generate_mesh(circulo, 164)#
#    schrodinger_poisson_fenics_plotting.graf(mesh,'mesh')
#############               Save mesh to a file           #####################
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
                
                elif near(x[1], 0, R0) and near(x[0], 0, R0):
                    return False 
                
                else:
                    return False
            else:
                return False       

    class bottomwall(SubDomain):
#        tol = 1E-14
        def inside(self, x, on_boundary):
#            return on_boundary and near(x[1], 0, tol)
            if on_boundary:
                if near(x[1], 0, tol):
                    return True
                
                elif near(x[1], 0, R0) and near(x[0], 0, R0):
                    return False 
                
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
    tol = 1E-14    
    left.mark(boundary_markers, 2)
    bottom.mark(boundary_markers, 1)
    no_origen.mark(boundary_markers, 3)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
####################     Define function space       ##########################
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1, P1]))
###################      Define test functions      ############################
    v_1, v_2, v_3= TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
#############         Split system functions to access components      ########
    psi, Phi, N = split(u)
###############          Define expressions used in variational forms       ###
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0)#
#    function = project(fx, FunctionSpace(mesh, 'P', 3))
#    plot(function)
#    plt.show()
####################         Constants           #############################
    k = Constant(E)
    g1 = Constant(0.)
    g2 = Constant(0.) #ésta está bien
    beta = Constant(bet)
#    pi = Constant(np.pi)
##############          Define variational problem     #########################
    F = fx*dot(grad(psi), grad(v_1))*dx  + fx*Phi*Phi*v_1*dx \
    + fx*dot(grad(Phi), grad(v_2))*dx + 2.*fx*psi*Phi*v_2*dx + 2.*fx*k*Phi*v_2*dx \
    + 2.*fx*beta*Phi*Phi*Phi*v_2*dx \
    - g1*v_1*ds(1) - g2*v_1*ds(2) - g2*v_1*ds(3) \
    - g1*v_2*ds(1) - g2*v_2*ds(3) \
    - g1*v_3*ds(1) - g2*v_3*ds(2) \
    + Phi*Phi*fx*v_3*dx + N.dx(0)*v_3.dx(1)*dx
#    + Phi*Phi*fx*v_3*dx + N.dx(0)*v_3.dx(1)*dx + N.dx(1)*v_3.dx(0)*dx
#############           Define boundaries             #########################
    def circlebord(x, on_boundary):
        if on_boundary:
            if near(x[0], 0, tol) or near(x[1],0, tol):
                return False
            elif near(x[1], 0, R0) and near(x[0], 0, R0):
                return False
            else:
                return True
        else:
            return False  
##############          boundary conditions         ###########################
    bc = DirichletBC(V.sub(0), Constant(Vf) , circlebord) # on psi
    bc2 = DirichletBC(V.sub(1), Constant(fif) , circlebord)  # on Phi
    bc3 = DirichletBC(V.sub(1), Constant(0.) , bottom)  # on Phi
    bc4 = DirichletBC(V.sub(2), Constant(Nf), circlebord) #on N
    bc5 = DirichletBC(V.sub(2), Constant(0.), no_origen) #on N
###############           Solve variational problem         ###################
    solve(F == 0, u, [bc, bc2, bc3, bc4, bc5]) ###ncor
################          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "w")
    output_file.write(u, "solution")
    output_file.close()
       
def read_saved_file(ncor, direct = '/home/jordi/grossN'):
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
################################################################################
di2R10 = {1 :{"Rf" : 10., "En" : 6.0, "expon" : -2, "fif" : 1e-2},
          2 :{"Rf" : 10., "En" : 5.82, "expon": -2, "fif" : 1e-2},
          3 :{"Rf" : 10., "En" : 4.5, "expon" : -2, "fif" : 1e-2},
          4 :{"Rf" : 10., "En" : 4.0, "expon" : -2, "fif" : 1e-2},
          5 :{"Rf" : 10., "En" : 5.0, "expon" : -3, "fif" : 1e-2},
          6 :{"Rf" : 10., "En" : 2.5, "expon" : -3, "fif" : 1e-2},
          7 :{"Rf" : 10., "En" : 2.0, "expon" : -3, "fif" : 1e-2},
          8 :{"Rf" : 10., "En" : 8.0, "expon" : -4, "fif" : 1e-2},
          9 :{"Rf" : 10., "En" : 10., "expon" : -5, "fif" : 1e-3}}
###############################################################################
def numero_particulas(f,r,R,dr):
    "integral definida de r*f(r,z)**2 drdz  r de 0 a R y z de 0 a R"
    A=0.
    elem = int(np.rint(R/dr))
    for i in range(0,elem,1):
        for j in range(0,elem,1):
            A+= dr*f[i,j]*r[j]*dr       
    return A
#
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
#################################################################################
#de = float(input("run"))
#print(de, type(de))
##ncor = float(de)+100
#ncor = float(de)

#dec= '/home/jordi/grossN'
#for n in range(1,103,4):
#    ncor = n + 100
#    En = n    
#    expon = -7
#    Ener = En*10**expon 
#    print(Ener)
#    Rf = 10.
#    Nf = 1.
#    Vf = -Nf/Rf
#    fif = 1e-2
#    beta = 0.

##    mainprog(ncor, Rf= Rf, E= Ener, Vf = Vf, fif =fif, 
##             bet = beta, Nf= Nf) 
#    u = read_saved_file(ncor)
#    u.set_allow_extrapolation(True)
#    plots_pyplot(u,Rf, En, expon, ncor,ruta=dec)
#    psi, phi, N = u.split()
#    graf(N, '', dec, name='N_%d'%ncor, zlabel= r'$N$', 
#         rango =True, vmin = 0., vmax = Nf)
#    

#dec = '/home/jordi/grossN/ncor2'
#
##for ncor in range(1,10,1):
#expon = di2R10[ncor]["expon"]
#En = di2R10[ncor]["En"]  
#fif = di2R10[ncor]['fif']
#Rf = di2R10[ncor]['Rf']
#Nf = 4.61
#Vf = -Nf/Rf   
#Ener = En*10**expon
#beta = 0.
#print(beta)
#print(ncor)
#print(Ener)
#mainprog(ncor, Rf= Rf, E= Ener, Vf = Vf, fif =fif, Nf = Nf, R0 = 0.1,
#         bet = beta, direct = dec) 
#
#u = read_saved_file(ncor, direct = dec)
#u.set_allow_extrapolation(True)
#plots_pyplot(u,Rf, En, expon, ncor,ruta=dec, show = False)
#psi, phi, N = u.split()
##graf(N, '', dec, name='N_%d'%ncor, zlabel= r'$N$', rango =True, vmin = 0., vmax = Nf)
######################        number of particles       ########################
#fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0)
#ene = fx*phi*phi*dx  # corregida
#Nn = assemble(ene)
#print('N=',Nn, 'N/rf= ', -Nn/Rf)
#################       Viralization        ####################################
#we = 2.*np.pi*phi*phi*psi*dx
#V = phi.function_space()
#mesh = V.mesh()
#fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0, domain=mesh)
#kee = -2.*np.pi*phi*div(fx*grad(phi))*dx
#We = assemble(we)
#Ke = assemble(kee)
#
#print('K=', Ke, 'W=', We)
#print('K/W', Ke/abs(We))
#
#
#psi.set_allow_extrapolation(True)
#print('Psi(Rf,0)=',psi(Rf,0))
#rf = Rf
#thf = np.pi/4
#rhof = rf*np.sin(thf)
#zf = rf*np.cos(thf)
#print('Psi(rf,pi/4)=',psi(rhof,zf))
#f= open("%s/cantidades_%d.txt"%(dec,ncor),"w+")
#f.write(" Ni=%f \r\n " %Nf)
#f.write(" N= %f, N/rf= %f \r\n " % (Nn, -Nn/Rf))
#f.write(" E_t= %f, Psi(Rf,0)= %f, Psi(rf,pi/4) = %f \r\n " % (Ener, psi(Rf,0), psi(rhof,zf)))
#f.write(" We= %f, Ke= %f, Ke/We= %f \r\n " % (We, Ke, Ke/abs(We)))
#f.close()
#    