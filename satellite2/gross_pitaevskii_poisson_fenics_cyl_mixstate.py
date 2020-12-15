#!/usr/bin/env python
# coding: utf-8
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace,
                    assemble,div, SpatialCoordinate)
#from dolfin import *
from mshr import Rectangle,generate_mesh, Circle
import plots_jordi as pts
#from time import time
from schrodinger_poisson_fenics_plotting import (plots_pyplot_mix,plotses,
                                                 graf)
import numpy as np

def mainprog(ncor, Rf = 10, Ee= 0.005, Vf = -0.3, fife = 1e-2, Eb = 0.0125,
             direct = '/home/jordi/satellite/schrodinger_poisson_mix',
             R0 = 0.01, Phicero=np.sqrt(0.99916), Vi = -2.6431, fifb = 1e-5,
             mur= 1., bet = 0., betb= 0.):
###########################     MESH         ##################################
    circulo = Circle(Point(0.0,0.0), Rf) - Circle(Point(0.0,0.0), R0) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    mesh = generate_mesh(circulo, 164)#
#    schrodinger_poisson_fenics_plotting.graf(mesh,'mesh')
#############               Save mesh to a file           #####################
    File('%s/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor)) << mesh
#    hdf = HDF5File(mesh.mpi_comm(),'%s/mesh_schrodinger_poisson_%d.h5'%(direct,ncor), "w")
#    hdf.write(mesh, "/mesh")
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
    v_1, v_2, v_3 = TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
#############         Split system functions to access components      ########
    psi, Phi, Phib = split(u)
###############          Define expressions used in variational forms       ###
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0)#
####################         Constants           #############################
    k = Constant(Ee)
    kb = Constant(Eb)
    g1 = Constant(0.)
    g2 = Constant(0.)
    g3 = Constant(0.)
    mur = Constant(mur)
    beta = Constant(bet)
    betab = Constant(betb)
##############          Define variational problem     #########################
    F = fx*dot(grad(psi), grad(v_1))*dx + fx*0.5*(mur*Phi*Phi + Phib*Phib)*v_1*dx \
    + fx*dot(grad(Phi), grad(v_2))*dx  + 2.*mur*fx*psi*Phi*v_2*dx + 2.*mur*fx*k*Phi*v_2*dx \
    + 2.*mur*fx*beta*Phi*Phi*Phi*v_2*dx \
    - g1*v_1*ds(2) - g2*v_1*ds(1) - g3*v_2*ds(1) \
    + fx*dot(grad(Phib), grad(v_3))*dx  + 2.*fx*psi*Phib*v_3*dx + 2.*fx*kb*Phib*v_3*dx \
    + 2.*fx*betab*Phib*Phib*Phib*v_3*dx \
    - g3*v_3*ds(1) - g3*v_3*ds(2)
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
    def no_origen(x, on_boundary):
        if on_boundary:
            if near(x[1], 0, R0) and near(x[0], 0, R0):
                return True           
            else:
                return False
        else:
            return False        
##############          boundary conditions         ###########################
    bc = DirichletBC(V.sub(0), Constant(Vf) , circlebord) # on psi
    bc1 = DirichletBC(V.sub(0), Constant(Vi) , no_origen)  # on psi
    bc2 = DirichletBC(V.sub(1), Constant(fife) , circlebord)  # on Phie
    bc3 = DirichletBC(V.sub(1), Constant(0.) , bottom)  # on Phie
    bc4 = DirichletBC(V.sub(2), Constant(Phicero) , no_origen)  # on Phib
    bc6 = DirichletBC(V.sub(2), Constant(fifb) , circlebord)  # on Phib
###############           Solve variational problem         ###################
    solve(F == 0, u, [bc, bc1, bc2, bc3, bc4, bc6])
################          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "w")
    output_file.write(u, "solution")
    output_file.close()
    return u

def read_saved_filen(ncor, direct = '/home/jordi/satellite/schrodinger_poisson_mix'):
    ################         Read mesh         #################################
    mesh = Mesh('%s/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor))
#    mesh = Mesh()
#    hdf = HDF5File(mesh.mpi_comm(),'%s/mesh_schrodinger_poisson_%d.h5'%(direct,ncor), "r")
#    hdf.read(mesh, "/mesh", False)
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1,P1]))
    ###############    Load solution        ###################################
    u = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "r")
    input_file.read(u, "solution")
    input_file.close()
    return u  
#####################        number of particles       ########################
def Numero_de_particulas(u):
    psi, phi, phib = u.split()
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0)
    
    ene = fx*(phib*phib + phi*phi*murel*murel)*dx  # corregida
    Nn = assemble(ene)
    eneb = fx*phib*phib*dx   ## corregida
    Nnb = assemble(eneb)
    enee = fx*phi*phi*murel*murel*dx ###corregida
    Nne = assemble(enee)
    return Nn, Nnb, Nne

def virializacion(u):
    psi, phi, phib = u.split()
    V = phi.function_space()
    mesh = V.mesh()
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0, domain=mesh)
    kee = -2.*np.pi*phi*div(fx*grad(phi))*dx(domain=mesh) 
    Ke = assemble(kee)
    keb = -2.*np.pi*phib*div(fx*grad(phib))*dx(domain=mesh) 
    Kb = assemble(keb)   
    we = 2.*np.pi*phi*phi*psi*fx*dx(domain=mesh)    
    We = assemble(we)
    wb = 2.*np.pi*phib*phib*psi*fx*dx(domain=mesh)    
    Wb = assemble(wb)
    print('K=', Ke, 'W=', We)
    print('K/W', Ke/abs(We))
    return Ke, Kb, We, Wb   

def data_fiele(u):
    psi, _, _ = u.split()
    psi.set_allow_extrapolation(True)
    rf = Rf
    thf = np.pi/4
    rhof = rf*np.sin(thf)
    zf = rf*np.cos(thf)
    Ke, Kb, We, Wb = virializacion(u)
    Nn, Nnb, Nne = Numero_de_particulas(u)
    Nf = Nne + Nnb       
    f= open("%s/%s/cantidades_%d.txt"%(dec,Nfe,ncorb),"w+")
    f.write(" Ni = %f, murel = %f,  E_t = %f \r\n " %(Nf,murel,(Eb*Nnb + Ee*Nne)/Nn))
    f.write("N = %f, N/rf = %f , Nb = %f, Ne = %f \r\n " % (Nn, -Nn/Rf, Nnb, Nne))
    f.write("Psi(Rf,0) = %f, Psi(rf,pi/4) = %f \r\n " % (psi(Rf,0), psi(rhof,zf)))
    f.write("Phi0 = %f \r\n "%fi0b)
    f.write("Eb = %f, Ee = %f \r\n " % (Enerb, Enere))
    f.write("Wb = %f, Kb = %f, Kb/Wb = %f \r\n " % (Wb, Kb, Kb/abs(Wb)))
    f.write("We = %f, Ke = %f, Ke/We = %f \r\n " % (We, Ke, Ke/abs(We)))
    f.write("WT = %f, KT = %f, KT/WT = %f \r\n " % ((Wb+We), (Kb+Ke), (Kb+Ke)/abs(Wb+We)))
    f.close()
###############################################################################
##################################################################################
#da2 = float(input(''))
#da2 = int(da2)
# da = float(input(""))
# da = int(da)

dec = '/home/jordi/satellite/schrodinger_poisson_mix/los_buenos'

dic_boson = {2 :{"Rf" : 10., "En": 4.791, "expon" : -3, 'V0':-0.1,
                 "Vf" : -1e-1, "fi0": -4.64e-4, "fif" : 1e-4},
             3 :{"Rf" : 10., "En": 4.791, "expon" : -3, 'V0':-0.1,
                 "Vf" : -1e-1, "fi0": -4.64e-2, "fif" : 1e-2},
             4 :{"Rf" : 10., "En" : 4.80520618, "expon": -3, 'V0':-0.022,
                 "Vf" : -2.27556040*1e-2, "fi0": 0.927224*1e-4, "fif" : 4.21902766*1e-4},
             5 :{"Rf" : 10., "En": 1.25, "expon" : -2, 'V0': -0.1,
                 "Vf" : -1e-1, "fi0":-4.85e-4 , "fif" : 1e-4},
             6 :{"Rf" : 10., "En": 4.80520618, "expon" : -3, 'V0':-0.022,
                 "Vf" : -2.27556040*1e-2, "fi0": 0.927224*1e-3, "fif" : 4.21902766*1e-3}}

di2R10 = {12:{"Rf" : 7.5, "En" : 1.73, "expon" : -2, "fif" : 1e-1,
              "N" : 1.369, "beta" : 0.0,'rlam': 1.}}

ncorb = 6

Rf = dic_boson[ncorb]["Rf"]
exponb = dic_boson[ncorb]["expon"]
Eb = dic_boson[ncorb]["En"]
V0b = dic_boson[ncorb]["V0"]
Vfb = dic_boson[ncorb]["Vf"]
fi0b = dic_boson[ncorb]["fi0"]
fifb = dic_boson[ncorb]["fif"]
R0 = 0.01

rlam = Rf/7.5
Rf = Rf/rlam
R0 = R0/rlam
Eb = rlam**2*Eb
V0b = rlam**2*V0b
Vfb = rlam**2*Vfb
fi0b = rlam**2*fi0b
fifb = rlam**2*fifb

ncore = 12
Rf = di2R10[ncore]["Rf"] ### para corridas 200 y 300
expone = di2R10[ncore]["expon"]
Ee = di2R10[ncore]["En"]
Nfe = di2R10[ncore]["N"]
fife = di2R10[ncore]["fif"]
Vfe = - Nfe/Rf

#rlam = 7.5/10.
#Rf = Rf/rlam
#Nfe = Nfe*rlam
#Ee = rlam**2*Ee
#Vfe = rlam**2*Vfe
#fife = rlam**2*fife


Enere = Ee*10**expone
Enerb = Eb*10**exponb
murel = 1.


mainprog(ncorb, Rf= Rf, Ee= Enere, Vf = Vfe + Vfb, fife = fife, Eb = Enerb,
         Phicero= fi0b, R0 = R0, Vi = V0b, fifb= fifb, mur = murel, 
         direct= '%s/%s'%(dec,Nfe))
  
u = read_saved_filen(ncorb, direct = "%s/%s"%(dec,Nfe))
u.set_allow_extrapolation(True)
data = [Eb, exponb, Ee, expone]
plots_pyplot_mix(u, Rf, ncorb, data, show = False, ruta = '%s/%s'%(dec,Nfe),
                  R0 = R0)
data_fiele(u)

#################################################################################
