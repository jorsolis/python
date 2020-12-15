#!/usr/bin/env python
# coding: utf-8
#####################################################################################
##########            NO SIRVE, ESTA MAL LA ECUACION DE N                ############
##########         VER GROSS_PITAEVSKII_POISSON_FENICS_CYL_MISXTATE    ############
#####################################################################################
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
             mur= 1., Nf = 3., bet = 0., betb= 0.):
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
#    V = FunctionSpace(mesh, MixedElement([P1, P1, P1, P1]))
    V = FunctionSpace(mesh, MixedElement([P1, P1, P1]))
###################      Define test functions      ############################
#    v_1, v_2, v_3, v_4= TestFunctions(V)
    v_1, v_2, v_3 = TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
#############         Split system functions to access components      ########
#    psi, Phi, Phib, N = split(u)
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
    # pi = Constant(np.pi)
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
#    + (Phib*Phib + Phi*Phi*mur*mur)*fx*v_4*dx + N.dx(0)*v_4.dx(1)*dx \
#    - g1*v_4*ds(1)
    # - g1*v_4*ds(2) - g1*v_4*ds(1)
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
#    bc7 = DirichletBC(V.sub(3), Constant(Nf) , circlebord)  # on N
#    bc8 = DirichletBC(V.sub(3), Constant(0.) , no_origen)  # on N
###############           Solve variational problem         ###################
    solve(F == 0, u, [bc, bc1, bc2, bc3, bc4, bc6])
#    solve(F == 0, u, [bc, bc1, bc2, bc3, bc4, bc6, bc7, bc8])
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
#    V = FunctionSpace(mesh, MixedElement([P1, P1,P1,P1]))
    V = FunctionSpace(mesh, MixedElement([P1, P1,P1]))
    ###############    Load solution        ###################################
    u = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "r")
    input_file.read(u, "solution")
    input_file.close()
    return u  
################################################################################
#di2R10 = {1 :{"Rf" : 10., "En" : 6.0, "expon" : -2, "fif" : 1e-2},
#          2 :{"Rf" : 10., "En" : 4.5, "expon" : -2, "fif" : 1e-2},
#          3 :{"Rf" : 10., "En" : 4.0, "expon" : -2, "fif" : 1e-2},
#          4 :{"Rf" : 10., "En" : 5.0, "expon" : -3, "fif" : 1e-2},
#          5 :{"Rf" : 10., "En" : 2.5, "expon" : -3, "fif" : 1e-2},
#          6 :{"Rf" : 10., "En" : 2.0, "expon" : -3, "fif" : 1e-2},
#          7 :{"Rf" : 10., "En" : 8.0, "expon" : -4, "fif" : 1e-2},
#          8 :{"Rf" : 10., "En" : 10., "expon" : -5, "fif" : 1e-3}}
di2R10 = {1 :{"Rf" : 10., "En" : 6.0, "expon" : -2, "fif" : 1e-2},
          2 :{"Rf" : 15., "En" : 4.5, "expon" : -2, "fif" : 1e-2},
          3 :{"Rf" : 15., "En" : 4.0, "expon" : -2, "fif" : 1e-2},
          4 :{"Rf" : 30., "En" : 5.0, "expon" : -3, "fif" : 1e-2},
          5 :{"Rf" : 30., "En" : 2.5, "expon" : -3, "fif" : 1e-2},
          6 :{"Rf" : 30., "En" : 2.0, "expon" : -3, "fif" : 1e-2},
          7 :{"Rf" : 30., "En" : 8.0, "expon" : -4, "fif" : 1e-2},
          8 :{"Rf" : 30., "En" : 10., "expon" : -5, "fif" : 1e-3},
          9 : {"Rf" : 30., "En" : 8., "expon" : -5, "fif" : 1e-3},
          10 :{"Rf" : 30., "En" : 2., "expon" : -5, "fif" : 1e-3},
          11 :{"Rf" : 30., "En" : 0.8, "expon" : -5, "fif" : 1e-3},
          12 :{"Rf" : 30., "En" : 0.5, "expon" : -5, "fif" : 1e-3},
          13 :{"Rf" : 30., "En" : 1., "expon" : -6, "fif" : 1e-3},
          14 :{"Rf" : 30., "En" : 0.8, "expon" : -6, "fif" : 1e-3},
          15 :{"Rf" : 30., "En" : 0.5, "expon" : -6, "fif" : 1e-3},
          16 :{"Rf" : 30., "En" : 1., "expon" : -8, "fif" : 1e-3},
          17 :{"Rf" : 30., "En" : 0.8, "expon" : -8, "fif" : 1e-3},
          18 :{"Rf" : 30., "En" : 0.5, "expon" : -8, "fif" : 1e-3}}
#################################################################################
#dec = '/home/jordi/satellite/schrodinger_poisson_mix'


da2 = float(input(''))
da2 = int(da2)
# da = float(input(""))
# da = int(da)

#dec = '/home/jordi/satellite/schrodinger_poisson_mix/jaja'
dec = '/home/jordi/satellite/schrodinger_poisson_mix/base2'

ncor = da2
# for ncor in range(9,13,1): 
    
# for ncor in (1,4, 5,7,10,15):
#for ncor in (3,5,7,10,15):

Rf = di2R10[ncor]["Rf"] ### para corridas 200 y 300
expone = di2R10[ncor]["expon"]
Ee = di2R10[ncor]["En"]

da = 1036
Nf = da/1000.
Vf = -Nf/Rf
murel = 1.

#Vi = -2.6431
#Eb = 1.25
#exponb = -2
#Enerb = Eb*10**exponb
#Enere = Ee*10**expone
#
## ncorr = ncor + 400  ## con ecuacion de N correcta
#ncorr = ncor
#print('ncor=',ncorr, 'Rf=', Rf, 'Ee=',Ee, 'expe=',expone, 'Vf=',Vf)
#phi0=np.sqrt(0.99916) # carpeta jaja
##phi0=0.5 #carpeta jaja2

####           base 2       #####################################
Eb = 4.79
exponb = -3
Enerb = Eb*10**exponb
Enere = Ee*10**expone
ncorr = ncor
print('ncor=',ncorr, 'Rf=', Rf, 'Ee=',Ee, 'expe=',expone, 'Vf=',Vf)
phi0=0.98
Vi = -1.12077695
###############################################################################

# mainprog(ncorr, Rf= Rf, Ee= Enere, Vf = Vf , fife = 1e-2, #no funciona con fife = 1e-5 ni 1e-3
#         Eb = Enerb, Phicero= phi0, R0 = 0.001,
#         Vi = Vi, fifb= 1e-5, mur = murel, Nf = Nf, 
#         direct= '%s/%s'%(dec,da))
  
u = read_saved_filen(ncorr, direct = "%s/%s"%(dec,da))
u.set_allow_extrapolation(True)
data = [Eb, exponb, Ee, expone]
plots_pyplot_mix(u, Rf, ncorr, data, show = False, ruta = '%s/%s'%(dec,da),
                  R0 = 0.001)
#
psi, phi, phib, N = u.split()
#graf(N, r'$\int (\Phi_0^2 + \mu_r^2 \Phi_1^2) \hat{\rho}d\hat{\rho} d\hat{z}$', 
#      '%s/%s'%(dec,da), name='N_%d'%ncorr, zlabel= r'$N$',rango = True, 
#      vmin=0, vmax=Nf)

#####################        number of particles       ########################
# fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0)

# ene = fx*(phib*phib + phi*phi*murel*murel)*dx  # corregida
# Nn = assemble(ene)
# eneb = fx*phib*phib*dx   ## corregida
# Nnb = assemble(eneb)
# enee = fx*phi*phi*murel*murel*dx ###corregida
# Nne = assemble(enee)
# ###########        W energy      ##########################    
# we = 2.*np.pi*phi*phi*psi*dx
# We = assemble(we)
# wb = 2.*np.pi*phib*phib*psi*dx
# Wb = assemble(wb)
# ###########          K energy      ########################## 
# V = phi.function_space()
# mesh = V.mesh()
# fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0, domain=mesh)
# kee = -2.*np.pi*phi*div(fx*grad(phi))*dx
# Ke = assemble(kee)

# Vb = phib.function_space()
# meshb = Vb.mesh()
# fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0, domain=meshb)
# keb = -2.*np.pi*phib*div(fx*grad(phib))*dx
# Kb = assemble(keb)  

# psi.set_allow_extrapolation(True)
# rf = Rf
# thf = np.pi/4
# rhof = rf*np.sin(thf)
# zf = rf*np.cos(thf)

# f= open("%s/%s/cantidades_%d.txt"%(dec,da,ncorr),"w+")
# f.write("Ni = %f, murel = %f,  E_t = %f \r\n " %(Nf,murel,(Eb*Nnb + Ee*Nne)/Nn))
# f.write(" N = %f, N/rf = %f , Nb = %f, Ne = %f \r\n " % (Nn, -Nn/Rf, Nnb, Nne))
# f.write(" Psi(Rf,0)= %f, Psi(rf,pi/4) = %f \r\n " % (psi(Rf,0), psi(rhof,zf)))
# f.write("Phi0=%f \r\n"%phi0)
# f.write(" Wb = %f, Kb = %f, Kb/Wb = %f \r\n " % (Wb, Kb, Kb/abs(Wb)))
# f.write(" We = %f, Ke = %f, Ke/We = %f \r\n " % (We, Ke, Ke/abs(We)))
# f.write(" WT = %f, KT = %f, KT/WT = %f \r\n " % ((Wb+We), (Kb+Ke), (Kb+Ke)/abs(Wb+We)))
# f.close()
# # #############################################################################
# # ###############################################################################
